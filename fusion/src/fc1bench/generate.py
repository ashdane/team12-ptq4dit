"""Generate matched pretrained DiT images for ADM evaluation (one mode/process).

Quantized modes use custom symmetric per-token A8 and per-channel W8/W4 on
attention/MLP projections. This is not calibrated PTQ4DiT CSB/SSC/AdaRound.
"""
import argparse
from dataclasses import asdict
import json
from pathlib import Path
import time

import numpy as np
import torch

from .benchmark import environment
from .eval_utils import git_record, matched_rng, model_storage, precision
from .full_model import load_checkpoint, load_official
from .model_layers import convert, projection_names, TunedLinear
from .optimized import Config
from .quality_data import batch_ranges, pack_shards, sha256, write_json
from .quantization import metrics
from .tuning import Bank


class DefaultBank:
    """Portable explicit configurations, not claimed to be tuned."""
    def get(self, m, n, k, bits, dtype):
        return Config(bm=64, bn=128, bk=64, warps=4, stages=1,
                      layout='kn_packed' if bits == 4 else 'nk')


def check_conversions(model, names):
    """Check one real-input invocation per converted projection before sampling."""
    rows, handles = [], []
    def make_hook(name):
        def hook(module, inputs, output):
            reference = module.reference(inputs[0])
            # Under autocast, LayerNorm emits FP32, so projections often see FP32 inputs
            # even when QUANT_DTYPE is float16. The FP32 F.linear reference itself
            # carries K-dependent summation rounding (K=4608 for mlp.fc2, outputs ~1e3),
            # observed up to ~2.3e-5 relative on real DiT-XL/2 activations; 1e-4 leaves
            # headroom while remaining far below INT8/INT4 quantization error (~1e-2).
            rtol = {torch.float32: 1e-4, torch.float16: 1e-3, torch.bfloat16: 1e-2}[output.dtype]
            torch.testing.assert_close(output.float(), reference.float(), rtol=rtol, atol=2e-3)
            rows.append(dict(layer=name, **metrics(reference, output)))
        return hook
    for name in names:
        if isinstance(model.get_submodule(name), TunedLinear):
            handles.append(model.get_submodule(name).register_forward_hook(make_hook(name)))
    return handles, rows


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--dit-repo', required=True)
    p.add_argument('--checkpoint', required=True)
    p.add_argument('--output', required=True, help='Shared root; each mode (or --label) gets its own directory')
    p.add_argument('--label', help='Directory name under --output (default: the mode); lets several INT8 '
                                   'variants share one root, e.g. int8_v4c')
    p.add_argument('--mode', choices=['fp32','fp16','int8','int4','fake8','fake4'], required=True)
    p.add_argument('--image-size', type=int, choices=[256,512], default=256)
    p.add_argument('--num-samples', type=int, default=10000)
    p.add_argument('--batch-size', type=int, default=4)
    p.add_argument('--steps', type=int, choices=[50,100,250], default=50)
    p.add_argument('--cfg', type=float, default=1.5)
    p.add_argument('--seed', type=int, default=1)
    p.add_argument('--scope', choices=['all','mlp'], default='all')
    p.add_argument('--quant-dtype', choices=['float32','float16','bfloat16'], default='float16')
    p.add_argument('--bank', help='Optional tuning bank from this GPU/software/source and exact row shape')
    p.add_argument('--vae', default='stabilityai/sd-vae-ft-ema')
    p.add_argument('--vae-revision', required=True, help='Pinned Hugging Face commit or local revision identifier')
    p.add_argument('--scaffold', help='latency_bench scaffold to generate with (e.g. sdpa_graph_pure16_compile_tuned_fused); '
                                      'default: legacy autocast + explicit attention path')
    p.add_argument('--tf32', action='store_true', help='Allow TF32 matmuls (official DiT/PTQ4DiT FP32 sampling setting)')
    p.add_argument('--labels', help='Comma-separated ImageNet class ids used for every batch, in order and repeated to '
                                    'fill it (default: random labels from the batch seed). Noise is unchanged.')
    p.add_argument('--save-png', action='store_true', help='Also save every image as img_<index>_c<class>.png')
    args = p.parse_args(argv)
    if min(args.num_samples, args.batch_size) < 1 or not np.isfinite(args.cfg) or args.cfg < 1:
        p.error('Counts must be positive and finite cfg >= 1')
    if not torch.cuda.is_available() or torch.cuda.get_device_capability() < (7,5):
        p.error('An NVIDIA CUDA GPU with compute capability >= 7.5 is required')
    torch.set_grad_enabled(False)
    torch.backends.cuda.matmul.allow_tf32 = args.tf32
    torch.backends.cudnn.allow_tf32 = args.tf32
    torch.backends.cudnn.benchmark = False
    torch.set_float32_matmul_precision('high' if args.tf32 else 'highest')
    scaffold = None
    if args.scaffold:
        from .latency_bench import Scaffold
        scaffold = Scaffold.parse(args.scaffold)
    label = args.label or args.mode
    if not label.replace('_', '').isalnum():
        p.error('--label must be alphanumeric with underscores')
    out = Path(args.output).resolve()/label
    out.mkdir(parents=True, exist_ok=True)
    manifest_path = out/'manifest.json'
    dtype = {'fp32':torch.float32, 'fp16':torch.float16}.get(args.mode, getattr(torch,args.quant_dtype))
    manifest = dict(mode=args.mode, label=label, checkpoint_sha256=sha256(args.checkpoint),
                    dit_commit=git_record(args.dit_repo), image_size=args.image_size,
                    num_samples=args.num_samples, batch_size=args.batch_size, steps=args.steps,
                    cfg=args.cfg, seed=args.seed, scope=args.scope, vae=args.vae,
                    vae_revision=args.vae_revision, dtype=str(dtype), fixed_labels=args.labels,
                    source_sha256={f.name:sha256(f) for f in sorted(Path(__file__).parent.glob('*.py'))},
                    attention=('explicit timm attention; fused_attn=False' if scaffold is None or scaffold.attention != 'sdpa'
                               else 'scaled_dot_product_attention (timm fused_attn=True)'),
                    sampler='official ancestral DDPM; matched per-batch latent, label and noise seeds',
                    method='pretrained floating point' if args.mode.startswith('fp') else
                           ('custom symmetric absmax W8/W4 + per-token A8; fused producer+quant kernels (fused_ops)'
                            if scaffold is not None and scaffold.fused else
                            'custom symmetric absmax W8/W4 + per-token A8; no CSB/SSC/AdaRound'),
                    configuration='portable fixed tiles' if not args.bank else sha256(args.bank),
                    scaffold=args.scaffold, tf32=args.tf32,
                    environment=environment(torch.device('cuda')))
    if manifest_path.exists():
        old = json.loads(manifest_path.read_text())
        for key,value in manifest.items():
            if old.get(key) != value:
                # Clock readings may change on resume; check stable GPU/software fields instead.
                if key == 'environment':
                    for field in ('gpu','cuda','packages'):
                        if old[key].get(field) != value.get(field):
                            raise ValueError(f'Resume environment mismatch: {field}')
                else:
                    raise ValueError(f'Resume configuration mismatch: {key}; choose a new output root')
        if old.get('status') == 'complete' and (out/'samples.npz').exists():
            if sha256(out/'samples.npz') != old['samples_sha256']:
                raise ValueError('Completed sample archive checksum mismatch')
            print(f'Already complete: {out}', flush=True)
            return
    manifest['status'] = 'running'
    write_json(manifest_path, manifest)
    try:
        if scaffold is not None:
            generate_with_scaffold(args, scaffold, out, manifest, manifest_path)
            return
        models, diffusion_module = load_official(args.dit_repo)
        state = load_checkpoint(args.checkpoint)
        model = models.DiT_models['DiT-XL/2'](input_size=args.image_size//8).eval()
        model.load_state_dict(state, strict=True)
        for block in model.blocks:
            if hasattr(block.attn,'fused_attn'):
                block.attn.fused_attn=False
        model = model.to(device='cuda', dtype=dtype)
        names = projection_names(model,args.scope)
        rows = args.batch_size*2*(args.image_size//16)**2
        bank = Bank(args.bank) if args.bank else DefaultBank()
        manifest['conversions'] = []
        if not args.mode.startswith('fp'):
            manifest['conversions'] = convert(model,names,state,bank,rows,int(args.mode[-1]),dtype,
                                               fake=args.mode.startswith('fake'))
        del state
        manifest['model_storage_bytes'] = model_storage(model)
        diffusion = diffusion_module.create_diffusion(str(args.steps))
        from diffusers import AutoencoderKL
        vae = AutoencoderKL.from_pretrained(args.vae, revision=args.vae_revision).eval().cuda().float()
        shards = out/'batches'
        shards.mkdir(exist_ok=True)
        def forward(x,t,y,cfg_scale):
            with precision(dtype):
                return model.forward_with_cfg(x,t,y,cfg_scale=cfg_scale)
        # Compile and check actual layer inputs, using real weights. No timed random-weight benchmark.
        with torch.inference_mode(), matched_rng(args.seed):
            z = torch.randn(args.batch_size,4,args.image_size//8,args.image_size//8,device='cuda')
            z = torch.cat([z,z])
            y = torch.cat([torch.arange(args.batch_size,device='cuda')%1000,
                           torch.full((args.batch_size,),1000,device='cuda',dtype=torch.long)])
            t = torch.full((args.batch_size*2,),999,device='cuda',dtype=torch.long)
            handles, checks = check_conversions(model,names)
            try:
                result = forward(z,t,y,args.cfg)
                if not torch.isfinite(result).all():
                    raise ValueError('Nonfinite warmup output')
            finally:
                for handle in handles: handle.remove()
            write_json(out/'implementation_checks.json',checks)
            del result,z,y,t
        _generate_batches(args, out, manifest, manifest_path, diffusion, vae, forward)
    except Exception as exc:
        manifest.update(status='failed',error=f'{type(exc).__name__}: {exc}')
        write_json(manifest_path,manifest)
        raise
    print(f'Completed {out}',flush=True)


def generate_with_scaffold(args, scaffold, out, manifest, manifest_path):
    """Generate with exactly the configuration latency_bench times (graphs, compile, fused, tuned)."""
    from types import SimpleNamespace
    from .latency_bench import build, make_forward, make_graphed, make_inputs, patch_timestep_embedding
    models, diffusion_module = load_official(args.dit_repo)
    patch_timestep_embedding(models)
    state = load_checkpoint(args.checkpoint)
    bank = Bank(args.bank) if args.bank else None
    ns = SimpleNamespace(image_size=args.image_size, scope=args.scope)
    with torch.inference_mode():
        model, dtype, _, conversions = build(models, state, ns, args.mode, args.batch_size, scaffold, bank)
        manifest['conversions'] = conversions
        del state
        manifest['model_storage_bytes'] = model_storage(model)
        fwd = make_forward(model, dtype, args.cfg, scaffold)
        x, t, y = make_inputs(args.batch_size, args.image_size, args.seed)
        warm = fwd(x, t, y)
        if not torch.isfinite(warm).all():
            raise ValueError('Nonfinite warmup output')
        if scaffold.graphs:
            replay, _graph = make_graphed(fwd, x, t, y)
            # .float() makes a new tensor: the graph's static output is overwritten by the next replay.
            model_fn = lambda xx, tt, y=None, cfg_scale=None: replay(xx, tt, y).float()
        else:
            model_fn = lambda xx, tt, y=None, cfg_scale=None: fwd(xx, tt, y).float()
    write_json(out/'implementation_checks.json',
               dict(note='scaffold generation: kernel correctness covered by tests/test_fused_ops.py, '
                         'tests/test_optimized.py and the tuner\'s exact INT32 checks', scaffold=args.scaffold))
    diffusion = diffusion_module.create_diffusion(str(args.steps))
    from diffusers import AutoencoderKL
    vae = AutoencoderKL.from_pretrained(args.vae, revision=args.vae_revision).eval().cuda().float()
    _generate_batches(args, out, manifest, manifest_path, diffusion, vae, model_fn)
    print(f'Completed {out}', flush=True)


def _generate_batches(args, out, manifest, manifest_path, diffusion, vae, forward):
    shards = out/'batches'
    shards.mkdir(exist_ok=True)
    if True:
        records = []
        with torch.inference_mode():
            for start,n in batch_ranges(args.num_samples,args.batch_size):
                image_path, record_path = shards/f'{start:06d}.npy', shards/f'{start:06d}.json'
                if image_path.exists() and record_path.exists():
                    record = json.loads(record_path.read_text())
                    a = np.load(image_path,mmap_mode='r',allow_pickle=False)
                    if (a.shape != (n,args.image_size,args.image_size,3) or a.dtype != np.uint8
                            or sha256(image_path) != record['sha256']):
                        raise ValueError(f'Corrupt saved batch {start}')
                    records.append(record)
                    continue
                # Fixed batch shape allows a tuned bank; final batch is trimmed after generation.
                with matched_rng(args.seed + start):
                    z = torch.randn(args.batch_size,4,args.image_size//8,args.image_size//8,device='cuda')
                    labels = torch.randint(0,1000,(args.batch_size,),device='cuda')   # keeps the RNG stream
                    if args.labels:
                        fixed = [int(c) for c in args.labels.split(',')]
                        labels = torch.tensor((fixed * args.batch_size)[:args.batch_size], device='cuda')
                    y = torch.cat([labels, torch.full_like(labels,1000)])
                    z = torch.cat([z,z])
                    torch.cuda.synchronize()
                    torch.cuda.reset_peak_memory_stats()
                    begin = time.perf_counter()
                    latent = diffusion.p_sample_loop(forward,z.shape,noise=z,clip_denoised=False,
                        model_kwargs=dict(y=y,cfg_scale=args.cfg),device='cuda',progress=False)[:args.batch_size]
                    torch.cuda.synchronize()
                    sampling_s = time.perf_counter()-begin
                    if not torch.isfinite(latent).all():
                        raise ValueError(f'Nonfinite latents at image {start}')
                    begin = time.perf_counter()
                    decoded = vae.decode(latent.float()/0.18215).sample
                    if not torch.isfinite(decoded).all():
                        raise ValueError(f'Nonfinite decoded images at image {start}')
                    images = (decoded*127.5+128).clamp(0,255).permute(0,2,3,1).to('cpu',torch.uint8).numpy()[:n]
                    decode_s = time.perf_counter()-begin
                    temp = image_path.with_suffix('.tmp')
                    with temp.open('wb') as stream: np.save(stream,images,allow_pickle=False)
                    temp.replace(image_path)
                    record = dict(start=start,count=n,generated_batch_size=args.batch_size,
                                  seed=args.seed+start,labels=labels[:n].cpu().tolist(),
                                  sampling_s=sampling_s,decode_and_transfer_s=decode_s,
                                  peak_allocated_bytes=torch.cuda.max_memory_allocated(),sha256=sha256(image_path))
                    write_json(record_path,record)
                    records.append(record)
                    if start == 0:
                        from PIL import Image
                        for i,img in enumerate(images[:4]): Image.fromarray(img).save(out/f'preview_{i}.png')
                    if args.save_png:
                        from PIL import Image
                        for i,img in enumerate(images):
                            Image.fromarray(img).save(out/f'img_{start+i:05d}_c{int(labels[i]):03d}.png')
                    del z,y,labels,latent,decoded,images
                print(f'{args.mode}: {start+n}/{args.num_samples} images; sampling {sampling_s:.2f}s/batch',flush=True)
                manifest['completed_samples'] = start+n
                write_json(manifest_path,manifest)
        pack_shards(shards,out/'samples.npz',args.num_samples,args.batch_size,args.image_size)
        manifest.update(status='complete',completed_samples=args.num_samples,
                        samples_sha256=sha256(out/'samples.npz'),batches=records,
                        sampling_seconds=sum(r['sampling_s'] for r in records),
                        timing_note='Warm generation, excludes download/setup/packing; VAE resident. Final padded batch included.')
        write_json(manifest_path,manifest)


if __name__ == '__main__':
    main()
