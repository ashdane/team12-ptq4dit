"""Evaluate pretrained image archives with the official ADM TensorFlow evaluator.

Run in an isolated TensorFlow environment after generation processes exit.
Reference statistics and images are both required: statistics alone cannot
compute generative Precision. Published PTQ4DiT numbers are contextual rows.
"""
import argparse
import csv
import importlib.util
import json
import math
from pathlib import Path
import subprocess
import sys

from .quality_data import METRICS, deltas, npz_shape, paper_rows, sha256, validate_manifests, write_json


def stable_frechet_distance(mu1, sigma1, mu2, sigma2):
    """Frechet distance via symmetric eigendecompositions.

    Same quantity as ADM/TTUR FID: |mu1-mu2|^2 + tr(S1) + tr(S2) - 2 tr((S1 S2)^1/2),
    using tr((S1 S2)^1/2) = sum sqrt(eig(S1^1/2 S2 S1^1/2)). Stays real and finite when
    a covariance is rank-deficient (fewer samples than feature dimensions), where
    scipy.linalg.sqrtm of the non-symmetric product can blow up.
    """
    import numpy as np
    mu1, mu2 = np.atleast_1d(mu1).astype(np.float64), np.atleast_1d(mu2).astype(np.float64)
    s1, s2 = np.atleast_2d(sigma1).astype(np.float64), np.atleast_2d(sigma2).astype(np.float64)
    s1, s2 = (s1 + s1.T) / 2, (s2 + s2.T) / 2
    w, v = np.linalg.eigh(s1)
    root1 = (v * np.sqrt(np.clip(w, 0, None))) @ v.T
    inner = root1 @ s2 @ root1
    tr_covmean = np.sqrt(np.clip(np.linalg.eigvalsh((inner + inner.T) / 2), 0, None)).sum()
    diff = mu1 - mu2
    return float(diff @ diff + np.trace(s1) + np.trace(s2) - 2 * tr_covmean)


def frechet_with_fallback(stats, ref_stats):
    """Official ADM Frechet distance; stable eigen formula only if ADM rejects the result."""
    try:
        value = float(stats.frechet_distance(ref_stats))
        if math.isfinite(value):
            return value, 'adm'
    except ValueError as exc:
        print(f'ADM Frechet distance failed ({exc}); using stable eigendecomposition fallback', flush=True)
    return stable_frechet_distance(stats.mu, stats.sigma, ref_stats.mu, ref_stats.sigma), 'stable_eigh_fallback'


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', required=True, help='Generation root containing mode directories')
    p.add_argument('--reference', required=True, help='Official ADM ImageNet reference NPZ')
    p.add_argument('--adm-repo', required=True)
    p.add_argument('--modes', nargs='+', default=['fp32','fp16','int8','int4'],
                   help='Generation directories (modes or generate.py --label names); fp32 is the baseline')
    p.add_argument('--feature-batch', type=int, default=32)
    p.add_argument('--manifold-batch', type=int, default=1024)
    p.add_argument('--allow-small-sample', action='store_true', help='Label below-paper sample counts as exploratory')
    args = p.parse_args(argv)
    if min(args.feature_batch,args.manifold_batch)<1:
        p.error('Batch sizes must be positive')
    if 'fp32' not in args.modes or len(set(args.modes)) != len(args.modes):
        p.error('Include fp32 baseline and list each mode once')
    root = Path(args.output).resolve()
    missing = [m for m in args.modes if not (root/m/'manifest.json').exists()]
    if missing:
        p.error(f'No generation manifest for {missing}')
    manifests = [json.loads((root/m/'manifest.json').read_text()) for m in args.modes]
    base = validate_manifests(manifests)
    target = 10000 if base['image_size']==256 else 5000
    exploratory = base['num_samples'] < target
    if exploratory and not args.allow_small_sample:
        p.error(f'The paper uses {target} samples at this resolution. Pass --allow-small-sample for exploratory metrics.')
    if base['num_samples'] < 4:
        p.error('ADM Precision requires at least four generated samples')
    ref = Path(args.reference).resolve()
    ref_shape = npz_shape(ref)
    if ref_shape[1:3] != (base['image_size'],base['image_size']) or ref_shape[0]<4:
        p.error('Reference images have the wrong resolution or too few samples')
    for mode,m in zip(args.modes,manifests):
        samples = root/mode/'samples.npz'
        if npz_shape(samples) != (base['num_samples'],base['image_size'],base['image_size'],3):
            raise ValueError(f'Invalid sample count/shape for {mode}')
        if sha256(samples) != m['samples_sha256']:
            raise ValueError(f'Sample checksum mismatch for {mode}')
    evaluator_path = Path(args.adm_repo).resolve()/'evaluations/evaluator.py'
    commit = subprocess.check_output(['git','-C',args.adm_repo,'rev-parse','HEAD'],text=True).strip()
    reference_hash = sha256(ref)
    report = dict(status='running',protocol='ADM TensorFlow evaluator; ImageNet reference statistics and images',
                  reference=str(ref),reference_sha256=reference_hash,
                  evaluator_commit=commit,evaluator_sha256=sha256(evaluator_path),
                  num_samples=base['num_samples'],image_size=base['image_size'],steps=base['steps'],cfg=base['cfg'],
                  exploratory=exploratory,scope=base['scope'],
                  interpretation='Custom quantization ablation, not a CSB/SSC PTQ4DiT reproduction',
                  paper_sample_count=target,published_reference=paper_rows(base['image_size'],base['steps']),
                  results=[])
    dest = root/'quality'
    dest.mkdir(exist_ok=True)
    write_json(dest/'comparison.json',report)
    # Imported here so archive/provenance validation and --help need no TensorFlow.
    import numpy as np
    import tensorflow.compat.v1 as tf
    tf.disable_eager_execution()
    spec = importlib.util.spec_from_file_location('official_adm_evaluator',evaluator_path)
    adm = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(adm)
    report['metric_environment'] = dict(python=sys.version,numpy=np.__version__,tensorflow=tf.__version__)
    # Place the official downloaded Inception graph in a persistent evaluator cache.
    adm.INCEPTION_V3_PATH = str(dest/'classify_image_graph_def.pb')
    config = tf.ConfigProto(allow_soft_placement=True)
    config.gpu_options.allow_growth=True
    try:
        with tf.Session(config=config) as session:
            evaluator = adm.Evaluator(session,batch_size=args.feature_batch)
            evaluator.manifold_estimator = adm.ManifoldEstimator(
                session,row_batch_size=args.manifold_batch,col_batch_size=args.manifold_batch)
            evaluator.warmup()
            print('Extracting ImageNet reference features...',flush=True)
            ref_acts = evaluator.read_activations(str(ref))
            ref_stats,ref_spatial = evaluator.read_statistics(str(ref),ref_acts)
            for mode,m in zip(args.modes,manifests):
                print(f'Evaluating {mode}...',flush=True)
                path = str(root/mode/'samples.npz')
                acts = evaluator.read_activations(path)
                stats,spatial = evaluator.read_statistics(path,acts)
                precision,recall = evaluator.compute_prec_recall(ref_acts[0],acts[0])
                fid,fid_method = frechet_with_fallback(stats,ref_stats)
                sfid,sfid_method = frechet_with_fallback(spatial,ref_spatial)
                values = dict(FID=fid,sFID=sfid,
                              IS=float(evaluator.compute_inception_score(acts[0])),
                              Precision=float(precision),Recall=float(recall))
                if not all(math.isfinite(v) for v in values.values()):
                    raise ValueError(f'Nonfinite metrics for {mode}')
                row = dict(mode=mode,method=m['method'],scaffold=m.get('scaffold'),**values,
                           FID_method=fid_method,sFID_method=sfid_method,
                           samples_sha256=m['samples_sha256'],model_storage_bytes=m['model_storage_bytes'],
                           sampling_seconds=m['sampling_seconds'],
                           images_per_second=m['num_samples']/m['sampling_seconds'],
                           peak_allocated_bytes=max(r['peak_allocated_bytes'] for r in m['batches']))
                report['results'].append(row)
                write_json(dest/'comparison.json',report)
                print(json.dumps(row),flush=True)
                del acts
        report['results'] = deltas(report['results'])
        report['status']='complete'
        write_json(dest/'comparison.json',report)
        with (dest/'comparison.csv').open('w',newline='') as f:
            writer=csv.DictWriter(f,fieldnames=list(report['results'][0]))
            writer.writeheader();writer.writerows(report['results'])
        lines=['# Pretrained DiT image-quality comparison','',
               f"{base['image_size']}px; {base['steps']} steps; CFG {base['cfg']}; {base['num_samples']} images/mode.",
               'Exploratory small-sample metrics; not paper-comparable.' if exploratory else
               'Paper sample count met or exceeded; method/precision/protocol differences still apply.',
               '', '| Measured mode | FID ↓ | sFID ↓ | IS ↑ | Precision ↑ |',
               '|---|---:|---:|---:|---:|']
        mark = lambda r, k: f"{r[k]:.4f}" + ('†' if r[f'{k}_method'] != 'adm' else '')
        for r in report['results']:
            lines.append(f"| {r['mode']} | {mark(r,'FID')} | {mark(r,'sFID')} | {r['IS']:.4f} | {r['Precision']:.4f} |")
        if any(r[f'{k}_method'] != 'adm' for r in report['results'] for k in ('FID','sFID')):
            lines += ['', '† Official ADM sqrtm failed (fewer samples than the 2048-d Inception features make the '
                      'covariance rank-deficient); value computed with the mathematically equivalent stable '
                      'eigendecomposition formula. Heavily biased at this sample size: compare modes with each other only.']
        lines += ['', 'Published PTQ4DiT reference (different quantization method; not measured here):','',
                  '| Published method | FID ↓ | sFID ↓ | IS ↑ | Precision ↑ |','|---|---:|---:|---:|---:|']
        for r in report['published_reference']:
            lines.append(f"| {r['method']} | {r['FID']:.2f} | {r['sFID']:.2f} | {r['IS']:.2f} | {r['Precision']:.4f} |")
        lines += ['', 'Source: https://arxiv.org/html/2405.16005v2', '',
                  'Quantized modes replace attention/MLP projections according to scope; other operations remain floating point.',
                  'Precision is generative precision. Recall is an additional ADM output, not one of the four paper headline metrics.',
                  'Deltas in CSV/JSON are measured minus FP32. Lower FID/sFID and higher IS/Precision are better.',
                  'Throughput is warmed sampling only; VAE decoding and disk I/O are excluded. Do not infer a statistically stable speedup from one run.']
        (dest/'comparison.md').write_text('\n'.join(lines)+'\n')
    except Exception as exc:
        report.update(status='failed',error=f'{type(exc).__name__}: {exc}')
        write_json(dest/'comparison.json',report)
        raise


if __name__ == '__main__':
    main()
