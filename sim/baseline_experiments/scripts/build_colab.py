"""Build a standalone, source-embedded Colab notebook and shareable source ZIP."""
import base64
import io
import json
from pathlib import Path
import zipfile

ROOT = Path(__file__).resolve().parents[1]


def cell(kind, source):
    result = dict(cell_type=kind, metadata={}, source=source.splitlines(keepends=True))
    if kind == "code":
        result.update(execution_count=None, outputs=[])
    return result


def build():
    paths = sorted([*ROOT.glob("fc1bench/*.py"), *ROOT.glob("tests/*.py"),
                    ROOT / "README.md", ROOT / "requirements.txt", ROOT / "agent.md"])
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as archive:
        for path in paths:
            info = zipfile.ZipInfo(str(path.relative_to(ROOT)), date_time=(2026, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            archive.writestr(info, path.read_bytes())
    payload = base64.b64encode(buffer.getvalue()).decode()
    bootstrap = '''import base64, io, os, pathlib, zipfile
ROOT = (pathlib.Path('/content') if pathlib.Path('/content').exists() else pathlib.Path.cwd()) / 'fc1_experiment'
ROOT.mkdir(parents=True, exist_ok=True)
PAYLOAD = """''' + payload + '''"""
with zipfile.ZipFile(io.BytesIO(base64.b64decode(PAYLOAD))) as archive:
    archive.extractall(ROOT)
os.chdir(ROOT)
print('Source extracted to', ROOT)
'''
    cells = [
        cell("markdown", """# FC1 baseline and integer experiments — A100 / L40

Select **Runtime → Change runtime type → GPU**, then run cells in order.
This notebook embeds all source files. No model download is required for the
synthetic FC1 experiment. A100 and L40 are supported targets; the notebook checks
which GPU is actually allocated. An L40 can be used through a compatible local
runtime. Read `README.md` for the full measurement contract.

Default FC1: M=2048, K=1152, N=4608. Integer kernels are experimental symmetric
quantization, **not calibrated PTQ4DiT**. W4 uses packed storage and in-kernel
INT8 expansion. GPU tests and exact accumulator checks gate benchmark success.
No speedup or image-quality result has been established on the development host.
"""),
        cell("code", bootstrap),
        cell("code", """import importlib.util, subprocess, sys, torch
assert torch.cuda.is_available(), 'Select a CUDA GPU runtime first.'
assert importlib.util.find_spec('triton'), 'Use CUDA-enabled PyTorch with its matching Triton package.'
import triton
print('GPU:', torch.cuda.get_device_name(0))
print('Compute capability:', torch.cuda.get_device_capability(0))
print('PyTorch:', torch.__version__, 'CUDA:', torch.version.cuda, 'Triton:', triton.__version__)
assert torch.cuda.get_device_capability(0)[0] >= 8, 'This experiment targets Ampere/Ada or newer.'
subprocess.run(['nvidia-smi'], check=True)
"""),
        cell("markdown", "## 1. Validate before measuring\nTests cover signed nibble packing, tails, odd K, exact integer accumulators, rounding, bias, and layer integration. A failure stops this cell; resolve it before continuing."),
        cell("code", """def run_logged(command, log_path):
    # Stream child output through notebook Python so Colab/Jupyter displays it.
    log_path = pathlib.Path(log_path)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open('w') as log:
        with subprocess.Popen(command, cwd=ROOT, stdout=subprocess.PIPE,
                              stderr=subprocess.STDOUT, text=True, bufsize=1) as process:
            for line in process.stdout:
                print(line, end='', flush=True)
                log.write(line)
                log.flush()
            returncode = process.wait()
    if returncode:
        raise RuntimeError(f'Command failed (exit {returncode}). Full output is above and in {log_path}.')

print('Python:', sys.executable, sys.version)
print('PyTorch:', torch.__version__, 'CUDA:', torch.version.cuda, 'Triton:', triton.__version__)
print('GPU:', torch.cuda.get_device_name(0))
run_logged([sys.executable, '-u', '-m', 'unittest', 'discover', '-s', 'tests', '-v'],
           ROOT / 'validation.log')
"""),
        cell("markdown", "## 2. Run FC1 experiments\nWeight preparation and compilation are outside the timed region. The integer reference uses FP64 for exact checks and may take extra time on L40. Set `RUN_INTEGER=False` to run only the baseline requested first in agent.md."),
        cell("code", """from datetime import datetime, timezone
RUN_INTEGER = True
PROFILE = True
RUN = ROOT / 'results' / datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S_%f')
modes = ['fp32', 'fp16', 'bf16'] + (['int8', 'int4'] if RUN_INTEGER else [])
command = [sys.executable, '-m', 'fc1bench.benchmark', '--modes', *modes,
           '--warmup', '20', '--repeats', '50', '--inner', '10', '--output', str(RUN)]
if PROFILE:
    command.append('--profile')
run_logged(command, RUN / 'benchmark.log')
(RUN / 'pip-freeze.txt').write_text(subprocess.check_output([sys.executable, '-m', 'pip', 'freeze'], text=True))
"""),
        cell("code", """import json
import pandas as pd
from IPython.display import display
table = pd.read_csv(RUN / 'results.csv')
display(table[[c for c in ['mode', 'status', 'core_cuda_event_median_ms',
                         'layer_cuda_event_median_ms', 'relative_l2', 'mse',
                         'max_abs_error', 'weight_storage_bytes',
                         'incremental_peak_allocated_bytes'] if c in table.columns]])
report = json.loads((RUN / 'results.json').read_text())
for result in report['results']:
    if result['status'] != 'ok':
        print(result)
    if 'assembly' in result.get('metadata', {}):
        print(result['mode'], result['metadata']['assembly'])
"""),
        cell("markdown", """## 3. Actual upstream fake-quant operator

Clones the official PTQ4DiT repository and imports its `QuantModule`.
`PTQ4DIT_REV=None` resolves the current checkout and records the exact commit.
Set it to that SHA to repeat the experiment. This initializes an isolated
operator using `max` scales; it does not load a CSB/SSC/AdaRound calibration.
"""),
        cell("code", """RUN_UPSTREAM = True
PTQ4DIT_REV = None  # Set to a recorded commit SHA for a repeatable upstream checkout.
if RUN_UPSTREAM:
    upstream = ROOT / 'third_party' / 'PTQ4DiT'
    upstream.parent.mkdir(exist_ok=True)
    if not upstream.exists():
        subprocess.run(['git', 'clone', 'https://github.com/adreamwu/PTQ4DiT.git', str(upstream)], check=True)
    if PTQ4DIT_REV:
        subprocess.run(['git', '-C', str(upstream), 'checkout', '--detach', PTQ4DIT_REV], check=True)
    print('Upstream commit:', subprocess.check_output(['git', '-C', str(upstream), 'rev-parse', 'HEAD'], text=True))
    upstream_out = RUN / ('upstream_' + datetime.now(timezone.utc).strftime('%H%M%S_%f'))
    run_logged([sys.executable, '-m', 'fc1bench.benchmark', '--ptq4dit-repo', str(upstream),
                    '--modes', 'fp32', 'fp16', 'bf16', 'fake_w8a8', 'fake_w4a8',
                    '--output', str(upstream_out)], upstream_out / 'benchmark.log')
    display(pd.read_csv(upstream_out / 'results.csv'))
"""),
        cell("markdown", """## 4. Full pretrained DiT-XL/2 evaluation & real-activation kernel tuning

Clones `facebookresearch/DiT`, downloads the official pretrained
`DiT-XL-2-256x256.pt` checkpoint, captures real FC1 (`2048x4608x1152`) and FC2
(`2048x1152x4608`) activations during the `fp32` pass, runs randomized
interleaved kernel tuning + holdout evaluation (`nk`, `kn_packed` with
`tl.interleave`, and `torch._int_mm`), evaluates `fp32`, `bf16`, `fp16`,
`fake8`, `int8`, `fake4`, and `int4` across matched 50-step ancestral sampling
trajectories, and decodes the generated latents with the SD VAE.

If `RUN_FULL_DIT = False`, standalone synthetic FC1/FC2 tuning runs instead.
"""),
        cell("code", """RUN_FULL_DIT = True
DECODE_VAE = True
DIT_SCOPE = 'mlp'  # 'mlp' (fc1+fc2) or 'all' (qkv+proj+fc1+fc2)
if RUN_FULL_DIT:
    subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', 'timm', 'diffusers', 'accelerate'], check=True)
    dit_repo = ROOT / 'third_party' / 'DiT'
    dit_repo.parent.mkdir(exist_ok=True)
    if not dit_repo.exists():
        subprocess.run(['git', 'clone', 'https://github.com/facebookresearch/DiT.git', str(dit_repo)], check=True)
    ckpt = ROOT / 'checkpoints' / 'DiT-XL-2-256x256.pt'
    ckpt.parent.mkdir(exist_ok=True)
    if not ckpt.exists():
        subprocess.run(['wget', '-nv', '-O', str(ckpt),
                        'https://dl.fbaipublicfiles.com/DiT/models/DiT-XL-2-256x256.pt'], check=True)
    dit_out = RUN / 'full_dit'
    cmd = [sys.executable, '-m', 'fc1bench.full_model', '--dit-repo', str(dit_repo),
           '--checkpoint', str(ckpt), '--scope', DIT_SCOPE, '--quant-dtype', 'bfloat16',
           '--profile', '--output', str(dit_out)]
    if DECODE_VAE:
        cmd.append('--decode')
    run_logged(cmd, dit_out / 'full_model_stdout.log')
    display(pd.read_csv(dit_out / 'tuning' / 'summary.csv'))
    display(pd.read_csv(dit_out / 'summary.csv'))
else:
    tuning_out = RUN / 'tuning'
    run_logged([sys.executable, '-m', 'fc1bench.tuning',
                '--shapes', '2048x4608x1152', '2048x1152x4608',
                '--dtype', 'bfloat16', '--rounds', '10', '--inner', '5',
                '--output', str(tuning_out)], tuning_out / 'tuning.log')
    display(pd.read_csv(tuning_out / 'summary.csv'))
"""),
        cell("markdown", "## 5. Download results and implementation\nThe archive contains raw timings, metadata, CSV, quantization state, PTX/cubin, full-model trajectories, decoded images, and traces. Temporary 1.26 GB per-layer FP32 reference dumps (`reference_layer_outputs`) are removed after `layer_errors.csv` is written so the archive stays small enough for browser download."),
        cell("code", """import shutil
for temp_dir in (RUN / 'full_dit' / 'reference_layer_outputs',
                 RUN / 'full_dit' / 'kernel_inputs'):
    if temp_dir.exists():
        shutil.rmtree(temp_dir)
for name in ('fc1bench', 'tests'):
    shutil.copytree(ROOT / name, RUN / 'source' / name, dirs_exist_ok=True,
                    ignore=shutil.ignore_patterns('__pycache__'))
shutil.copy2(ROOT / 'README.md', RUN / 'source' / 'README.md')
shutil.copy2(ROOT / 'validation.log', RUN / 'validation.log')
archive_path = shutil.make_archive(str(RUN) + '_artifacts', 'zip', RUN)
print(archive_path)
try:
    from google.colab import files
    files.download(archive_path)
except ImportError:
    print('Download the archive from your Jupyter file browser.')
"""),
    ]
    notebook = dict(cells=cells, metadata=dict(accelerator="GPU",
                    kernelspec=dict(display_name="Python 3", language="python", name="python3"),
                    language_info=dict(name="python")), nbformat=4, nbformat_minor=5)
    for index, item in enumerate(cells):
        item["id"] = f"fc1-{index:02d}"
    target = ROOT / "notebooks" / "FC1_Colab.ipynb"
    target.parent.mkdir(exist_ok=True)
    target.write_text(json.dumps(notebook, indent=1) + "\n")
    with zipfile.ZipFile(ROOT / "fc1_colab_bundle.zip", "w", zipfile.ZIP_DEFLATED) as archive:
        for path in paths + [target, Path(__file__)]:
            archive.write(path, str(path.relative_to(ROOT)))
    print(target)
    print(ROOT / "fc1_colab_bundle.zip")


if __name__ == "__main__":
    build()
