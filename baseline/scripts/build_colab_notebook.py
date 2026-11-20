#!/usr/bin/env python3
"""Build the self-contained Team12_Triton_Kernels_Colab.ipynb notebook."""
import base64
import hashlib
import io
import json
from pathlib import Path
import zipfile

ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK_PATH = ROOT / "Team12_Triton_Kernels_Colab.ipynb"
ZIP_TIMESTAMP = (2026, 1, 1, 0, 0, 0)


def bundled_files():
    paths = [
        ROOT / "README.md",
        ROOT / "README_TRITON_KERNELS.md",
        ROOT / "requirements-colab.txt",
        ROOT / "requirements-adm.txt",
    ]
    for folder, patterns in (
        ("scripts", ("*.py",)),
        ("src/fc1bench", ("*.py",)),
        ("tests", ("*.py",)),
        ("validation", ("*.py", "*.sh", "*.md")),
    ):
        base = ROOT / folder
        for pattern in patterns:
            paths.extend(sorted(base.glob(pattern)))
    unique = sorted({p.resolve() for p in paths if p.exists()}, key=lambda p: p.relative_to(ROOT).as_posix())
    return unique


def build_payload():
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for path in bundled_files():
            rel = path.relative_to(ROOT).as_posix()
            info = zipfile.ZipInfo(rel, date_time=ZIP_TIMESTAMP)
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = 0o644 << 16
            archive.writestr(info, path.read_bytes())
    raw = buffer.getvalue()
    return base64.b64encode(raw).decode("ascii"), hashlib.sha256(raw).hexdigest()


def markdown_cell(cell_id, text):
    return {
        "cell_type": "markdown",
        "id": cell_id,
        "metadata": {},
        "source": text,
    }


def code_cell(cell_id, text):
    return {
        "cell_type": "code",
        "id": cell_id,
        "metadata": {},
        "execution_count": None,
        "outputs": [],
        "source": text,
    }


def build_notebook():
    payload, sha = build_payload()
    cells = [
        markdown_cell(
            "cell-00",
            """# Team 12 — Pretrained DiT-XL/2 & Triton Quantization Evaluation on Google Colab

This self-contained notebook bundles the repository source, downloads **official pretrained DiT-XL/2 weights** (`facebookresearch/DiT`), generates actual ImageNet-class images across modes, and evaluates **FID ↓, sFID ↓, Inception Score (IS) ↑, and generative Precision ↑** using OpenAI's official **ADM TensorFlow evaluator** (`guided-diffusion/evaluations/evaluator.py`). It also records generation latency, model storage, GPU peak allocation, and per-layer implementation checks against floating-point fake quantization.

### Paper & Accuracy Metrics Reference
- **Paper**: [PTQ4DiT: Post-training Quantization for Diffusion Transformers (NeurIPS 2024)](https://arxiv.org/html/2405.16005v2), Section 5.1 and Tables 1–2 ([official code](https://github.com/adreamwu/PTQ4DiT)).
- **Evaluation Protocol**: **10,000 generated images at 256×256** (or **5,000 at 512×512**), across **250, 100, or 50 DDPM sampling steps** with classifier-free guidance **CFG = 1.5**, evaluated against the [official ADM ImageNet reference archives](https://github.com/openai/guided-diffusion/tree/main/evaluations) (`VIRTUAL_imagenet256_labeled.npz` / `VIRTUAL_imagenet512.npz`).
- **Metrics**:
  - **FID ↓** (Fréchet Inception Distance): Pool-3 feature distribution distance.
  - **sFID ↓** (spatial FID): intermediate spatial (`mixed_6/conv`) feature distribution distance.
  - **IS ↑** (Inception Score): class-conditional sharpness and diversity score.
  - **Precision ↑**: generative precision in feature manifold space (**Recall** is also saved as an extra ADM output).

### What Is Being Compared
- `fp32`: official pretrained DiT-XL/2 checkpoint in FP32 (TF32 disabled).
- `fp16`: same pretrained checkpoint with FP16 mixed-precision execution.
- `int8` / `int4`: pretrained weights quantized with this repository's symmetric per-output-channel weight quantizer (W8/W4) and dynamic per-token activation quantizer (A8), replacing `attn.qkv`, `attn.proj`, `mlp.fc1`, and `mlp.fc2` across all 28 DiT blocks by default (`SCOPE="all"`).
- Optional `fake8` / `fake4`: identical quantization grids evaluated with floating-point `F.linear` for diagnostic separation of quantization error vs. kernel implementation error.

> **Important**: These custom modes are an ablation on the official pretrained DiT weights, **not** a reproduction of calibrated PTQ4DiT (which uses CSB, SSC, offline reconstruction, and different quantized op scope). Published PTQ4DiT Table 1/2 numbers are displayed separately as contextual reference rows and never presented as values measured here.
""",
        ),
        markdown_cell(
            "cell-01",
            """## 1. Restore bundled source and select settings
Extracts the SHA-256-verified repository bundle and configures the experiment.
- `PROFILE="paper"` uses the paper sample counts (**10,000** at 256×256; **5,000** at 512×512).
- `PROFILE="preview"` uses **250** images/mode (`--allow-small-sample`) and labels results exploratory.
- Choose a persistent Google Drive directory (`MOUNT_GOOGLE_DRIVE = True`) if you want resumable image shards and downloaded checkpoints to survive Colab runtime deletion.
""",
        ),
        code_cell(
            "cell-02",
            """import base64, hashlib, io, json, os, shutil, subprocess, sys, zipfile
from pathlib import Path

REPO = Path('/content/Team12_Triton_Kernels_Final')
REPO.mkdir(parents=True, exist_ok=True)
PAYLOAD = '__PAYLOAD__'
EXPECTED_SHA256 = '__SHA256__'
archive = base64.b64decode(PAYLOAD)
assert hashlib.sha256(archive).hexdigest() == EXPECTED_SHA256, 'Bundled source checksum mismatch'
with zipfile.ZipFile(io.BytesIO(archive)) as bundle:
    for entry in bundle.infolist():
        target = (REPO / entry.filename).resolve()
        if not target.is_relative_to(REPO.resolve()):
            raise ValueError(f'Unsafe archive entry: {entry.filename}')
    bundle.extractall(REPO)
os.chdir(REPO)
if str(REPO / 'src') not in sys.path:
    sys.path.insert(0, str(REPO / 'src'))
os.environ['PYTHONPATH'] = str(REPO / 'src') + os.pathsep + os.environ.get('PYTHONPATH', '')

PROFILE = "paper" # @param ["paper", "preview"]
IMAGE_SIZE = 256 # @param [256, 512] {type:"raw"}
STEPS = 50 # @param [50, 100, 250] {type:"raw"}
BATCH_SIZE = 4 # @param {type:"integer"}
CFG = 1.5 # @param {type:"number"}
SEED = 1 # @param {type:"integer"}
SCOPE = "all" # @param ["all", "mlp"]
QUANT_DTYPE = "float16" # @param ["float16", "bfloat16", "float32"]
INCLUDE_FAKE_QDQ = False # @param {type:"boolean"}
MOUNT_GOOGLE_DRIVE = False # @param {type:"boolean"}
STORAGE_DIR = "/content/dit_pretrained_eval" # @param {type:"string"}
RUN_NAME = "default" # @param {type:"string"}

if MOUNT_GOOGLE_DRIVE:
    from google.colab import drive
    drive.mount('/content/drive')
    if STORAGE_DIR == "/content/dit_pretrained_eval":
        STORAGE_DIR = "/content/drive/MyDrive/dit_pretrained_eval"

PAPER_SAMPLES = 10000 if IMAGE_SIZE == 256 else 5000
NUM_SAMPLES = PAPER_SAMPLES if PROFILE == "paper" else 250
MODES = ['fp32', 'fp16', 'int8', 'int4'] + (['fake8', 'fake4'] if INCLUDE_FAKE_QDQ else [])

STORAGE_ROOT = Path(STORAGE_DIR).resolve()
ASSETS_DIR = STORAGE_ROOT / 'assets'
resolved_run_name = (
    f'dit_xl2_{IMAGE_SIZE}px_{STEPS}steps_n{NUM_SAMPLES}_cfg{CFG}_seed{SEED}_{SCOPE}'
    if RUN_NAME.strip() in ('', 'default') else RUN_NAME.strip()
)
OUTPUT_DIR = STORAGE_ROOT / 'runs' / resolved_run_name
LOGS_DIR = OUTPUT_DIR / 'logs'
LOGS_DIR.mkdir(parents=True, exist_ok=True)

def run_cmd(command, log_name=None):
    \"\"\"Stream subprocess output live, optionally tee to a log file, and fail closed.\"\"\"
    cmd = [str(c) for c in command]
    print('$', ' '.join(cmd), flush=True)
    log_path = (LOGS_DIR / log_name) if log_name else None
    stream = log_path.open('a') if log_path else None
    try:
        with subprocess.Popen(cmd, cwd=REPO, env=os.environ.copy(),
                              stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                              text=True, bufsize=1) as proc:
            for line in proc.stdout:
                print(line, end='')
                if stream:
                    stream.write(line)
            code = proc.wait()
        if code:
            raise subprocess.CalledProcessError(code, cmd)
    finally:
        if stream:
            stream.close()

print(json.dumps(dict(
    repo=str(REPO), storage_root=str(STORAGE_ROOT), assets_dir=str(ASSETS_DIR),
    output_dir=str(OUTPUT_DIR), profile=PROFILE, image_size=IMAGE_SIZE,
    num_samples=NUM_SAMPLES, paper_samples=PAPER_SAMPLES, batch_size=BATCH_SIZE,
    steps=STEPS, cfg=CFG, seed=SEED, scope=SCOPE, quant_dtype=QUANT_DTYPE, modes=MODES
), indent=2))
""".replace("__PAYLOAD__", payload).replace("__SHA256__", sha),
        ),
        markdown_cell(
            "cell-03",
            """## 2. Install model dependencies & create isolated Python 3.11 ADM evaluator environment
Preserves Colab's preinstalled CUDA `torch` / `torchvision` / `triton` pair, installs model dependencies (`requirements-colab.txt`), runs the unit test suite, and creates an isolated Python 3.11 virtual environment (`.venv-adm`) with TensorFlow (`requirements-adm.txt`) for the official ADM evaluator.
""",
        ),
        code_cell(
            "cell-04",
            """import torch, triton

assert torch.cuda.is_available(), 'Select a GPU runtime in Runtime -> Change runtime type.'
capability = torch.cuda.get_device_capability(0)
assert capability >= (7, 5), f'Compute capability >= 7.5 required; got {capability}'

run_cmd([sys.executable, '-m', 'pip', 'install', '-q', '-r', 'requirements-colab.txt'], 'pip_colab.log')
run_cmd([sys.executable, '-m', 'unittest', 'discover', '-s', 'tests', '-v'], 'unit_tests.log')

ADM_VENV = STORAGE_ROOT / '.venv-adm'
ADM_PYTHON = ADM_VENV / 'bin' / 'python'
if not ADM_PYTHON.exists():
    run_cmd(['uv', 'venv', '--python', '3.11', ADM_VENV], 'adm_venv.log')
run_cmd(['uv', 'pip', 'install', '--python', ADM_PYTHON, '-r', 'requirements-adm.txt'], 'adm_pip.log')

print(json.dumps(dict(
    gpu=torch.cuda.get_device_name(0),
    capability=capability,
    torch=torch.__version__,
    triton=triton.__version__,
    cuda=torch.version.cuda,
    adm_python=str(ADM_PYTHON),
), indent=2))
""",
        ),
        markdown_cell(
            "cell-05",
            """## 3. Download official DiT-XL/2 weights, pinned upstream checkouts, and ADM ImageNet reference
Downloads the official `DiT-XL-2-{256x256,512x512}.pt` checkpoint, clones `facebookresearch/DiT` and `openai/guided-diffusion` at pinned commits, and downloads the official ADM ImageNet reference batch (`VIRTUAL_imagenet256_labeled.npz` or `VIRTUAL_imagenet512.npz`) required for FID, sFID, IS, and Precision.
""",
        ),
        code_cell(
            "cell-06",
            """run_cmd([
    sys.executable, 'scripts/setup_pretrained.py',
    '--assets', ASSETS_DIR,
    '--image-size', IMAGE_SIZE,
], f'setup_pretrained_{IMAGE_SIZE}.log')

ASSETS = json.loads((ASSETS_DIR / f'assets_{IMAGE_SIZE}.json').read_text())
print(json.dumps(ASSETS, indent=2))
""",
        ),
        markdown_cell(
            "cell-07",
            """## 4. Generate matched images for each mode (resumable)
Runs `fc1bench.generate` in a fresh process per mode (`fp32`, `fp16`, `int8`, `int4`).
- Initial diffusion noise, class labels, and per-step ancestral sampling draws are seed-matched across modes.
- Before sampling, quantized modes verify every converted projection layer against floating-point QDQ on a real forward pass and save `implementation_checks.json`.
- Completed batch shards are verified by SHA-256 and skipped automatically when resuming an interrupted run with identical settings.
""",
        ),
        code_cell(
            "cell-08",
            """for mode in MODES:
    run_cmd([
        sys.executable, '-m', 'fc1bench.generate',
        '--dit-repo', ASSETS['dit_repo'],
        '--checkpoint', ASSETS['checkpoint'],
        '--output', OUTPUT_DIR,
        '--mode', mode,
        '--image-size', IMAGE_SIZE,
        '--num-samples', NUM_SAMPLES,
        '--batch-size', BATCH_SIZE,
        '--steps', STEPS,
        '--cfg', CFG,
        '--seed', SEED,
        '--scope', SCOPE,
        '--quant-dtype', QUANT_DTYPE,
        '--vae', ASSETS['vae'],
        '--vae-revision', ASSETS['vae_revision'],
    ], f'generate_{mode}.log')
""",
        ),
        markdown_cell(
            "cell-09",
            """## 5. Run official ADM evaluation and view accuracy metrics comparison
Executes `fc1bench.image_quality` inside the isolated Python 3.11 TensorFlow environment after PyTorch generation has exited.
- Computes **FID ↓**, **sFID ↓**, **IS ↑**, and **Precision ↑** (plus **Recall**), alongside deltas vs. measured `fp32`, throughput (`images_per_second`), `model_storage_bytes`, and `peak_allocated_bytes`.
- Displays measured results separately from published PTQ4DiT Table 1/2 reference rows.
""",
        ),
        code_cell(
            "cell-10",
            """from IPython.display import Image as IPyImage, Markdown, display

quality_cmd = [
    ADM_PYTHON, '-m', 'fc1bench.image_quality',
    '--output', OUTPUT_DIR,
    '--adm-repo', ASSETS['adm_repo'],
    '--reference', ASSETS['reference'],
    '--modes', *MODES,
]
if NUM_SAMPLES < PAPER_SAMPLES:
    quality_cmd.append('--allow-small-sample')

run_cmd(quality_cmd, 'image_quality.log')

comparison_md = (OUTPUT_DIR / 'quality' / 'comparison.md').read_text()
display(Markdown(comparison_md))
print((OUTPUT_DIR / 'quality' / 'comparison.csv').read_text())

for mode in MODES:
    preview = OUTPUT_DIR / mode / 'preview_0.png'
    if preview.exists():
        print(f'Preview sample 0 — {mode}:')
        display(IPyImage(filename=str(preview)))
""",
        ),
        markdown_cell(
            "cell-11",
            """## 6. Download compact reports and previews
Packages `quality/comparison.{json,csv,md}`, per-mode `manifest.json`, `implementation_checks.json`, and `preview_*.png` into a compact ZIP archive for download. Large `samples.npz` archives, batch shards, and model weights remain in `STORAGE_ROOT`.
""",
        ),
        code_cell(
            "cell-12",
            """report_bundle = OUTPUT_DIR / f'{OUTPUT_DIR.name}_reports.zip'
with zipfile.ZipFile(report_bundle, 'w', compression=zipfile.ZIP_DEFLATED) as zf:
    for rel_path in sorted(OUTPUT_DIR.rglob('*')):
        if not rel_path.is_file():
            continue
        rel = rel_path.relative_to(OUTPUT_DIR).as_posix()
        if (
            rel.startswith('quality/comparison.')
            or rel.startswith('logs/')
            or rel.endswith('/manifest.json')
            or rel.endswith('/implementation_checks.json')
            or '/preview_' in f'/{rel}'
        ):
            zf.write(rel_path, rel)

print(f'Saved compact report bundle: {report_bundle} ({report_bundle.stat().st_size} bytes)')
try:
    from google.colab import files
    files.download(str(report_bundle))
except ImportError:
    pass
""",
        ),
    ]
    notebook = {
        "nbformat": 4,
        "nbformat_minor": 5,
        "metadata": {
            "colab": {
                "name": "Team12_Triton_Kernels_Colab.ipynb",
                "provenance": [],
            },
            "accelerator": "GPU",
            "kernelspec": {
                "name": "python3",
                "display_name": "Python 3",
            },
            "language_info": {
                "name": "python",
            },
        },
        "cells": cells,
    }
    return notebook


def main():
    # Write once, then rebuild so the bundled copy of build_colab_notebook.py is up to date.
    nb = build_notebook()
    NOTEBOOK_PATH.write_text(json.dumps(nb, indent=1) + "\n")
    print(f"Wrote {NOTEBOOK_PATH} ({NOTEBOOK_PATH.stat().st_size} bytes)")


if __name__ == "__main__":
    main()
