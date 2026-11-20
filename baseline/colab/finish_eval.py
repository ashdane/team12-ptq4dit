"""Finish a Colab run whose ADM evaluation failed with
'Cannot compare runs with different dit_commit'.

Cause: the DiT checkout's `git status` recorded untracked `__pycache__/` folders
for every mode after the first. This strips untracked ('??') entries from each
manifest's dit_commit.status (backup kept as manifest.json.bak), then runs the
ADM evaluation and builds the compact report zip. Generated images are untouched.

Run on the Colab VM:  python3 /content/finish_eval.py [RUN_NAME]
"""
import json, os, subprocess, sys, zipfile
from pathlib import Path

ROOT = Path('/content/dit_pretrained_eval')
REPO = Path('/content/Team12_Triton_Kernels_Final')
RUN = sys.argv[1] if len(sys.argv) > 1 else 'dit_xl2_256px_50steps_n250_cfg1.5_seed1_all'
OUT = ROOT / 'runs' / RUN
MODES = [m for m in ('fp32', 'fp16', 'int8', 'int4', 'fake8', 'fake4') if (OUT / m / 'manifest.json').exists()]
assert 'fp32' in MODES, f'No fp32 manifest under {OUT}'

for mode in MODES:
    path = OUT / mode / 'manifest.json'
    m = json.loads(path.read_text())
    if m.get('status') != 'complete':
        sys.exit(f'{mode} has not finished generating (status={m.get("status")}); wait and rerun.')
    status = m['dit_commit']['status']
    cleaned = ''.join(l for l in status.splitlines(keepends=True) if not l.startswith('?? '))
    if cleaned != status:
        backup = path.with_suffix('.json.bak')
        if not backup.exists():
            backup.write_text(path.read_text())
        m['dit_commit']['status'] = cleaned
        path.write_text(json.dumps(m, indent=2) + '\n')
        print(f'{mode}: removed untracked entries {status!r}', flush=True)

num = json.loads((OUT / 'fp32' / 'manifest.json').read_text())['num_samples']
size = json.loads((OUT / 'fp32' / 'manifest.json').read_text())['image_size']
assets_file = ROOT / 'assets' / f'assets_{size}.json'
assets = json.loads(assets_file.read_text()) if assets_file.exists() else {}
reference = assets.get('reference') or str(ROOT / 'assets' / 'references' /
                                           ('VIRTUAL_imagenet256_labeled.npz' if size == 256 else 'VIRTUAL_imagenet512.npz'))
cmd = [str(ROOT / '.venv-adm' / 'bin' / 'python'), '-m', 'fc1bench.image_quality',
       '--output', str(OUT), '--adm-repo', assets.get('adm_repo', str(ROOT / 'assets' / 'guided-diffusion')),
       '--reference', reference, '--modes', *MODES]
if num < (10000 if size == 256 else 5000):
    cmd.append('--allow-small-sample')
print('$', ' '.join(cmd), flush=True)
env = dict(os.environ, PYTHONPATH=str(REPO / 'src'), PYTHONUNBUFFERED='1')
(OUT / 'logs').mkdir(exist_ok=True)
with (OUT / 'logs' / 'image_quality.log').open('a') as log, subprocess.Popen(
        cmd, cwd=REPO, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1) as proc:
    for line in proc.stdout:
        print(line, end='', flush=True)
        log.write(line)
if proc.returncode:
    sys.exit(f'ADM evaluation failed (exit {proc.returncode})')

print((OUT / 'quality' / 'comparison.md').read_text())
bundle = OUT / f'{RUN}_reports.zip'
with zipfile.ZipFile(bundle, 'w', compression=zipfile.ZIP_DEFLATED) as zf:
    for p in sorted(OUT.rglob('*')):
        rel = p.relative_to(OUT).as_posix()
        if p.is_file() and (rel.startswith('quality/comparison.') or rel.startswith('logs/')
                            or rel.endswith('/manifest.json') or rel.endswith('/manifest.json.bak')
                            or rel.endswith('/implementation_checks.json') or '/preview_' in f'/{rel}'):
            zf.write(p, rel)
print(f'Saved compact report bundle: {bundle} ({bundle.stat().st_size} bytes)')
