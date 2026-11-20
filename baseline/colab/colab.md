# Running Team12_Triton_Kernels_Colab.ipynb on Colab via the `colab` CLI (live output)

> [!IMPORTANT]
> L4 / A100 / H100 GPUs need **Colab Pro**, **Pro+**, or Pay-As-You-Go compute units.
> If `colab new` returns **400**, your account has no quota for that GPU — try another (`--gpu T4`).

Set these once per terminal (edit `GPU` to `A100`, `H100` or `T4` if you prefer):

```bash
export GPU=L4
export SESSION=team12-l4
export PROJECT="$(git rev-parse --show-toplevel)/baseline"   # run from inside your clone of team12-ptq4dit
# (WSL: same command from inside the clone)
export RUN=dit_xl2_256px_50steps_n250_cfg1.5_seed1_all   # default run folder for the preview profile
```

---

## Step 1 — Install the CLI

```bash
pip3 install google-colab-cli      # or: uv tool install google-colab-cli
colab --help
```

## Step 2 — Authenticate the CLI

```bash
colab sessions     # first use opens a browser OAuth consent; afterwards it just lists sessions
```

> [!NOTE]
> `colab auth` is **not** CLI login — it injects GCP credentials *into the VM*. For a CLI 401/403,
> check `colab whoami` and re-run the login (or use `colab --auth=adc ...` with `gcloud auth application-default login`).

## Step 3 — Start a GPU session

```bash
colab new -s "$SESSION" --gpu "$GPU"
colab status -s "$SESSION"
```

> [!WARNING]
> Supported `--gpu` values: `T4`, `L4`, `G4`, `H100`, `A100`. An unrecognised value (e.g. `V100`)
> silently falls back to **A100**.

## Step 4 — Upload the notebook

```bash
cd "$PROJECT"
colab upload -s "$SESSION" colab/Team12_Triton_Kernels_Colab.ipynb /content/Team12_Triton_Kernels_Colab.ipynb
```

The notebook is self-contained (the source is embedded and SHA-256 checked), so nothing else needs uploading.

## Step 5 — Run with live output

Open a shell on the VM:

```bash
colab ssh -s "$SESSION"
```

Then **inside the VM shell**:

```bash
cd /content
jupyter nbconvert --to python Team12_Triton_Kernels_Colab.ipynb     # -> Team12_Triton_Kernels_Colab.py
ls -l Team12_Triton_Kernels_Colab.py                                 # must exist before the next line
nohup env PYTHONUNBUFFERED=1 ipython Team12_Triton_Kernels_Colab.py > run.log 2>&1 &
tail -f run.log
```

You now see everything live: pip, unit tests, ~2 GB asset download, per-batch generation
(`int8: 48/250 images; sampling …`), and the final FID / sFID / IS / Precision table.

- `Ctrl+C` stops only `tail`; the job keeps running.
- Reattach any time (also after a disconnect): `colab ssh -s "$SESSION"`, then `tail -f /content/run.log`.
- Check it is alive: `pgrep -af ipython`.
- When finished, `exit` the VM shell.

> [!TIP]
> Alternative without SSH (also streams to your terminal, but dies if your terminal closes and
> writes `Team12_Triton_Kernels_Colab_output.ipynb` locally):
> ```bash
> colab exec -s "$SESSION" --timeout 14400 -f Team12_Triton_Kernels_Colab.ipynb
> ```
> `--timeout` is required — the default 30 s kills long cells.

## Step 6 — Download results

```bash
cd "$PROJECT"
colab download -s "$SESSION" "/content/dit_pretrained_eval/runs/$RUN/${RUN}_reports.zip" .
colab download -s "$SESSION" /content/run.log .
```

The last notebook cell tries a browser download and may print an error outside Jupyter — harmless,
the zip is already saved and fetched above.

## Step 7 — Stop the session

> [!CAUTION]
> Always stop the session when done to save compute units!

```bash
colab stop -s "$SESSION"
colab sessions      # confirm nothing is left running
```

---

## Latency benchmark (Phase 0) — no notebook needed

Times one DiT `forward_with_cfg` per sampling step for fp16 / int8 (W8A8) / int4 (W4A8) under shared
"scaffolds" (attention impl, CUDA graphs, fp16 I/O, tuned INT tiles), batches 4–32. ~30–60 min on A100.

```bash
# on your machine: package the code and upload it
cd "$PROJECT"
tar czf /tmp/repo.tgz --exclude=__pycache__ README.md requirements-colab.txt requirements-adm.txt scripts src tests
colab new -s team12-lat --gpu A100
colab upload -s team12-lat /tmp/repo.tgz /content/repo.tgz
colab ssh -s team12-lat
```

Inside the VM:

```bash
mkdir -p /content/Team12_Triton_Kernels_Final && tar xzf /content/repo.tgz -C /content/Team12_Triton_Kernels_Final
cd /content/Team12_Triton_Kernels_Final
nohup bash scripts/latency_on_colab.sh /content/latency/sweep1 --batches 4 8 16 32 --profile > /content/latency/sweep1.log 2>&1 &
tail -f /content/latency/sweep1.log
```

Results: `/content/latency/sweep1/summary.csv` (median/p90 ms, speedup vs fp16 same scaffold and vs best fp16,
output error vs fp16, per-module attribution), `results.json`, `tuning_bank.json`, `traces/`.
Download: `colab download -s team12-lat /content/latency/sweep1/summary.csv .`

## Notebook settings (Cell 2, Section 1)

| Setting            | `"preview"` (default) | `"paper"`   |
| ------------------ | --------------------- | ----------- |
| Samples (256×256)  | 250                   | 10,000      |
| Samples (512×512)  | 250                   | 5,000       |
| Steps (default)    | 50                    | 50 (50/100/250 selectable) |
| Batch size         | 4                     | 4           |
| Modes              | fp32, fp16, int8, int4 | same       |

To change settings, edit the `# @param` lines in cell 2 locally and re-upload (Step 4), or edit
`/content/Team12_Triton_Kernels_Colab.py` on the VM before running it. The run folder name
(`$RUN`) follows the settings: `dit_xl2_{size}px_{steps}steps_n{samples}_cfg{cfg}_seed{seed}_{scope}`.

> [!WARNING]
> Preview (250 images) metrics are exploratory — FID/sFID are biased upward at this size; compare
> modes against each other, not against the paper. The paper profile is several hours even on A100.

---

## Troubleshooting

| Issue | Fix |
| --- | --- |
| `colab: command not found` | `pip3 install google-colab-cli`; ensure `~/.local/bin` is on `PATH` |
| `400` on `colab new` | No quota for that GPU — use `--gpu T4` (or another supported one) |
| 401 / 403 from the CLI | `colab whoami`; redo login (not `colab auth`) |
| `Cannot compare runs with different dit_commit` (older notebooks) | Images are fine. From your machine: `colab upload -s "$SESSION" colab/finish_eval.py /content/finish_eval.py`, then on the VM: `python3 -u /content/finish_eval.py` (runs the evaluation + builds the zip) |
| `module 'numpy' has no attribute 'bool'` (older notebooks) | On the VM: `uv pip install --python /content/dit_pretrained_eval/.venv-adm/bin/python numpy==1.23.5`, then `python3 -u /content/finish_eval.py` |
| `ValueError: Imaginary component …` in `frechet_distance` (older notebooks) | Fewer than 2,048 images/mode makes ADM's FID sqrtm fail. Upload the fixed evaluator: `colab upload -s "$SESSION" src/fc1bench/image_quality.py /content/Team12_Triton_Kernels_Final/src/fc1bench/image_quality.py`, then `python3 -u /content/finish_eval.py`. Values from the fallback are marked † |
| `Resume configuration mismatch` | Output folder was made by different code/settings — change `RUN_NAME` in cell 2 or delete the run folder |
| `Bundled source checksum mismatch` | Notebook was edited by hand inside the payload — rebuild with `scripts/build_colab_notebook.py` |
| Out of GPU memory | Lower `BATCH_SIZE` (counts as a new configuration → new run folder) |
| Disconnected | Job keeps running under `nohup`; `colab ssh -s "$SESSION"` then `tail -f /content/run.log` |
| Stray sessions billing | `colab sessions`, then `colab stop -s <name>` |
