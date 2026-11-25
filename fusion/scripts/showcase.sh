#!/usr/bin/env bash
# Side-by-side sample images: every variant generates the same 10 images (same 10 ImageNet classes, same
# initial noise and sampling noise), saved as PNGs plus one grid (rows = variants, columns = classes).
#   bash scripts/showcase.sh <gpu-tag> [tuning_bank.json]
# Output: /content/showcase (<variant>/img_*.png, showcase_grid_<gpu>.png), packed into /content/showcase/pkg.tgz.
set -uo pipefail
cd "$(dirname "$0")/.."
GPU=${1:?gpu tag}
BANK=${2:-/content/full/latency/tuning_bank.json}
ASSETS=/content/dit_pretrained_eval/assets
CK=$ASSETS/pretrained_models/DiT-XL-2-256x256.pt
DIT=$ASSETS/DiT
VAE=f04b2c4b98319346dad8c65879f680b1997b204a
OUT=/content/showcase
mkdir -p "$OUT"
export PYTHONPATH=$PWD/src PYTHONUNBUFFERED=1 CUTLASS_DIR=${CUTLASS_DIR:-/content/cutlass}
export PATH=/usr/local/cuda/bin:$PATH CUDA_HOME=/usr/local/cuda
# goldfish, macaw, golden retriever, lion, giant panda, balloon, lighthouse (beacon), convertible,
# cheeseburger, volcano
LABELS=1,88,207,291,388,417,437,511,933,980
LOG=$OUT/showcase.log

show() {
  local label=$1; shift
  python3 -u -m fc1bench.generate --dit-repo "$DIT" --checkpoint "$CK" --output "$OUT" --label "$label" \
    --image-size 256 --batch-size 32 --steps 50 --cfg 1.5 --seed 7 --vae-revision "$VAE" --num-samples 10 \
    --labels "$LABELS" --save-png "$@" > "$OUT/gen_$label.log" 2>&1
  echo "[$(date +%T)] $label exit $?" >> "$LOG"
}
show fp32 --mode fp32 --scaffold sdpa_graph_compile --tf32
show fp16 --mode fp16 --scaffold sdpa_graph_pure16_compile
show int8 --mode int8 --scaffold sdpa_graph_pure16_compile_tuned --bank "$BANK"
show int8_v2 --mode int8 --scaffold sdpa_graph_pure16_compile_tuned_fused --bank "$BANK"
show int8_v3 --mode int8 --scaffold sdpa_graph_pure16_compile_v3
show int8_v4 --mode int8 --scaffold sdpa_graph_pure16_compile_v4
show int8_v4w --mode int8 --scaffold sdpa_graph_pure16_compile_v4w
show int8_v4c --mode int8 --scaffold sdpa_graph_pure16_compile_v4c

python3 - "$OUT" "$GPU" <<'PY' >> "$LOG" 2>&1
import sys
from pathlib import Path
from PIL import Image, ImageDraw
out, gpu = Path(sys.argv[1]), sys.argv[2]
classes = [(1, "goldfish"), (88, "macaw"), (207, "golden retriever"), (291, "lion"), (388, "giant panda"),
           (417, "balloon"), (437, "lighthouse"), (511, "convertible"), (933, "cheeseburger"), (980, "volcano")]
rows = [("fp32", "FP32"), ("fp16", "FP16"), ("int8", "INT8 unfused"), ("int8_v2", "v2"), ("int8_v3", "v3"),
        ("int8_v4", "v4"), ("int8_v4w", "v4w"), ("int8_v4c", "v4c")]
cell, left, top = 256, 150, 40
grid = Image.new("RGB", (left + cell * len(classes), top + cell * len(rows)), "white")
d = ImageDraw.Draw(grid)
for j, (c, name) in enumerate(classes):
    d.text((left + j * cell + 8, 12), f"{name} ({c})", fill="black")
for i, (label, title) in enumerate(rows):
    d.text((10, top + i * cell + cell // 2), title, fill="black")
    for j, (c, _) in enumerate(classes):
        f = out / label / f"img_{j:05d}_c{c:03d}.png"
        if f.exists():
            grid.paste(Image.open(f), (left + j * cell, top + i * cell))
        else:
            d.text((left + j * cell + 90, top + i * cell + cell // 2), "missing", fill="red")
grid.save(out / f"showcase_grid_{gpu}.png")
print(f"saved showcase_grid_{gpu}.png")
PY
(cd "$OUT" && tar -czf pkg.tgz showcase_grid_*.png showcase.log */img_*.png */manifest.json 2>/dev/null)
echo "DONE $(date +%s)" > "$OUT/DONE"
