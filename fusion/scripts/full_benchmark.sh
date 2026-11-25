#!/usr/bin/env bash
# Full benchmark of every variant on one Colab GPU VM (resumable, one stage at a time).
#   bash scripts/full_benchmark.sh <gpu-tag: a100|l4> <images per variant for FID>
#
# Variants: FP32 (quality reference), FP16 compiled, INT8 unfused, v2 (fused, published), v3, v4, v4w, v4c.
# Stages (results under /content/full; small files of each finished stage packed into pkg/<stage>.tgz):
#   s0_env       GPU / software record
#   s1_tests     kernel unit tests
#   s2_latency   per-step latency at batches 4/8/16/32 (+ full 50-step sampling time, peak memory, kernel
#                breakdown, per-step error vs FP16 and FP32)
#   s3_accuracy  paired 16-image full-sampling accuracy vs FP32 (latent rel-L2, decoded PSNR)
#   s4_nsys      Nsight Systems graphed-step profiles at batches 4 and 32
#   gen_<label>  image generation per variant (batch 32, 50 DDPM steps, CFG 1.5, seed 1)
#   s6_quality   ADM evaluator: FID, sFID, IS, Precision, Recall vs the ImageNet reference batch
set -uo pipefail
cd "$(dirname "$0")/.."
GPU=${1:?gpu tag (a100|l4)}
NS=${2:?images per variant}
ASSETS=/content/dit_pretrained_eval/assets
CK=$ASSETS/pretrained_models/DiT-XL-2-256x256.pt
DIT=$ASSETS/DiT
VAE=f04b2c4b98319346dad8c65879f680b1997b204a
OUT=/content/full
PKG=$OUT/pkg
mkdir -p "$PKG"
export PYTHONPATH=$PWD/src PYTHONUNBUFFERED=1 CUTLASS_DIR=${CUTLASS_DIR:-/content/cutlass}
export PATH=/usr/local/cuda/bin:$PATH CUDA_HOME=/usr/local/cuda
LOG=$OUT/run.log
SCAFFOLDS="sdpa_graph_pure16_compile_tuned sdpa_graph_pure16_compile_tuned_fused sdpa_graph_pure16_compile_v3 \
sdpa_graph_pure16_compile_v4 sdpa_graph_pure16_compile_v4w sdpa_graph_pure16_compile_v4c"

say() { echo "[$(date +%T)] $*" >> "$LOG"; }
stage_ok() { grep -q "^$1 " "$OUT/STAGES" 2>/dev/null; }
# done_stage <name> "<paths or globs relative to $OUT>"
done_stage() {
  (cd "$OUT" && tar -czf "$PKG/$1.tgz" $2 2>/dev/null)
  echo "$1 $(date +%s)" >> "$OUT/STAGES"
  say "stage $1 done"
}

say "start gpu=$GPU images=$NS"

if ! stage_ok s0_env; then
  mkdir -p "$OUT/env"
  nvidia-smi > "$OUT/env/nvidia-smi.txt" 2>&1
  { python3 -c "import torch, triton; print('torch', torch.__version__, 'triton', triton.__version__, 'cuda', torch.version.cuda)"
    echo "cutlass $(git -C "$CUTLASS_DIR" rev-parse HEAD)"; nvcc --version | tail -1; } > "$OUT/env/versions.txt" 2>&1
  done_stage s0_env "env"
fi

if ! stage_ok s1_tests; then
  say "tests"
  mkdir -p "$OUT/tests"
  for t in test_fused_ops test_numerics test_optimized test_fused_gemm test_cutlass_gemm; do
    timeout 3600 python3 "tests/$t.py" > "$OUT/tests/$t.log" 2>&1
    echo "$t exit $?" >> "$OUT/tests/summary.txt"
  done
  done_stage s1_tests "tests"
fi

if ! stage_ok s2_latency; then
  say "latency"
  python3 -u -m fc1bench.latency_bench --dit-repo "$DIT" --checkpoint "$CK" --output "$OUT/latency" \
    --modes fp16 int8 --batches 4 8 16 32 --warmup 10 --repeats 50 --profile --sampling --sampling-repeats 2 \
    --scaffolds $SCAFFOLDS > "$OUT/latency_raw.log" 2>&1
  grep -av "Warning\|warnings.warn" "$OUT/latency_raw.log" > "$OUT/latency/latency.log"
  done_stage s2_latency "latency/summary.csv latency/results.json latency/manifest.json latency/tuning_bank.json \
latency/tuning_summary.json latency/latency.log"
fi
BANK=$OUT/latency/tuning_bank.json

if ! stage_ok s3_accuracy; then
  say "accuracy"
  mkdir -p "$OUT/accuracy"
  for v in int8 v2 v3 v4 v4w v4c; do
    case $v in int8) flag="" ;; v2) flag="--fused" ;; v3) flag="--v3" ;; *) flag="--v4 $v" ;; esac
    python3 -u -m fc1bench.csb evaluate --dit-repo "$DIT" --checkpoint "$CK" --scales "results/$GPU/csb_scales.pt" \
      --output "$OUT/accuracy/accuracy_$v.json" --modes int8 --images 16 --decode --no-csb $flag \
      > "$OUT/accuracy/accuracy_$v.log" 2>&1
    say "accuracy $v exit $?"
  done
  done_stage s3_accuracy "accuracy"
fi

if ! stage_ok s4_nsys; then
  say "nsys"
  mkdir -p "$OUT/nsys"
  PAIRS="fp16:sdpa_graph_pure16_compile_tuned"
  for s in $SCAFFOLDS; do PAIRS="$PAIRS int8:$s"; done
  for b in 4 32; do
    for pr in $PAIRS; do
      mode=${pr%%:*}; sc=${pr#*:}; tag="step_${mode}_${sc}_b$b"
      nsys profile --capture-range=cudaProfilerApi --capture-range-end=stop --cuda-graph-trace=node -t cuda,nvtx \
        --force-overwrite true -o "$OUT/nsys/$tag" python3 -m fc1bench.nsys_step --dit-repo "$DIT" --checkpoint "$CK" \
        --bank "$BANK" --mode "$mode" --scaffold "$sc" --batch "$b" --replays 20 > "$OUT/nsys/$tag.log" 2>&1 \
        || { echo "FAILED $tag" >> "$OUT/nsys/failures.txt"; continue; }
      nsys stats --report cuda_gpu_trace --format csv --output "$OUT/nsys/$tag" --force-export true \
        "$OUT/nsys/$tag.nsys-rep" > /dev/null 2>&1
      say "nsys $tag"
    done
  done
  python3 scripts/nsys_summary.py "$OUT/nsys" > "$OUT/nsys/summary.txt" 2>&1
  done_stage s4_nsys "nsys/nsys_summary.csv nsys/summary.txt nsys/*.log nsys/failures.txt"
fi

GEN=$OUT/gen
run_gen() {
  local label=$1; shift
  if ! stage_ok "gen_$label"; then
    say "generate $label ($NS images)"
    python3 -u -m fc1bench.generate --dit-repo "$DIT" --checkpoint "$CK" --output "$GEN" --label "$label" \
      --image-size 256 --batch-size 32 --steps 50 --cfg 1.5 --seed 1 --vae-revision "$VAE" --num-samples "$NS" \
      "$@" > "$OUT/gen_$label.log" 2>&1
    say "generate $label exit $?"
    grep -av "Warning\|warnings.warn" "$OUT/gen_$label.log" | tail -40 > "$GEN/$label/generate_tail.log" 2>/dev/null
    done_stage "gen_$label" "gen/$label/manifest.json gen/$label/preview_0.png gen/$label/preview_1.png \
gen/$label/preview_2.png gen/$label/preview_3.png gen/$label/generate_tail.log"
  fi
}
run_gen fp32 --mode fp32 --scaffold sdpa_graph_compile --tf32
run_gen fp16 --mode fp16 --scaffold sdpa_graph_pure16_compile
run_gen int8 --mode int8 --scaffold sdpa_graph_pure16_compile_tuned --bank "$BANK"
run_gen int8_v2 --mode int8 --scaffold sdpa_graph_pure16_compile_tuned_fused --bank "$BANK"
run_gen int8_v3 --mode int8 --scaffold sdpa_graph_pure16_compile_v3
run_gen int8_v4 --mode int8 --scaffold sdpa_graph_pure16_compile_v4
run_gen int8_v4w --mode int8 --scaffold sdpa_graph_pure16_compile_v4w
run_gen int8_v4c --mode int8 --scaffold sdpa_graph_pure16_compile_v4c

if ! stage_ok s6_quality; then
  say "quality"
  python3 scripts/setup_pretrained.py --assets "$ASSETS" --image-size 256 > "$OUT/quality_setup.log" 2>&1
  ADM_ENV=/content/adm-env
  ADMPY=$ADM_ENV/bin/python
  [ -x "$ADMPY" ] || uv venv -q --python 3.11 "$ADM_ENV" >> "$OUT/quality_setup.log" 2>&1
  # Same pins as baseline/requirements-adm.txt; the and-cuda extra lets TensorFlow use the GPU (CPU otherwise).
  uv pip install -q --python "$ADMPY" "tensorflow[and-cuda]==2.15.1" numpy==1.23.5 scipy==1.11.4 requests==2.32.4 \
      tqdm==4.67.1 >> "$OUT/quality_setup.log" 2>&1 \
    || uv pip install -q --python "$ADMPY" tensorflow==2.15.1 numpy==1.23.5 scipy==1.11.4 requests==2.32.4 \
      tqdm==4.67.1 >> "$OUT/quality_setup.log" 2>&1
  SMALL=""
  [ "$NS" -lt 10000 ] && SMALL="--allow-small-sample"
  "$ADMPY" -m fc1bench.image_quality --output "$GEN" --adm-repo "$ASSETS/guided-diffusion" \
    --reference "$ASSETS/references/VIRTUAL_imagenet256_labeled.npz" \
    --modes fp32 fp16 int8 int8_v2 int8_v3 int8_v4 int8_v4w int8_v4c $SMALL > "$OUT/quality.log" 2>&1
  say "quality exit $?"
  done_stage s6_quality "gen/quality/comparison.json gen/quality/comparison.csv gen/quality/comparison.md \
quality.log quality_setup.log"
fi

echo "ALL_DONE $(date +%s)" >> "$OUT/STAGES"
say "all done"
