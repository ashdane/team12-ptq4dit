#!/usr/bin/env bash
# Optimization round 2 on a Colab GPU VM: tests -> sweep5 (fresh tuning) -> accuracy -> nsys.
#   bash scripts/final_opt.sh <csb_scales.pt>
set -uo pipefail
cd "$(dirname "$0")/.."
ASSETS=/content/dit_pretrained_eval/assets; CK=$ASSETS/pretrained_models/DiT-XL-2-256x256.pt; DIT=$ASSETS/DiT
SCALES=${1:?csb_scales.pt}
OUT=/content/opt2; mkdir -p $OUT/nsys
export PYTHONPATH=$PWD/src PYTHONUNBUFFERED=1
echo "== 1. tests"
for t in tests/test_fused_ops.py tests/test_numerics.py tests/test_optimized.py; do
  timeout 1800 python3 $t 2>&1 | grep -aE "^test_|mismatch|bit-identical|exact fraction|Error|FAIL|^OK|Ran " | sed "s#^#[$t] #"
done
echo "== 2. sweep5"
python3 -u -m fc1bench.latency_bench --dit-repo $DIT --checkpoint $CK --output $OUT/sweep5 --modes fp16 int8 \
  --batches 4 8 16 32 --warmup 10 --repeats 50 --profile \
  --scaffolds sdpa_graph_pure16_compile_tuned sdpa_graph_pure16_compile_tuned_fused 2>&1 | grep -v "Warning\|warnings.warn" > $OUT/sweep5.log
grep -aE "^Saved|Traceback|FAILED" $OUT/sweep5.log | tail -3
echo "== 3. accuracy (16 images, full sampling, vs fp32)"
python3 -u -m fc1bench.csb evaluate --dit-repo $DIT --checkpoint $CK --scales $SCALES --output $OUT/accuracy_fused.json \
  --modes int8 --images 16 --decode --fused --no-csb 2>&1 | grep -aE "^\{|Traceback|Error" | tail -5
echo "== 4. nsys final config"
for b in 4 32; do for cfg in "fp16 sdpa_graph_pure16_compile_tuned" "int8 sdpa_graph_pure16_compile_tuned_fused"; do
  set -- $cfg; tag="step_${1}_${2}_b$b"
  nsys profile --capture-range=cudaProfilerApi --capture-range-end=stop --cuda-graph-trace=node -t cuda,nvtx \
    --force-overwrite true -o "$OUT/nsys/$tag" python3 -m fc1bench.nsys_step --dit-repo $DIT --checkpoint $CK \
    --bank $OUT/sweep5/tuning_bank.json --mode $1 --scaffold $2 --batch $b --replays 20 > "$OUT/nsys/$tag.log" 2>&1 || { echo "FAILED $tag"; continue; }
  nsys stats --report cuda_gpu_trace --format csv --output "$OUT/nsys/$tag" --force-export true "$OUT/nsys/$tag.nsys-rep" > /dev/null 2>&1
  echo "done $tag"
done; done
echo "ALL DONE"
