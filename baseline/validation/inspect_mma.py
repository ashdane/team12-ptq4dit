import sys
import os
for p in ['src', '../src', 'sim/baseline_experiments', '../sim/baseline_experiments']:
    abs_p = os.path.abspath(p)
    if os.path.exists(abs_p) and abs_p not in sys.path:
        sys.path.insert(0, abs_p)
import torch
import re
from fc1bench.kernels import integer_linear, quantize_activation
from fc1bench.quantization import quantize_rows, pack_int4

x = torch.randn(2048, 1152, device="cuda", dtype=torch.float16)
w = torch.randn(3456, 1152, device="cuda", dtype=torch.float16) / 34.0
b = torch.randn(3456, device="cuda", dtype=torch.float32)

xq, sx = quantize_activation(x)
wq, sw = quantize_rows(w, 8)
_, kernel_w8 = integer_linear(xq, wq, sx, sw, b, packed=False, return_kernel=True)
ptx_w8 = kernel_w8.asm.get("ptx", "")
mma_w8 = [line.strip() for line in ptx_w8.splitlines() if "mma." in line]
print(f"W8A8 MMA instructions in PTX: {len(mma_w8)}")
for m in sorted(set(mma_w8)):
    print(f"  {m}")

wq4, sw4 = quantize_rows(w, 4)
w4_packed = pack_int4(wq4)
_, kernel_w4 = integer_linear(xq, w4_packed, sx, sw4, b, packed=True, return_kernel=True)
ptx_w4 = kernel_w4.asm.get("ptx", "")
mma_w4 = [line.strip() for line in ptx_w4.splitlines() if "mma." in line]
print(f"W4A8 MMA instructions in PTX: {len(mma_w4)}")
for m in sorted(set(mma_w4)):
    print(f"  {m}")

