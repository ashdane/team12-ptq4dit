import sys
import os
for p in ['src', '../src', 'sim/baseline_experiments', '../sim/baseline_experiments']:
    abs_p = os.path.abspath(p)
    if os.path.exists(abs_p) and abs_p not in sys.path:
        sys.path.insert(0, abs_p)
import torch
from fc1bench.optimized import Config, gemm, prepare_weight
from fc1bench.quantization import metrics, pack_int4_kn, unpack_int4_kn

workloads = {
    'attn.qkv': (2048, 1152, 3456),
    'attn.proj': (2048, 1152, 1152),
    'mlp.fc1': (2048, 1152, 4608),
    'mlp.fc2': (2048, 4608, 1152),
}

print('=== VALIDATING FULL DIT SHAPES ON NVIDIA RTX 2080 Ti ===')
for name, (m, k, n) in workloads.items():
    print(f'Testing {name}: M={m}, K={k}, N={n}...')
    torch.manual_seed(42)
    x = torch.randint(-127, 128, (m, k), dtype=torch.int8, device='cuda')
    sx = torch.rand(m, device='cuda', dtype=torch.float32) * 0.1 + 0.01
    sw = torch.rand(n, device='cuda', dtype=torch.float32) * 0.1 + 0.01
    bias = torch.randn(n, device='cuda', dtype=torch.float32)

    # 1. INT8 GEMM Validation
    w8 = torch.randint(-127, 128, (n, k), dtype=torch.int8, device='cuda')
    cfg8 = Config(bm=32, bn=64, bk=64, warps=4, stages=2, layout='nk')
    stored8 = prepare_weight(w8, 8, cfg8)

    ref_raw = (x.cpu().long() @ w8.cpu().long().T).cuda().int()
    ref_out = ref_raw.float() * sx[:, None] * sw[None, :] + bias[None, :]

    raw8 = gemm(x, stored8, sx, sw, bias, bits=8, config=cfg8, raw=True)
    out8 = gemm(x, stored8, sx, sw, bias, bits=8, config=cfg8, dtype=torch.float32)

    torch.testing.assert_close(raw8, ref_raw, rtol=0, atol=0)
    torch.testing.assert_close(out8, ref_out, rtol=2e-5, atol=2e-5)
    m8 = metrics(ref_out, out8)

    # 2. Packed INT4 GEMM Validation
    w4 = torch.randint(-8, 8, (n, k), dtype=torch.int8, device='cuda')
    cfg4 = Config(bm=32, bn=64, bk=64, warps=4, stages=2, layout='kn_packed')
    stored4 = prepare_weight(w4, 4, cfg4)

    ref_raw4 = (x.cpu().long() @ w4.cpu().long().T).cuda().int()
    ref_out4 = ref_raw4.float() * sx[:, None] * sw[None, :] + bias[None, :]

    raw4 = gemm(x, stored4, sx, sw, bias, bits=4, config=cfg4, raw=True)
    out4 = gemm(x, stored4, sx, sw, bias, bits=4, config=cfg4, dtype=torch.float32)

    torch.testing.assert_close(raw4, ref_raw4, rtol=0, atol=0)
    torch.testing.assert_close(out4, ref_out4, rtol=2e-5, atol=2e-5)
    m4 = metrics(ref_out4, out4)

    e8, l8, c8 = m8['max_abs_error'], m8['relative_l2'], m8['cosine_similarity']
    e4, l4, c4 = m4['max_abs_error'], m4['relative_l2'], m4['cosine_similarity']
    print(f'  {name} W8A8: RawIntDiff=0, MaxAbsErr={e8:.2e}, RelL2={l8:.2e}, CosSim={c8:.6f}')
    print(f'  {name} W4A8: RawIntDiff=0, MaxAbsErr={e4:.2e}, RelL2={l4:.2e}, CosSim={c4:.6f}')

print('ALL FOUR DIT SHAPES FULLY VALIDATED AT FULL SIZE (M=2048) WITH ZERO ERRORS!')
