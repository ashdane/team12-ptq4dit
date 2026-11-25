"""v3 fused-epilogue INT8 GEMMs (fused_gemm.py) vs exact float64 references.

Run: PYTHONPATH=src python3 tests/test_fused_gemm.py
"""
import unittest

import torch

CUDA = torch.cuda.is_available()
SHAPES = {"qkv": (3456, 1152), "proj": (1152, 1152), "fc1": (4608, 1152), "fc2": (1152, 4608)}


@unittest.skipUnless(CUDA, "CUDA required")
class FusedGemmTests(unittest.TestCase):
    def setUp(self):
        from fc1bench import fused_gemm
        from fc1bench.quantization import quantize_rows
        self.f = fused_gemm
        g = torch.Generator(device="cpu").manual_seed(0)
        self.rand = lambda *s, scale=1.0: (torch.randn(*s, generator=g) * scale).cuda()
        self.qrows = quantize_rows

    def operands(self, m, layer, grouped=False):
        n, k = SHAPES[layer]
        x = torch.randint(-127, 128, (m, k), device="cuda", dtype=torch.int8)
        sx = (self.rand(m, k // 128) if grouped else self.rand(m)).abs() * 0.01 + 1e-3
        wq, sw = self.qrows(self.rand(n, k) / k ** 0.5, 8)
        b = self.rand(n, scale=0.01).float()
        return x, sx.float().contiguous(), wq.contiguous(), sw, b

    def check_close(self, name, got, ref, rtol=2e-3):
        err = ((got.float() - ref).norm() / ref.norm()).item()
        print(f"{name}: rel L2 {err:.2e}")
        self.assertLess(err, rtol, name)

    def test_plain(self):
        for m in (2048, 16384):
            x, sx, w, sw, b = self.operands(m, "qkv")
            y = self.f.gemm_plain(x, sx, w, sw, b)
            self.check_close(f"plain qkv M={m}", y, self.f.ref_linear(x, sx, w, sw, b))

    def test_residual(self):
        tokens = 256
        for m in (2048, 16384):
            for layer, grouped, g in (("proj", False, 0), ("fc2", True, 128), ("fc2", True, 256)):
                x, sx, w, sw, b = self.operands(m, layer, grouped)
                if g == 256:
                    sx = sx[:, ::2].contiguous()
                n = w.shape[0]
                r = self.rand(m, n, scale=2.0).half()
                mods = self.rand(m // tokens, 28 * 6 * n, scale=0.5).half()
                off = 6 * n * 7 + 2 * n
                y = self.f.gemm_res(x, sx, w, sw, b, r, mods, off, tokens, grouped)
                lin = self.f.ref_linear(x, sx, w, sw, b, grouped)
                gate = mods.float()[torch.arange(m, device="cuda") // tokens, off:off + n]
                ref = r.float() + gate * lin
                self.check_close(f"residual {layer} group={g} M={m}", y, ref)

    def test_gelu_group_quant(self):
        for m, group in ((2048, 128), (16384, 128), (2048, 256), (16384, 256)):
            x, sx, w, sw, b = self.operands(m, "fc1")
            q, s = self.f.gemm_gelu_gq(x, sx, w, sw, b, group)
            qr, sr = self.f.ref_gelu_gq(self.f.ref_linear(x, sx, w, sw, b), group)
            dq = (q.int() - qr.int()).abs()
            srel = ((s - sr).abs() / sr).max().item()
            frac = (dq > 0).float().mean().item()
            print(f"gelu_gq M={m} G={group}: INT8 mismatch fraction {frac:.2e} (max |dq| {dq.max().item()}), "
                  f"scale max rel diff {srel:.2e}")
            self.assertLessEqual(dq.max().item(), 1)
            self.assertLess(frac, 2e-2)        # tanh.approx.f32 flips a small fraction by one step
            self.assertLess(srel, 2e-3)

    def test_mlp_chain_vs_per_token(self):
        """fc1 -> GELU -> group quant -> fc2 is at least as close to FP32 as per-token quant."""
        from fc1bench import fused_ops
        m = 4096
        x, sx, w1, sw1, b1 = self.operands(m, "fc1")
        _, _, w2, sw2, b2 = self.operands(m, "fc2")
        h = self.f.ref_linear(x, sx, w1, sw1, b1)
        g = torch.nn.functional.gelu(h, approximate="tanh")
        ref = (g.double() @ (w2.double() * sw2.double()[:, None]).T + b2.double()).float()
        q, s = self.f.gemm_gelu_gq(x, sx, w1, sw1, b1)
        r = torch.zeros(m, w2.shape[0], device="cuda", dtype=torch.float16)
        ones = torch.ones(m // 256, 6 * w2.shape[0], device="cuda", dtype=torch.float16)
        grouped = self.f.gemm_res(q, s, w2, sw2, b2, r, ones, 0, 256, True)
        qt, st = fused_ops.gelu_quant(h.half(), None, True)
        per_token = self.f.gemm_plain(qt, st, w2, sw2, b2)
        q2, s2 = self.f.gemm_gelu_gq(x, sx, w1, sw1, b1, 256)
        grouped2 = self.f.gemm_res(q2, s2, w2, sw2, b2, r, ones, 0, 256, True)
        eg = ((grouped.float() - ref).norm() / ref.norm()).item()
        eg2 = ((grouped2.float() - ref).norm() / ref.norm()).item()
        et = ((per_token.float() - ref).norm() / ref.norm()).item()
        print(f"MLP chain rel L2 vs FP32: group-128 {eg:.2e}, group-256 {eg2:.2e}, per-token {et:.2e}")
        self.assertLess(eg, et * 1.05)
        self.assertLess(eg2, et * 1.05)


if __name__ == "__main__":
    unittest.main(verbosity=2)
