"""Fused producer+quant kernels (fused_ops.py v2) vs FP32 PyTorch references.

fast=False: exact grid (IEEE division per element, libdevice tanh) -> INT8 must match except rare
+-1 flips where reduction order moves an FP32 value across a rounding boundary.
fast=True: reciprocal-multiply quantization + tanh.approx.f32 -> compared against the matching
reciprocal reference; GELU additionally differs by the approximate tanh (small +-1 fraction).
Run: PYTHONPATH=src python3 tests/test_fused_ops.py
"""
import unittest

import torch

CUDA = torch.cuda.is_available()


@unittest.skipUnless(CUDA, "CUDA required")
class FusedOpsTests(unittest.TestCase):
    def setUp(self):
        from fc1bench import fused_ops
        self.f = fused_ops
        g = torch.Generator(device="cpu").manual_seed(0)
        self.rand = lambda *s, scale=1.0: (torch.randn(*s, generator=g) * scale)

    def compare(self, name, q_ref, s_ref, q, s, max_frac=1e-4):
        dq = (q_ref.int() - q.int()).abs()
        frac = (dq > 0).float().mean().item()
        srel = ((s - s_ref).abs() / s_ref).max().item()
        print(f"{name}: INT8 mismatch fraction {frac:.2e} (max |dq| {dq.max().item()}), scale max rel diff {srel:.2e}")
        self.assertLessEqual(dq.max().item(), 1, name)
        self.assertLess(frac, max_frac, name)
        self.assertLess(srel, 1e-4, name)

    def inputs(self, rows):
        tokens, c = 256, 1152
        n = rows // tokens
        x = (self.rand(rows, c, scale=3.0)).cuda().half()
        d = (self.rand(rows, c, scale=2.0)).cuda().half()
        mods = (self.rand(n, 28 * 6 * c, scale=0.5)).cuda().half()    # batched adaLN layout (strided views)
        mod, prev = mods[:, 6 * c * 3:6 * c * 4], mods[:, 6 * c * 2:6 * c * 3]
        return tokens, c, x, d, mod, prev

    def test_ln_and_res_ln(self):
        for rows in (2048, 16384):
            tokens, c, x, d, mod, prev = self.inputs(rows)
            for fast in (False, True):
                q, s = self.f.ln_mod_quant(x, mod, 0, c, tokens, 1e-6, fast)
                _, qr, sr = self.f.res_ln_mod_quant_ref(x, mod, 0, c, tokens, 1e-6, fast=fast)
                self.compare(f"ln_mod_quant M={rows} fast={fast}", qr, sr, q, s)
                xo, q, s = self.f.res_ln_mod_quant(x, d, prev, 5 * c, mod, 3 * c, 4 * c, tokens, 1e-6, fast)
                xr, qr, sr = self.f.res_ln_mod_quant_ref(x, mod, 3 * c, 4 * c, tokens, 1e-6, d, prev, 5 * c, fast)
                exact = (xo == xr).float().mean().item()
                print(f"  residual stream M={rows}: exact fraction {exact:.6f}")
                self.assertGreater(exact, 0.999)
                self.compare(f"res_ln_mod_quant M={rows} fast={fast}", qr, sr, q, s)

    def test_gelu_quant(self):
        for rows in (2048, 16384):
            h = (self.rand(rows, 4608, scale=2.0)).cuda().half()
            q, s = self.f.gelu_quant(h, None, False)
            qr, sr = self.f.act_quant_ref(h, True)
            self.compare(f"gelu_quant M={rows} exact", qr, sr, q, s, max_frac=5e-4)
            q, s = self.f.gelu_quant(h, None, True)
            qr, sr = self.f.act_quant_ref(h, True, fast=True)
            self.compare(f"gelu_quant M={rows} fast (tanh.approx)", qr, sr, q, s, max_frac=2e-2)
        bx = (self.rand(4608).abs() + 0.5).cuda().float()
        q, s = self.f.gelu_quant(h, bx, False)
        qr, sr = self.f.act_quant_ref(h, True, bx)
        self.compare("gelu_quant (bx) exact", qr, sr, q, s, max_frac=5e-4)

    def test_row_quant(self):
        from fc1bench.optimized import quantize_native
        x = (self.rand(16384, 1152, scale=3.0)).cuda().half()
        q, s = self.f.row_quant(x, False)
        qn, sn = quantize_native(x)
        self.assertTrue(torch.equal(q, qn) and torch.equal(s, sn))
        print("row_quant exact: bit-identical to quantize_native")
        q, s = self.f.row_quant(x, True)
        qr, sr = self.f.act_quant_ref(x, False, fast=True)
        self.compare("row_quant fast", qr, sr, q, s, max_frac=1e-5)


if __name__ == "__main__":
    unittest.main(verbosity=2)
