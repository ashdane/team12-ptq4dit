"""CUTLASS INT8 GEMMs (cutlass_gemm.py) vs float64 references, for every tile config that runs.

`tune` itself checks each config's output (rel L2 < 2e-3) before timing it, so tuning every DiT
shape for both epilogues exercises all configs. Needs nvcc and CUTLASS_DIR.
Run: PYTHONPATH=src python3 tests/test_cutlass_gemm.py
"""
import unittest

import torch

CUDA = torch.cuda.is_available()
SHAPES = {"qkv": (3456, 1152), "proj": (1152, 1152), "fc1": (4608, 1152), "fc2": (1152, 4608)}


@unittest.skipUnless(CUDA, "CUDA required")
class CutlassGemmTests(unittest.TestCase):
    def setUp(self):
        from fc1bench import cutlass_gemm, fused_gemm
        from fc1bench.quantization import quantize_rows
        self.c, self.f, self.qrows = cutlass_gemm, fused_gemm, quantize_rows
        torch.manual_seed(0)

    def operands(self, m, layer):
        n, k = SHAPES[layer]
        x = torch.randint(-127, 128, (m, k), device="cuda", dtype=torch.int8)
        sx = torch.rand(m, device="cuda") * 0.01 + 1e-3
        wq, sw = self.qrows(torch.randn(n, k, device="cuda") / k ** 0.5, 8)
        b = (torch.randn(n, device="cuda") * 0.01).float()
        return x, sx.float(), wq.contiguous(), sw, b

    def test_all_shapes_both_epilogues(self):
        tokens = 256
        for m in (2048, 16384):
            for layer in SHAPES:
                x, sx, w, sw, b = self.operands(m, layer)
                n = w.shape[0]
                cfg, rep = self.c.tune("plain", x, sx, w, sw, b, tokens)
                y = self.c.cutlass_gemm(x, sx, w, sw, b, tokens)
                ref = self.f.ref_linear(x, sx, w, sw, b)
                err = ((y.float() - ref).norm() / ref.norm()).item()
                self.assertLess(err, 2e-3)
                # per-image gate differs per image: checks the batched (L = images) indexing
                r = (torch.randn(m, n, device="cuda") * 2).half()
                mods = (torch.randn(m // tokens, 28 * 6 * n, device="cuda") * 0.5).half()
                off = 6 * n * 11 + 5 * n
                self.c.tune("res", x, sx, w, sw, b, tokens, r, mods, off)
                z = self.c.cutlass_gemm_res(x, sx, w, sw, b, r, mods, off, tokens)
                gate = mods.float()[torch.arange(m, device="cuda") // tokens, off:off + n]
                zref = r.float() + gate * ref
                zerr = ((z.float() - zref).norm() / zref.norm()).item()
                print(f"M={m} {layer}: plain rel L2 {err:.2e}, residual rel L2 {zerr:.2e}")
                self.assertLess(zerr, 2e-3)
                # no bias path
                y0 = self.c.launch(cfg, "plain", x, sx, w, sw, None, tokens)
                ref0 = self.f.ref_linear(x, sx, w, sw, None)
                self.assertLess(((y0.float() - ref0).norm() / ref0.norm()).item(), 2e-3)


if __name__ == "__main__":
    unittest.main(verbosity=2)
