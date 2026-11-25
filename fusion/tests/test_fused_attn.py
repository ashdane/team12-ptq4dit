"""Single-pass DiT attention (fused_attn.py) vs PyTorch SDPA, plus a timing comparison.

Run: PYTHONPATH=src python3 tests/test_fused_attn.py
"""
import unittest

import torch

CUDA = torch.cuda.is_available()


def timeit(fn, inner=20, samples=20):
    for _ in range(5):
        fn()
    torch.cuda.synchronize()
    out = []
    for _ in range(samples):
        a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        a.record()
        for _ in range(inner):
            fn()
        b.record()
        b.synchronize()
        out.append(a.elapsed_time(b) / inner)
    return sorted(out)[len(out) // 2]


@unittest.skipUnless(CUDA, "CUDA required")
class FusedAttnTests(unittest.TestCase):
    def test_matches_sdpa_and_time(self):
        from fc1bench.fused_attn import attention_ref, dit_attention
        torch.manual_seed(0)
        tokens, heads, c = 256, 16, 1152
        for n_img in (8, 64):
            qkv = (torch.randn(n_img * tokens, 3 * c, device="cuda") * 1.5).half()
            ref = attention_ref(qkv, n_img, tokens, heads)
            got = dit_attention(qkv, n_img, tokens, heads)
            err = ((got.float() - ref.float()).norm() / ref.float().norm()).item()
            t_ref = timeit(lambda: attention_ref(qkv, n_img, tokens, heads))
            t_new = timeit(lambda: dit_attention(qkv, n_img, tokens, heads))
            print(f"n_img={n_img}: rel L2 vs SDPA {err:.2e}; SDPA {t_ref * 1e3:.1f} us, "
                  f"fused {t_new * 1e3:.1f} us ({t_ref / t_new:.2f}x)")
            self.assertLess(err, 2e-3)


if __name__ == "__main__":
    unittest.main(verbosity=2)
