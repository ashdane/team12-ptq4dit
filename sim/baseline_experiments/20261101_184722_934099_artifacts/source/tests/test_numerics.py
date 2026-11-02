import importlib.util
import unittest

HAS_TORCH = importlib.util.find_spec("torch") is not None
if HAS_TORCH:
    import torch
    import torch.nn.functional as F
    from fc1bench.quantization import metrics, pack_int4, quantize_rows, unpack_int4


@unittest.skipUnless(HAS_TORCH, "PyTorch is not installed")
class NumericsTests(unittest.TestCase):
    def test_all_signed_nibbles_and_odd_padding(self):
        for k in (1, 15, 16, 17, 1152):
            q = ((torch.arange(3 * k) % 16) - 8).to(torch.int8).reshape(3, k)
            packed = pack_int4(q)
            self.assertEqual(packed.numel(), 3 * ((k + 1) // 2))
            torch.testing.assert_close(unpack_int4(packed, k), q, rtol=0, atol=0)
        q = torch.tensor([[-8, -1, 0, 7]], dtype=torch.int8)
        self.assertEqual(pack_int4(q).tolist(), [[248, 112]])

    def test_invalid_int4(self):
        with self.assertRaises(ValueError):
            pack_int4(torch.tensor([[8]], dtype=torch.int8))

    def test_zero_scales_and_rounding(self):
        q, s = quantize_rows(torch.zeros(2, 4))
        self.assertTrue(bool((q == 0).all()))
        self.assertTrue(bool((s > 0).all()))
        x = torch.tensor([[127., .5, 1.5, 2.5, -.5, -1.5, -2.5]])
        q, s = quantize_rows(x)
        self.assertEqual(q.tolist(), [[127, 0, 2, 2, 0, -2, -2]])

    def test_metrics(self):
        x = torch.tensor([1., 2., 3.])
        self.assertEqual(metrics(x, x)["mse"], 0.)
        self.assertAlmostEqual(metrics(x, 2*x)["relative_l2"], 1.)
        self.assertEqual(metrics(torch.zeros(3), torch.zeros(3))["cosine_similarity"], 1.)
        with self.assertRaises(ValueError):
            metrics(x, torch.full_like(x, float("nan")))


HAS_CUDA = HAS_TORCH and torch.cuda.is_available() and importlib.util.find_spec("triton") is not None


@unittest.skipUnless(HAS_CUDA, "CUDA and Triton are required")
class CudaKernelTests(unittest.TestCase):
    def test_quantizer_matches_torch(self):
        from fc1bench.kernels import quantize_activation
        torch.manual_seed(42)
        for k in (1, 31, 65, 1152, 4608):
            x = torch.randn(7, k, device="cuda")
            x[0] = 0
            for scale in (1., 1e-9, 1e4):
                scaled = x * scale
                q, s = quantize_activation(scaled)
                qr, sr = quantize_rows(scaled)
                torch.testing.assert_close(q, qr, rtol=0, atol=0)
                torch.testing.assert_close(s, sr, rtol=0, atol=0)
        x = torch.tensor([[127., .5, 1.5, 2.5, -.5, -1.5, -2.5]], device="cuda")
        torch.testing.assert_close(quantize_activation(x)[0], quantize_rows(x)[0], rtol=0, atol=0)

    def test_integer_accumulator_and_epilogue(self):
        from fc1bench.kernels import integer_linear
        torch.manual_seed(9)
        for m, k, n in ((1, 1, 1), (33, 65, 67), (32, 128, 64), (7, 1152, 17)):
            for packed in (False, True):
                with self.subTest(shape=(m, k, n), packed=packed):
                    x = torch.randint(-128, 128, (m, k), dtype=torch.int8)
                    w = torch.randint(-8 if packed else -128, 8 if packed else 128,
                                      (n, k), dtype=torch.int8)
                    expected = x.to(torch.int64) @ w.to(torch.int64).T
                    x, w = x.cuda(), w.cuda()
                    stored = pack_int4(w) if packed else w
                    sx, sw = torch.rand(m, device="cuda"), torch.rand(n, device="cuda")
                    for bias in (None, torch.randn(n, device="cuda")):
                        raw = integer_linear(x, stored, sx, sw, bias, packed=packed, raw=True)
                        torch.testing.assert_close(raw.cpu().long(), expected, rtol=0, atol=0)
                        y = integer_linear(x, stored, sx, sw, bias, packed=packed)
                        reference = expected.cuda().float() * sx[:, None] * sw[None, :]
                        if bias is not None:
                            reference += bias
                        torch.testing.assert_close(y, reference, rtol=2e-6, atol=2e-5)

    def test_module_integration(self):
        from fc1bench.integration import ExperimentalLinear, replace_linears
        from fc1bench.kernels import integer_linear, quantize_activation
        for bits in (4, 8):
            source = torch.nn.Linear(65, 17).cuda().eval()
            replacement = ExperimentalLinear(source, bits).eval()
            x = torch.randn(2, 3, 65, device="cuda")
            with torch.inference_mode():
                y = replacement(x)
                q, s = quantize_activation(x.reshape(-1, 65))
                expected = integer_linear(q, replacement.weight, s, replacement.weight_scale,
                                          replacement.bias, packed=bits == 4).reshape(2, 3, 17)
            torch.testing.assert_close(y, expected)
            model = torch.nn.Sequential(source)
            replace_linears(model, ["0"], bits)
            self.assertIsInstance(model[0], ExperimentalLinear)


if __name__ == "__main__":
    unittest.main()
