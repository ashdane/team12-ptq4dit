import importlib.util
import unittest

HAS_TORCH = importlib.util.find_spec("torch") is not None
if HAS_TORCH:
    import torch
    import torch.nn.functional as F
    from fc1bench.quantization import (
        metrics, pack_int4, quantize_rows, unpack_int4,
        pack_int4_kn, unpack_int4_kn,
        quantize_activation_ref, int8_gemm_ref, int4_gemm_ref
    )


@unittest.skipUnless(HAS_TORCH, "PyTorch is not installed")
class NumericsTests(unittest.TestCase):
    def test_all_signed_nibbles_and_odd_padding(self):
        for k in (1, 15, 16, 17, 65, 1152):
            q = ((torch.arange(3 * k) % 16) - 8).to(torch.int8).reshape(3, k)
            packed = pack_int4(q)
            self.assertEqual(packed.numel(), 3 * ((k + 1) // 2))
            torch.testing.assert_close(unpack_int4(packed, k), q, rtol=0, atol=0)
        q = torch.tensor([[-8, -1, 0, 7]], dtype=torch.int8)
        self.assertEqual(pack_int4(q).tolist(), [[248, 112]])

    def test_kn_packed_int4_roundtrip(self):
        for n, k in ((16, 64), (17, 65), (1152, 2048), (3456, 1152)):
            q = torch.randint(-8, 8, (n, k), dtype=torch.int8)
            packed = pack_int4_kn(q)
            self.assertEqual(packed.shape, (k, (n + 1) // 2))
            unpacked = unpack_int4_kn(packed, n, k)
            torch.testing.assert_close(unpacked, q, rtol=0, atol=0)

    def test_invalid_int4(self):
        with self.assertRaises(ValueError):
            pack_int4(torch.tensor([[8]], dtype=torch.int8))
        with self.assertRaises(ValueError):
            pack_int4(torch.tensor([[-9]], dtype=torch.int8))

    def test_zero_scales_and_rounding(self):
        q, s = quantize_rows(torch.zeros(2, 4))
        self.assertTrue(bool((q == 0).all()))
        self.assertTrue(bool((s > 0).all()))
        x = torch.tensor([[127., .5, 1.5, 2.5, -.5, -1.5, -2.5]])
        q, s = quantize_rows(x)
        self.assertEqual(q.tolist(), [[127, 0, 2, 2, 0, -2, -2]])

    def test_fused_activation_quantization_ref(self):
        torch.manual_seed(42)
        for k in (65, 1152, 4608):
            x = torch.randn(8, k)
            # Test symmetric
            q, s = quantize_activation_ref(x)
            self.assertEqual(q.shape, (8, k))
            self.assertEqual(s.shape, (8,))
            self.assertTrue(bool((q >= -127).all() and (q <= 127).all()))

            # Test with CSB channel scaling
            bx = torch.rand(k) + 0.5
            q_bx, s_bx = quantize_activation_ref(x, bx=bx)
            self.assertEqual(q_bx.shape, (8, k))
            self.assertEqual(s_bx.shape, (8,))

            # Test affine asymmetric mode (signed int8 in [-128, 127])
            q_aff, s_aff, zp_aff = quantize_activation_ref(x, affine=True)
            self.assertEqual(q_aff.shape, (8, k))
            self.assertEqual(q_aff.dtype, torch.int8)
            self.assertEqual(zp_aff.dtype, torch.int32)
            self.assertTrue(bool((q_aff >= -128).all() and (q_aff <= 127).all()))
            self.assertTrue(bool((zp_aff >= -128).all() and (zp_aff <= 127).all()))
            # Parity check: reconstruct activations
            x_rec = (q_aff.float() - zp_aff.float()[:, None]) * s_aff[:, None]
            self.assertLess((x - x_rec).abs().max().item(), s_aff.max().item())

    def test_affine_activation_dtype_and_range(self):
        torch.manual_seed(999)
        # Test extreme ranges and post-GELU shape (min around -0.17, large positive tails)
        x = torch.randn(12, 128) * 8.0
        x[0, :10] = -0.1702
        q, s, zp = quantize_activation_ref(x, affine=True)
        self.assertEqual(q.dtype, torch.int8)
        self.assertEqual(zp.dtype, torch.int32)
        self.assertEqual(s.dtype, torch.float32)
        self.assertTrue(bool((q >= -128).all() and (q <= 127).all()))
        self.assertTrue(bool((zp >= -128).all() and (zp <= 127).all()))
        # Check that x=0 maps exactly to zp
        x_with_zero = torch.zeros(1, 16)
        q_z, s_z, zp_z = quantize_activation_ref(x_with_zero, affine=True)
        torch.testing.assert_close(q_z.int(), zp_z[:, None].expand(-1, 16), rtol=0, atol=0)

    def test_rounding_boundaries_nearest_even(self):
        # Round-to-nearest-even (banker's rounding) test vectors
        # Halfway cases: ties must break to the nearest even number
        halfways = torch.tensor([
            0.5, 1.5, 2.5, 3.5, 4.5,
            -0.5, -1.5, -2.5, -3.5, -4.5
        ])
        expected_even = torch.tensor([
            0.0, 2.0, 2.0, 4.0, 4.0,
            0.0, -2.0, -2.0, -4.0, -4.0
        ])
        torch.testing.assert_close(torch.round(halfways), expected_even, rtol=0, atol=0)

        # Values immediately below and above the halfway boundary
        below_above = torch.tensor([
            0.4999, 0.5001,
            1.4999, 1.5001,
            2.4999, 2.5001,
            -0.4999, -0.5001,
            -1.4999, -1.5001,
            -2.4999, -2.5001,
        ])
        expected_ba = torch.tensor([
            0.0, 1.0,
            1.0, 2.0,
            2.0, 3.0,
            0.0, -1.0,
            -1.0, -2.0,
            -2.0, -3.0,
        ])
        torch.testing.assert_close(torch.round(below_above), expected_ba, rtol=0, atol=0)

    def test_zero_valued_rows(self):
        zeros = torch.zeros(5, 64)
        # 1. Symmetric
        q_sym, s_sym = quantize_activation_ref(zeros, affine=False)
        self.assertTrue(bool((q_sym == 0).all()))
        self.assertTrue(bool((s_sym >= 1e-8).all()))
        self.assertTrue(bool(torch.isfinite(s_sym).all()))

        # 2. Affine
        q_aff, s_aff, zp_aff = quantize_activation_ref(zeros, affine=True)
        self.assertTrue(bool((s_aff >= 1e-8).all()))
        torch.testing.assert_close(q_aff.int(), zp_aff[:, None].expand(-1, 64), rtol=0, atol=0)
        reconstructed = (q_aff.float() - zp_aff.float()[:, None]) * s_aff[:, None]
        torch.testing.assert_close(reconstructed, zeros, rtol=0, atol=0)

    def test_odd_k_dimensions(self):
        torch.manual_seed(77)
        for k in (1, 3, 17, 65, 1151, 1153):
            m, n = 8, 16
            xq = torch.randint(-128, 128, (m, k), dtype=torch.int8)
            wq = torch.randint(-8, 8, (n, k), dtype=torch.int8)
            sx = torch.rand(m) * 0.1 + 0.01
            sw = torch.rand(n) * 0.1 + 0.01
            bias = torch.randn(n)

            # INT4 packing round trip on odd K
            packed_nk = pack_int4(wq)
            unpacked_nk = unpack_int4(packed_nk, k)
            torch.testing.assert_close(unpacked_nk, wq, rtol=0, atol=0)

            packed_kn = pack_int4_kn(wq)
            unpacked_kn = unpack_int4_kn(packed_kn, n, k)
            torch.testing.assert_close(unpacked_kn, wq, rtol=0, atol=0)

            # GEMM on odd K
            y_ref = int4_gemm_ref(xq, packed_kn, sx, sw, bias=bias, layout="kn_packed")
            expected = (xq.long() @ wq.long().T).float() * sx[:, None] * sw[None, :] + bias[None, :]
            torch.testing.assert_close(y_ref, expected, rtol=1e-5, atol=1e-5)

    def test_zero_point_correction_actual_operand_representation(self):
        torch.manual_seed(42)
        m, k, n = 16, 128, 32
        # Signed int8 activations with signed zero points
        xq = torch.randint(-128, 128, (m, k), dtype=torch.int8)
        wq = torch.randint(-127, 128, (n, k), dtype=torch.int8)
        zx = torch.randint(-128, 128, (m,), dtype=torch.int32)
        sx = torch.rand(m) * 0.05 + 0.01
        sw = torch.rand(n) * 0.05 + 0.01
        bias = torch.randn(n)

        # 1. Ground truth in float64 dequantized domain
        x_deq = (xq.double() - zx.double()[:, None]) * sx.double()[:, None]
        w_deq = wq.double() * sw.double()[:, None]
        expected_y = (x_deq @ w_deq.T + bias.double()[None, :]).float()

        # 2. Integer reference GEMM with zero point correction
        actual_y = int8_gemm_ref(xq, wq, sx, sw, bias=bias, zx=zx)
        torch.testing.assert_close(actual_y, expected_y, rtol=1e-5, atol=2e-5)

    def test_invalid_dtypes_and_unsupported_combinations(self):
        xq = torch.randint(-128, 128, (4, 16), dtype=torch.int8)
        wq = torch.randint(-128, 128, (8, 16), dtype=torch.int8)
        sx = torch.rand(4)
        sw = torch.rand(8)

        # Float activation passed to integer GEMM
        with self.assertRaises(TypeError):
            int8_gemm_ref(xq.float(), wq, sx, sw)
        # Float weight passed to integer GEMM
        with self.assertRaises(TypeError):
            int8_gemm_ref(xq, wq.float(), sx, sw)
        # K > 131071 (unsafe for INT32 accumulation)
        with self.assertRaises(ValueError):
            int8_gemm_ref(torch.zeros(1, 131072, dtype=torch.int8),
                          torch.zeros(1, 131072, dtype=torch.int8),
                          torch.ones(1), torch.ones(1))

    def test_all_four_dit_shapes_integer_gemm_ref(self):
        # Workloads: (M, K, N)
        workloads = {
            "attn.qkv": (2048, 1152, 3456),
            "attn.proj": (2048, 1152, 1152),
            "mlp.fc1": (2048, 1152, 4608),
            "mlp.fc2": (2048, 4608, 1152),
            "irregular": (129, 65, 67),
        }
        torch.manual_seed(123)
        for name, (m, k, n) in workloads.items():
            # Use smaller M for speedy CPU reference test while keeping exact K and N
            test_m = 16 if m > 16 else m
            xq = torch.randint(-127, 128, (test_m, k), dtype=torch.int8)
            wq = torch.randint(-127, 128, (n, k), dtype=torch.int8)
            sx = torch.rand(test_m) * 0.1 + 0.01
            sw = torch.rand(n) * 0.1 + 0.01
            bias = torch.randn(n)

            # 1. Raw accumulator
            raw = int8_gemm_ref(xq, wq, sx, sw, raw=True)
            expected_raw = (xq.long() @ wq.long().T).int()
            torch.testing.assert_close(raw, expected_raw, rtol=0, atol=0)

            # 2. Epilogue with scaling and bias
            y_fp32 = int8_gemm_ref(xq, wq, sx, sw, bias=bias, dtype=torch.float32)
            ref_fp32 = expected_raw.float() * sx[:, None] * sw[None, :] + bias[None, :]
            torch.testing.assert_close(y_fp32, ref_fp32, rtol=1e-5, atol=1e-5)

            # 3. Packed INT4 GEMM
            w4 = torch.randint(-8, 8, (n, k), dtype=torch.int8)
            packed_kn = pack_int4_kn(w4)
            y_int4 = int4_gemm_ref(xq, packed_kn, sx, sw, bias=bias, layout="kn_packed")
            ref_int4 = (xq.long() @ w4.long().T).float() * sx[:, None] * sw[None, :] + bias[None, :]
            torch.testing.assert_close(y_int4, ref_int4, rtol=1e-5, atol=1e-5)

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
                qr, sr = quantize_activation_ref(scaled)
                torch.testing.assert_close(q, qr, rtol=0, atol=0)
                torch.testing.assert_close(s, sr, rtol=0, atol=0)

    def test_quantizer_with_csb_scaling(self):
        from fc1bench.kernels import quantize_activation
        torch.manual_seed(99)
        for k in (65, 1152, 4608):
            x = torch.randn(8, k, device="cuda")
            bx = torch.rand(k, device="cuda") + 0.5
            q, s = quantize_activation(x, bx=bx)
            qr, sr = quantize_activation_ref(x, bx=bx)
            torch.testing.assert_close(q, qr, rtol=0, atol=0)
            torch.testing.assert_close(s, sr, rtol=1e-5, atol=1e-5)

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

    def test_all_four_dit_shapes_triton_gemm(self):
        from fc1bench.optimized import Config, gemm, prepare_weight
        shapes = [
            ("attn.qkv", 64, 1152, 3456),
            ("attn.proj", 64, 1152, 1152),
            ("mlp.fc1", 64, 1152, 4608),
            ("mlp.fc2", 64, 4608, 1152),
            ("irregular", 65, 65, 67),
        ]
        torch.manual_seed(2026)
        for name, m, k, n in shapes:
            for bits in (8, 4):
                with self.subTest(shape=name, bits=bits):
                    x = torch.randint(-127, 128, (m, k), dtype=torch.int8, device="cuda")
                    w = torch.randint(-8 if bits == 4 else -127, 8 if bits == 4 else 128, (n, k), dtype=torch.int8, device="cuda")
                    sx = torch.rand(m, device="cuda") * 0.1 + 0.01
                    sw = torch.rand(n, device="cuda") * 0.1 + 0.01
                    bias = torch.randn(n, device="cuda")

                    layout = "kn_packed" if bits == 4 else "nk"
                    config = Config(bm=32, bn=64, bk=64, warps=4, stages=2, layout=layout)
                    stored = prepare_weight(w, bits, config)

                    # 1. Raw accumulator parity
                    expected_raw = (x.cpu().long() @ w.cpu().long().T).cuda().int()
                    raw = gemm(x, stored, sx, sw, bias, bits=bits, config=config, raw=True)
                    torch.testing.assert_close(raw, expected_raw, rtol=0, atol=0)

                    # 2. Fused scaling & bias parity
                    out = gemm(x, stored, sx, sw, bias, bits=bits, config=config, dtype=torch.float32)
                    ref = expected_raw.float() * sx[:, None] * sw[None, :] + bias[None, :]
                    torch.testing.assert_close(out, ref, rtol=2e-5, atol=2e-5)


if __name__ == "__main__":
    unittest.main()
