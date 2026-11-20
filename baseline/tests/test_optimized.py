import importlib.util
import unittest

HAS_TORCH = importlib.util.find_spec("torch") is not None
if HAS_TORCH:
    import torch
    from fc1bench.optimized import Config, candidates, prepare_weight
    from fc1bench.quantization import pack_int4, unpack_int4, pack_int4_kn, unpack_int4_kn


@unittest.skipUnless(HAS_TORCH, "PyTorch is required")
class CPUOptimizedTests(unittest.TestCase):
    def test_candidates_generator(self):
        w8_configs = list(candidates(8))
        self.assertGreater(len(w8_configs), 0)
        self.assertTrue(any(c.backend == "triton" for c in w8_configs))
        w4_configs = list(candidates(4))
        self.assertGreater(len(w4_configs), 0)
        self.assertTrue(all(c.layout in ("nk", "kn_packed") for c in w4_configs if c.backend == "triton"))

    def test_prepare_weight_shapes(self):
        n, k = 1152, 4608
        q = torch.randint(-8, 8, (n, k), dtype=torch.int8)

        # 8-bit nk
        w8 = prepare_weight(q, 8, Config(layout="nk"))
        self.assertEqual(w8.shape, (n, k))
        self.assertEqual(w8.dtype, torch.int8)

        # 4-bit nk
        w4_nk = prepare_weight(q, 4, Config(layout="nk"))
        self.assertEqual(w4_nk.shape, (n, (k + 1) // 2))
        self.assertEqual(w4_nk.dtype, torch.uint8)

        # 4-bit kn_packed
        w4_kn = prepare_weight(q, 4, Config(layout="kn_packed"))
        self.assertEqual(w4_kn.shape, (k, (n + 1) // 2))
        self.assertEqual(w4_kn.dtype, torch.uint8)

    def test_zero_point_correction_parity(self):
        m, k, n = 16, 64, 32
        # Use actual signed int8 representation for both xq and wq
        xq = torch.randint(-128, 128, (m, k), dtype=torch.int8)
        wq = torch.randint(-127, 128, (n, k), dtype=torch.int8)
        zx = torch.randint(-128, 128, (m,), dtype=torch.int32)
        sx = torch.rand(m) * 0.05 + 0.01
        sw = torch.rand(n) * 0.05 + 0.01
        bias = torch.randn(n)

        # Mathematical ground truth: dequantize activation as (xq - zx) * sx, and weight as wq * sw in float64
        x_dequant = (xq.double() - zx.double()[:, None]) * sx.double()[:, None]
        w_dequant = wq.double() * sw.double()[:, None]
        expected_y = (x_dequant @ w_dequant.T + bias.double()[None, :]).float()

        # Integer GEMM epilogue formula:
        acc = (xq.long() @ wq.long().T).int()
        w_red = wq.float().sum(dim=1)
        acc_corr = acc.float() - zx.float()[:, None] * w_red[None, :]
        actual_y = acc_corr * sx[:, None] * sw[None, :] + bias[None, :]

        torch.testing.assert_close(actual_y, expected_y, rtol=1e-5, atol=2e-5)

    def test_w_red_precomputation_for_int4_and_int8(self):
        n, k = 32, 64
        w_int8 = torch.randint(-8, 8, (n, k), dtype=torch.int8)
        expected_w_red = w_int8.float().sum(dim=1)

        # 1. 8-bit nk
        w8 = prepare_weight(w_int8, 8, Config(layout="nk"))
        w_red_8 = w8.float().sum(dim=1)
        torch.testing.assert_close(w_red_8, expected_w_red, rtol=0, atol=0)

        # 2. 4-bit nk unpacked reduction
        w4_nk = prepare_weight(w_int8, 4, Config(layout="nk"))
        w_unpacked_nk = unpack_int4(w4_nk, k)
        w_red_4_nk = w_unpacked_nk.float().sum(dim=1)
        torch.testing.assert_close(w_red_4_nk, expected_w_red, rtol=0, atol=0)

        # 3. 4-bit kn_packed unpacked reduction
        w4_kn = prepare_weight(w_int8, 4, Config(layout="kn_packed"))
        w_unpacked_kn = unpack_int4_kn(w4_kn, n, k)
        w_red_4_kn = w_unpacked_kn.float().sum(dim=1)
        torch.testing.assert_close(w_red_4_kn, expected_w_red, rtol=0, atol=0)

    def test_unsupported_configurations_raise(self):
        q = torch.randint(-8, 8, (16, 32), dtype=torch.int8)
        # 8-bit does not support kn_packed
        with self.assertRaises(ValueError):
            prepare_weight(q, 8, Config(layout="kn_packed"))
        # packed INT4 does not support torch_int_mm
        with self.assertRaises(ValueError):
            prepare_weight(q, 4, Config(backend="torch_int_mm"))


HAS_CUDA = HAS_TORCH and torch.cuda.is_available() and importlib.util.find_spec("triton") is not None


@unittest.skipUnless(HAS_CUDA, "CUDA and Triton are required")
class OptimizedTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(912)
        torch.backends.cuda.matmul.allow_tf32 = False

    def test_every_triton_candidate_with_tails_and_groups(self):
        from fc1bench.optimized import candidates, gemm, prepare_weight
        # Includes multiple M groups and partial M/N/K tiles. Signed extrema
        # exercise -128 and -8, even though symmetric absmax avoids those codes.
        m, n, k = 257, 67, 65
        a = torch.randint(-128, 128, (m, k), dtype=torch.int8)
        for bits in (8, 4):
            w = torch.randint(-128 if bits == 8 else -8, 128 if bits == 8 else 8,
                              (n, k), dtype=torch.int8)
            expected = (a.long() @ w.long().T).cuda().int()
            x, w = a.cuda(), w.cuda()
            sx, sw = torch.rand(m, device="cuda"), torch.rand(n, device="cuda")
            bias = torch.randn(n, device="cuda")
            ref = expected.float()*sx[:, None]*sw[None, :]+bias
            for config in candidates(bits):
                if config.backend != "triton":
                    continue
                with self.subTest(bits=bits, config=config):
                    stored = prepare_weight(w, bits, config)
                    raw = gemm(x, stored, sx, sw, bias, bits=bits, config=config, raw=True)
                    torch.testing.assert_close(raw, expected, rtol=0, atol=0)
                    out = gemm(x, stored, sx, sw, bias, bits=bits, config=config)
                    torch.testing.assert_close(out, ref, rtol=2e-5, atol=2e-5)

    def test_native_dtypes_quantization_and_store(self):
        from fc1bench.optimized import Config, gemm, prepare_weight, quantize_native
        from fc1bench.quantization import quantize_rows
        # sm_80+ (Ampere/Hopper/Blackwell) supports native bfloat16 PTX instructions in Triton; sm_75 supports fp32 and fp16.
        dtypes = [torch.float32, torch.float16]
        if torch.cuda.get_device_capability()[0] >= 8:
            dtypes.append(torch.bfloat16)
        for dtype in dtypes:
            x = torch.randn(17, 129, device="cuda", dtype=dtype)
            x[0] = 0
            a, sx = quantize_native(x)
            ar, sr = quantize_rows(x)
            torch.testing.assert_close(a, ar, rtol=0, atol=0)
            torch.testing.assert_close(sx, sr, rtol=0, atol=0)
            for bits in (8, 4):
                w, sw = quantize_rows(torch.randn(31, 129, device="cuda"), bits)
                config = Config(layout="nk" if bits == 8 else "kn_packed")
                raw = (a.cpu().long() @ w.cpu().long().T).cuda().int()
                ref = (raw.float()*sx[:, None]*sw[None, :]).to(dtype)
                out = gemm(a, prepare_weight(w, bits, config), sx, sw,
                           bits=bits, config=config, dtype=dtype)
                rtol = {torch.float32: 2e-5, torch.float16: 1e-3, torch.bfloat16: 1e-2}[dtype]
                torch.testing.assert_close(out.float(), ref.float(), rtol=rtol, atol=2e-5)

    def test_optional_torch_integer_backend(self):
        if not hasattr(torch, "_int_mm"):
            self.skipTest("torch._int_mm is unavailable")
        from fc1bench.optimized import Config, gemm
        a = torch.randint(-128, 128, (32, 128), dtype=torch.int8, device="cuda")
        w = torch.randint(-128, 128, (64, 128), dtype=torch.int8, device="cuda")
        expected = (a.cpu().long() @ w.cpu().long().T).cuda().int()
        sx, sw = torch.ones(32, device="cuda"), torch.ones(64, device="cuda")
        config = Config(backend="torch_int_mm")
        raw = gemm(a, w, sx, sw, config=config, raw=True)
        torch.testing.assert_close(raw, expected, rtol=0, atol=0)
        out = gemm(a, w, sx, sw, config=config)
        torch.testing.assert_close(out, expected.float(), rtol=0, atol=0)

    def test_tuned_linear_fake_weight_layout_and_shape_gate(self):
        from fc1bench.model_layers import TunedLinear
        from fc1bench.optimized import Config
        from fc1bench.quantization import quantize_rows
        for bits in (8, 4):
            w = torch.randn(33, 65, device="cuda")
            q, _ = quantize_rows(w, bits)
            config = Config(layout="nk" if bits == 8 else "kn_packed")
            layer = TunedLinear(w, None, bits, config, 6).eval()
            torch.testing.assert_close(layer.unpack(), q, rtol=0, atol=0)
            x = torch.randn(2, 3, 65, device="cuda")
            with torch.inference_mode():
                actual = layer(x)
                reference = layer.reference(x)
                torch.testing.assert_close(actual, reference, rtol=2e-4, atol=2e-5)
                with self.assertRaises(ValueError):
                    layer(x[:1])

    def test_rng_restored_and_replayed(self):
        from fc1bench.eval_utils import matched_rng
        before_cpu = torch.get_rng_state().clone()
        before_gpu = torch.cuda.get_rng_state().clone()
        with matched_rng(131):
            first = torch.randn(19, device="cuda")
            first_cpu = torch.randn(3)
        torch.testing.assert_close(torch.get_rng_state(), before_cpu, rtol=0, atol=0)
        torch.testing.assert_close(torch.cuda.get_rng_state(), before_gpu, rtol=0, atol=0)
        with matched_rng(131):
            torch.testing.assert_close(torch.randn(19, device="cuda"), first, rtol=0, atol=0)
            torch.testing.assert_close(torch.randn(3), first_cpu, rtol=0, atol=0)

    def test_tune_and_bank_roundtrip(self):
        import tempfile
        from pathlib import Path
        from fc1bench.tuning import Bank, tune
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            # Reuse (257, 67, 65) FP32 so Triton hits the JIT cache from the first test.
            x = torch.randn(257, 65, device="cuda", dtype=torch.float32)
            w = torch.randn(67, 65, device="cuda")
            b = torch.randn(67, device="cuda")
            bank = Bank(root / "bank.json")
            for bits in (8, 4):
                config, report = tune(x, w, b, bits, root, rounds=2, inner=1)
                self.assertEqual(report["status"], "ok")
                bank.add(257, 67, 65, bits, torch.float32, config)
            reloaded = Bank(root / "bank.json")
            self.assertEqual(reloaded.get(257, 67, 65, 8, torch.float32),
                             bank.get(257, 67, 65, 8, torch.float32))
            self.assertEqual(reloaded.get(257, 67, 65, 4, torch.float32),
                             bank.get(257, 67, 65, 4, torch.float32))
