"""CUTLASS INT8 GEMMs with fused scale / bias / gated-residual epilogues (csrc/cutlass_int8).

Built on first use with torch.utils.cpp_extension (needs nvcc and a CUTLASS checkout; set
CUTLASS_DIR, default /content/cutlass). Each tile configuration is a separate translation unit so
ninja compiles them in parallel, for the current GPU architecture only.

  cutlass_gemm(x, sx, w, sw, bias, tokens)                       -> FP16 [M, N]
      y = sx[row] * (x @ w^T)[row, col] * sw[col] + bias[col]
  cutlass_gemm_res(x, sx, w, sw, bias, r, gate, g_off, tokens)   -> FP16 [M, N]
      r + gate[row // tokens, g_off + col] * y

The tile configuration for each (epilogue, M, N, K) is chosen by `tune`, which times every config
that builds and launches on this GPU inside a CUDA graph and checks it against a float64 reference.
Choices are kept in memory and optionally in a JSON file (CUTLASS_BANK).
"""
import json
import os
import statistics
from pathlib import Path
from typing import Optional

import torch

# (BM, BN, BK, WM, WN, stages); warp K = BK; instruction 16x8x32 (INT8 mma.sync)
CONFIGS = [
    (128, 128, 64, 64, 64, 5),
    (128, 256, 64, 64, 64, 3),
    (256, 128, 64, 64, 64, 3),
    (64, 128, 128, 32, 64, 4),
    (128, 128, 128, 64, 64, 3),
    (64, 64, 128, 32, 32, 5),
    (128, 64, 64, 64, 32, 6),
    (64, 256, 64, 64, 64, 4),
    (128, 128, 64, 64, 64, 3),
    (64, 128, 64, 32, 64, 6),
]
EPIS = ("plain", "res")
_EXT = None
_BANK = {}
_ROOT = Path(__file__).resolve().parents[2] / "csrc" / "cutlass_int8"


def _sources(build):
    build.mkdir(parents=True, exist_ok=True)
    srcs, decls, rows = [], [], []
    for i, (bm, bn, bk, wm, wn, st) in enumerate(CONFIGS):
        for e, epi in enumerate(EPIS):
            fn = f"run_{i}_{epi}"
            path = build / f"inst_{i}_{epi}.cu"
            text = (f'#include "t12_gemm.cuh"\nnamespace t12 {{\nint {fn}(T12_ARGS) {{\n'
                    f'  return run<Kernel<{bm}, {bn}, {bk}, {wm}, {wn}, {st}, {"true" if epi == "res" else "false"}>>'
                    f'(T12_CALL);\n}}\n}}\n')
            if not path.exists() or path.read_text() != text:
                path.write_text(text)
            srcs.append(str(path))
            decls.append(f"int {fn}(T12_ARGS);")
        rows.append("{" + ", ".join(f"&run_{i}_{epi}" for epi in EPIS) + "}")
    binding = build / "binding.cu"
    text = ('#include "t12_gemm.cuh"\nnamespace t12 {\n' + "\n".join(decls) + "\n"
            "using Fn = int (*)(T12_ARGS);\n"
            f"static Fn table[{len(CONFIGS)}][{len(EPIS)}] = {{" + ", ".join(rows) + "};\n}\n"
            "int t12_gemm(int64_t cfg, int64_t epi, T12_ARGS) {\n"
            "  return t12::table[cfg][epi](T12_CALL);\n}\n"
            "PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) { m.def(\"gemm\", &t12_gemm); }\n")
    if not binding.exists() or binding.read_text() != text:
        binding.write_text(text)
    return [str(binding)] + srcs


def ext():
    global _EXT
    if _EXT is None:
        from torch.utils.cpp_extension import load
        major, minor = torch.cuda.get_device_capability()
        os.environ["TORCH_CUDA_ARCH_LIST"] = f"{major}.{minor}"
        os.environ.setdefault("MAX_JOBS", str(os.cpu_count() or 4))
        cutlass = Path(os.environ.get("CUTLASS_DIR", "/content/cutlass"))
        build = Path(os.environ.get("T12_CUTLASS_BUILD", Path.home() / ".cache" / f"t12_cutlass_sm{major}{minor}"))
        _EXT = load(name=f"t12_cutlass_sm{major}{minor}", sources=_sources(build / "src"),
                    extra_include_paths=[str(_ROOT), str(cutlass / "include"), str(cutlass / "tools/util/include")],
                    extra_cuda_cflags=["-O3", "-std=c++20", "--expt-relaxed-constexpr", "-DNDEBUG",
                                       "-Xcompiler=-Wno-psabi"],
                    build_directory=str(build), verbose=False)
    return _EXT


def _check(x, sx, w, sw, tokens):
    if x.dtype != torch.int8 or w.dtype != torch.int8 or not (x.is_contiguous() and w.is_contiguous()):
        raise ValueError("expected contiguous INT8 x [M, K] and w [N, K]")
    if x.shape[0] % tokens or sx.shape != (x.shape[0],) or sw.shape != (w.shape[0],):
        raise ValueError("bad shapes for cutlass gemm")


def launch(cfg, epi, x, sx, w, sw, bias, tokens, r=None, gate=None, g_off=0, out=None):
    _check(x, sx, w, sw, tokens)
    d = out if out is not None else torch.empty((x.shape[0], w.shape[0]), device=x.device, dtype=torch.float16)
    res = epi == "res"
    status = ext().gemm(cfg, EPIS.index(epi), x, w, sx, sw, bias if bias is not None else sw, bias is not None,
                        d, r if res else d, gate if res else d, g_off, gate.stride(0) if res else 0, tokens)
    if status:
        raise RuntimeError(f"CUTLASS config {cfg} {CONFIGS[cfg]} failed with status {status}")
    return d


def _graph_time(fn, inner=10, samples=15):
    fn()
    torch.cuda.synchronize()
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        fn()
    torch.cuda.current_stream().wait_stream(s)
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        for _ in range(inner):
            fn()
    for _ in range(3):
        g.replay()
    torch.cuda.synchronize()
    out = []
    for _ in range(samples):
        a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        a.record()
        g.replay()
        b.record()
        b.synchronize()
        out.append(a.elapsed_time(b) / inner)
    return statistics.median(out)


def _bank_path():
    return os.environ.get("CUTLASS_BANK")


def _load_bank():
    p = _bank_path()
    if p and Path(p).exists() and not _BANK:
        _BANK.update({tuple(json.loads(k)): v for k, v in json.load(open(p)).items()})


def _save_bank():
    p = _bank_path()
    if p:
        json.dump({json.dumps(list(k)): v for k, v in _BANK.items()}, open(p, "w"), indent=1)


def tune(epi, x, sx, w, sw, bias, tokens, r=None, gate=None, g_off=0, verbose=True):
    """Pick the fastest correct config for this (epi, M, N, K); returns (cfg, report)."""
    _load_bank()
    m, k = x.shape
    n = w.shape[0]
    key = (epi, m, n, k)
    if key in _BANK:
        return _BANK[key]["cfg"], _BANK[key]
    from .fused_gemm import ref_linear
    ref = ref_linear(x, sx, w, sw, bias)
    if epi == "res":
        rows = torch.arange(m, device=x.device) // tokens
        ref = r.float() + gate.float()[rows, g_off:g_off + n] * ref
    times = {}
    for cfg in range(len(CONFIGS)):
        try:
            y = launch(cfg, epi, x, sx, w, sw, bias, tokens, r, gate, g_off)
            torch.cuda.synchronize()
        except RuntimeError as exc:
            times[cfg] = f"unsupported: {exc}"
            continue
        err = ((y.float() - ref).norm() / ref.norm()).item()
        if not err < 2e-3:
            raise AssertionError(f"CUTLASS config {CONFIGS[cfg]} {epi} wrong: rel L2 {err:.3e}")
        times[cfg] = _graph_time(lambda c=cfg: launch(c, epi, x, sx, w, sw, bias, tokens, r, gate, g_off))
    ok = {c: t for c, t in times.items() if isinstance(t, float)}
    if not ok:
        raise RuntimeError(f"no CUTLASS config runs for {key}")
    best = min(ok, key=ok.get)
    rep = dict(cfg=best, tile=CONFIGS[best], ms=ok[best], all_ms={str(CONFIGS[c]): t for c, t in times.items()})
    _BANK[key] = rep
    _save_bank()
    if verbose:
        print(f"  cutlass {epi} M={m} N={n} K={k}: {CONFIGS[best]} {ok[best] * 1e3:.1f} us "
              f"({2 * m * n * k / ok[best] / 1e9:.0f} TOPS)", flush=True)
    return best, rep


def _cfg(epi, m, n, k):
    _load_bank()
    ent = _BANK.get((epi, m, n, k))
    if ent is None:
        raise RuntimeError(f"CUTLASS shape {(epi, m, n, k)} not tuned; call cutlass_gemm.tune first")
    return ent["cfg"]


# --------------------------------------------------------------------------- custom ops
@torch.library.custom_op("fc1bench::cutlass_gemm", mutates_args=())
def cutlass_gemm(x: torch.Tensor, sx: torch.Tensor, w: torch.Tensor, sw: torch.Tensor,
                 bias: Optional[torch.Tensor], tokens: int) -> torch.Tensor:
    m, k = x.shape
    return launch(_cfg("plain", m, w.shape[0], k), "plain", x, sx, w, sw, bias, tokens)


@cutlass_gemm.register_fake
def _(x, sx, w, sw, bias, tokens):
    return x.new_empty((x.shape[0], w.shape[0]), dtype=torch.float16)


@torch.library.custom_op("fc1bench::cutlass_gemm_res", mutates_args=())
def cutlass_gemm_res(x: torch.Tensor, sx: torch.Tensor, w: torch.Tensor, sw: torch.Tensor,
                     bias: Optional[torch.Tensor], r: torch.Tensor, gate: torch.Tensor, g_off: int,
                     tokens: int) -> torch.Tensor:
    m, k = x.shape
    return launch(_cfg("res", m, w.shape[0], k), "res", x, sx, w, sw, bias, tokens, r, gate, g_off)


@cutlass_gemm_res.register_fake
def _(x, sx, w, sw, bias, r, gate, g_off, tokens):
    return torch.empty_like(r)


def bench_variants(name, xq, sx, sxg, wq, sw, b, r, mods, tokens):
    """Callables for gemm_bench_v3 (tunes on first use)."""
    out = {}
    tune("plain", xq, sx, wq, sw, b, tokens, verbose=False)
    out["cutlass_plain"] = lambda: cutlass_gemm(xq, sx, wq, sw, b, tokens)
    if name in ("proj", "fc2"):
        tune("res", xq, sx, wq, sw, b, tokens, r, mods, 0, verbose=False)
        out["cutlass_res"] = lambda: cutlass_gemm_res(xq, sx, wq, sw, b, r, mods, 0, tokens)
    return out
