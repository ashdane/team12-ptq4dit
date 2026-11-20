"""Quantization routines and reference numerical implementations for PTQ4DiT."""
import torch
import torch.nn.functional as F


def quantize_rows(x, bits=8):
    if bits not in (4, 8) or x.ndim != 2:
        raise ValueError("Expected a matrix and bits=4 or 8")
    limit = (1 << (bits - 1)) - 1
    scale = (x.float().abs().amax(dim=1) * (1.0 / limit)).clamp_min(1e-8)
    q = (x.float() / scale[:, None]).round().clamp(-limit, limit).to(torch.int8)
    return q.contiguous(), scale.contiguous()


def quantize_activation_ref(x, bx=None, affine=False):
    """
    Reference implementation of activation quantization with optional CSB channel scaling.
    Uses IEEE-754 round-to-nearest-even (banker's rounding).
    Supports FP32, FP16, and BF16 activations [M, K].

    Returns:
        - If affine is False: (q, scale)
            q: [M, K] signed int8 in [-127, 127]
            scale: [M] float32
        - If affine is True: (q, scale, zp)
            q: [M, K] signed int8 in [-128, 127]
            scale: [M] float32
            zp: [M] int32 in [-128, 127]
            Reconstruction: x ~= (q - zp) * scale
    """
    if x.ndim != 2:
        raise ValueError("Expected a 2D activation matrix [M, K]")
    x_val = x.float()
    if bx is not None:
        if bx.ndim != 1 or bx.shape[0] != x.shape[1]:
            raise ValueError(f"Channel scaling vector bx shape {bx.shape} must match K={x.shape[1]}")
        x_val = x_val * bx.float()[None, :]

    if affine:
        # Affine asymmetric quantization to signed int8 [-128, 127]
        x_min = torch.minimum(x_val.amin(dim=1, keepdim=True), torch.zeros(1, device=x.device, dtype=torch.float32))
        x_max = torch.maximum(x_val.amax(dim=1, keepdim=True), torch.zeros(1, device=x.device, dtype=torch.float32))
        scale = ((x_max - x_min) / 255.0).clamp_min(1e-8)
        # Round-to-nearest-even zero point mapping x_min to -128, 0 to zp, x_max to 127
        zp = torch.clamp(torch.round(-128.0 - x_min / scale), -128, 127).to(torch.int32)
        q = torch.clamp(torch.round(x_val / scale) + zp, -128, 127).to(torch.int8)
        return q.contiguous(), scale.squeeze(1).contiguous(), zp.squeeze(1).contiguous()
    else:
        # Symmetric signed quantization [-127, 127] with round-to-nearest-even
        scale = (x_val.abs().amax(dim=1) * (1.0 / 127.0)).clamp_min(1e-8)
        q = torch.clamp(torch.round(x_val / scale[:, None]), -127, 127).to(torch.int8)
        return q.contiguous(), scale.contiguous()


def pack_int4(q):
    """Row-major signed nibbles: even K in low nibble; odd K in high nibble."""
    if q.ndim != 2 or q.dtype != torch.int8:
        raise ValueError("Expected a 2D int8 tensor")
    if bool(((q < -8) | (q > 7)).any()):
        raise ValueError("INT4 values must be in [-8, 7]")
    if q.shape[1] % 2:
        q = F.pad(q, (0, 1))
    return ((q[:, 0::2].to(torch.uint8) & 15) |
            ((q[:, 1::2].to(torch.uint8) & 15) << 4)).contiguous()


def unpack_int4(packed, k):
    if packed.ndim != 2 or packed.dtype != torch.uint8 or packed.shape[1] != (k + 1) // 2:
        raise ValueError("Packed matrix shape/dtype does not match K")
    values = torch.stack((packed & 15, packed >> 4), dim=-1).reshape(packed.shape[0], -1)
    return ((values.to(torch.int16) ^ 8) - 8).to(torch.int8)[:, :k].contiguous()


def pack_int4_kn(q):
    """Interleaved K-major byte packing [K, (N+1)//2] where each byte packs (2n, 2n+1)."""
    return pack_int4(q.T.contiguous())


def unpack_int4_kn(packed, n, k):
    """Unpack [K, (N+1)//2] into [N, K] signed int8."""
    return unpack_int4(packed, n).T[:n, :k].contiguous()


def int8_gemm_ref(xq, weight, sx, sw, bias=None, zx=None, dtype=torch.float32, raw=False):
    """
    Reference integer GEMM with fused scaling, optional zero-point correction, and bias.
    Supports signed INT8 activations (and UINT8 for mathematical verification).
    Weights are strictly symmetric around zero (zw = 0).

    Formula:
        raw_acc = xq @ weight.T
        corrected_acc = raw_acc - zx * sum(weight, dim=K)
        out = corrected_acc * sx[:, None] * sw[None, :] + (bias if bias is not None else 0)
    """
    if xq.dtype not in (torch.int8, torch.uint8):
        raise TypeError(f"Expected int8 or uint8 activations, got {xq.dtype}")
    if weight.dtype != torch.int8:
        raise TypeError(f"Expected int8 weights, got {weight.dtype}")
    m, k = xq.shape
    n = weight.shape[0]
    if k > 131071:
        raise ValueError("K is too large for safe INT32 accumulation without potential overflow")
    acc = torch.matmul(xq.to(torch.int64), weight.to(torch.int64).T).to(torch.int32)
    if raw:
        return acc

    if zx is not None:
        w_red = weight.float().sum(dim=1)  # [N]
        acc_corr = acc.float() - zx.float()[:, None] * w_red[None, :]
        out = acc_corr * sx.float()[:, None] * sw.float()[None, :]
    else:
        out = acc.float() * sx.float()[:, None] * sw.float()[None, :]

    if bias is not None:
        out = out + bias.float()[None, :]
    return out.to(dtype)


def int4_gemm_ref(xq, packed_w, sx, sw, bias=None, zx=None, layout="kn_packed", dtype=torch.float32, raw=False):
    """Reference packed INT4 GEMM matching Triton unpacking and computation."""
    m, k = xq.shape
    n = sw.numel()
    if layout == "kn_packed":
        w_int8 = unpack_int4_kn(packed_w, n, k)
    elif layout == "nk":
        w_int8 = unpack_int4(packed_w, k)
    else:
        raise ValueError(f"Unknown layout: {layout}")
    return int8_gemm_ref(xq, w_int8, sx, sw, bias=bias, zx=zx, dtype=dtype, raw=raw)


def metrics(reference, candidate):
    if reference.shape != candidate.shape:
        raise ValueError("Output shapes differ")
    a, b = reference.detach().double().flatten(), candidate.detach().double().flatten()
    if not bool(torch.isfinite(a).all() & torch.isfinite(b).all()):
        raise ValueError("Non-finite output")
    delta = b - a
    an, bn = torch.linalg.vector_norm(a), torch.linalg.vector_norm(b)
    cosine = (torch.dot(a, b) / (an * bn)).item() if an > 0 and bn > 0 else float(an == bn)
    return dict(mse=delta.square().mean().item(), max_abs_error=delta.abs().max().item(),
                relative_l2=(torch.linalg.vector_norm(delta) / an.clamp_min(1e-12)).item(),
                cosine_similarity=cosine)
