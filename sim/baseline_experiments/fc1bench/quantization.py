"""Experimental symmetric quantization; deliberately separate from PTQ4DiT."""
import torch
import torch.nn.functional as F


def quantize_rows(x, bits=8):
    if bits not in (4, 8) or x.ndim != 2:
        raise ValueError("Expected a matrix and bits=4 or 8")
    limit = (1 << (bits - 1)) - 1
    scale = (x.float().abs().amax(dim=1) * (1.0 / limit)).clamp_min(1e-8)
    q = (x.float() / scale[:, None]).round().clamp(-limit, limit).to(torch.int8)
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
