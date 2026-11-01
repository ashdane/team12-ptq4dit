"""
Slot-Safe Block Reconstruction & Calibration (`quant/block_recon.py`)
---------------------------------------------------------------------
Implements block-wise quantization parameter calibration for PTQ4DiT with
per-block atomic checkpointing (`atomic_torch_save`) and SlotGuard monitoring.
If your CS3406 GPU slot expires mid-reconstruction, re-running with `--resume`
automatically skips all completed DiT blocks and finishes in seconds.
"""

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.slot_guard import SlotGuard, atomic_torch_save
from .quant_layer import QuantModule


def calibrate_and_reconstruct_blocks(
    qnn,
    cali_data: dict,
    use_ssc: bool = True,
    do_recon: bool = True,
    recon_iters: int = 40,
    lr: float = 1e-4,
    resume_ckpt_path: str = None,
    save_ckpt_path: str = "output/ckpt.pth",
    slot_guard: SlotGuard = None,
):
    """
    Runs PTQ4DiT calibration:
      1. Passes calibration samples across timesteps through the model to record
         per-timestep activation salience s(X^(t)).
      2. Applies Spearman's rho-guided Salience Calibration (SSC) & Channel-wise
         Salience Balancing (CSB) to every QuantModule.
      3. Optionally fine-tunes block quantization scales via block reconstruction,
         saving atomic checkpoints after every block.
    """
    device = next(qnn.parameters()).device
    xs = cali_data["xs"].to(device)
    ts = cali_data["ts"].to(device)
    ys = cali_data["ys"].to(device)

    start_block = 0
    if resume_ckpt_path and os.path.isfile(resume_ckpt_path):
        print(f"[block_recon] Resuming calibration state from {resume_ckpt_path} ...")
        state = torch.load(resume_ckpt_path, map_location=device, weights_only=False)
        qnn.load_state_dict(state["model_state"], strict=False)
        start_block = state.get("completed_blocks", 0)
        print(f"[block_recon] Resumed at block {start_block}/{len(qnn.model.blocks)}.")

    # Step 1 & 2: If starting fresh, record per-timestep activation saliences and compute CSB+SSC
    if start_block == 0:
        print("[block_recon] Step 1/2: Recording multi-timestep activation salience and computing CSB + SSC...")
        qnn.set_quant_state(weight_quant=False, act_quant=False)
        qnn.set_record_salience(True)
        with torch.no_grad():
            n_timesteps = xs.shape[0]
            for t_idx in range(n_timesteps):
                x_b = xs[t_idx]
                t_b = ts[t_idx]
                y_b = ys[t_idx]
                _ = qnn(x_b, t_b, y_b)
        qnn.set_record_salience(False)
        qnn.compute_all_csb_ssc(use_ssc=use_ssc)

        # Initialize static activation scales with one forward pass
        qnn.set_quant_state(weight_quant=True, act_quant=True)
        with torch.no_grad():
            _ = qnn(xs[0], ts[0], ys[0])

        atomic_torch_save(
            {"model_state": qnn.state_dict(), "completed_blocks": 0, "csb_ssc_done": True},
            save_ckpt_path,
        )

    if not do_recon or recon_iters <= 0:
        print(f"[block_recon] CSB + SSC calibration complete. Saved checkpoint -> {save_ckpt_path}")
        return qnn

    # Step 3: Lightweight block-wise scale reconstruction with per-block atomic checkpointing
    print(f"[block_recon] Step 3: Running block-wise scale calibration ({recon_iters} iters/block)...")
    # Prepare cached block inputs from calibration midpoint timestep
    mid_t = xs.shape[0] // 2
    with torch.no_grad():
        qnn.set_quant_state(weight_quant=False, act_quant=False)
        x_in = qnn.model.x_embedder(xs[mid_t]) + qnn.model.pos_embed.to(xs.dtype)
        c_in = qnn.model.t_embedder(ts[mid_t]) + qnn.model.y_embedder(ys[mid_t], False)

    def _save_progress(b_idx):
        atomic_torch_save(
            {"model_state": qnn.state_dict(), "completed_blocks": b_idx, "csb_ssc_done": True},
            save_ckpt_path,
        )

    if slot_guard is not None:
        slot_guard.register_emergency_save(lambda: _save_progress(start_block))

    for b_idx, block in enumerate(qnn.model.blocks):
        if b_idx < start_block:
            with torch.no_grad():
                for m in block.modules():
                    if isinstance(m, QuantModule):
                        m.set_quant_state(True, True)
                x_in = block(x_in, c_in)
            continue

        if slot_guard is not None and slot_guard.should_stop_safely():
            _save_progress(b_idx)
            print(f"[block_recon] Paused safely at block {b_idx}. Resume next slot with --resume!")
            return qnn

        # Compute FP target output for this block
        with torch.no_grad():
            for m in block.modules():
                if isinstance(m, QuantModule):
                    m.set_quant_state(False, False)
            fp_out = block(x_in, c_in).detach()

        # Enable quantization and fine-tune weight quantizer scales (delta) for this block
        deltas = []
        for m in block.modules():
            if isinstance(m, QuantModule):
                m.set_quant_state(True, True)
                m.weight_quantizer.delta = nn.Parameter(m.weight_quantizer.delta.detach().float())
                deltas.append(m.weight_quantizer.delta)

        if len(deltas) > 0:
            optimizer = torch.optim.Adam(deltas, lr=lr)
            for _ in range(recon_iters):
                optimizer.zero_grad()
                q_out = block(x_in, c_in)
                loss = F.mse_loss(q_out.float(), fp_out.float())
                loss.backward()
                optimizer.step()

            # Convert delta back to buffer
            for m in block.modules():
                if isinstance(m, QuantModule) and isinstance(m.weight_quantizer.delta, nn.Parameter):
                    val = m.weight_quantizer.delta.detach().to(m.weight.dtype)
                    del m.weight_quantizer._parameters["delta"]
                    m.weight_quantizer.register_buffer("delta", val)

        with torch.no_grad():
            x_in = block(x_in, c_in).detach()

        start_block = b_idx + 1
        _save_progress(start_block)
        if (b_idx + 1) % 7 == 0 or (b_idx + 1) == len(qnn.model.blocks):
            print(f"  [block_recon] Completed block {b_idx + 1}/{len(qnn.model.blocks)} (checkpoint saved)")

    print(f"[block_recon] All {len(qnn.model.blocks)} blocks calibrated and saved -> {save_ckpt_path}")
    return qnn
