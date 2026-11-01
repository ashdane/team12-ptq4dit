"""
CS3406 Slot Guard & Atomic Checkpointer (`utils/slot_guard.py`)
---------------------------------------------------------------
Designed specifically for the CS3406 GPU cluster (`ssh teamNN@10.4.25.63`):
  1. All files are stored under `/data/cs3406/teamNN` (or a configurable base dir).
  2. When a GPU slot expires (or if SSH disconnects), the OS sends SIGHUP / SIGTERM
     and terminates all processes (including tmux/screen/nohup).
  3. This module registers signal handlers for SIGTERM, SIGHUP, and SIGINT, writes
     checkpoints atomically (`*.tmp` -> `os.replace`), and checks remaining slot
     time before starting expensive operations so your work is never corrupted or lost.
"""

import os
import sys
import time
import signal
import torch


class SlotGuard:
    def __init__(self, slot_minutes=120.0, warn_minutes=11.0, checkpoint_dir="checkpoints"):
        self.start_time = time.time()
        self.slot_seconds = float(slot_minutes) * 60.0
        self.warn_seconds = float(warn_minutes) * 60.0
        self.checkpoint_dir = checkpoint_dir
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        self.interrupted = False
        self.interrupt_signal = None
        self._emergency_callback = None

        # Register signal handlers for graceful checkpoint flush before slot kill
        for sig in (signal.SIGINT, signal.SIGTERM, signal.SIGHUP):
            try:
                signal.signal(sig, self._handle_signal)
            except Exception:
                pass

    def register_emergency_save(self, callback):
        """Register a zero-argument callback that saves current state on signal."""
        self._emergency_callback = callback

    def _handle_signal(self, signum, frame):
        sig_name = signal.Signals(signum).name if hasattr(signal, "Signals") else str(signum)
        print(f"\n[SlotGuard] Caught signal {sig_name}! Flushing emergency checkpoint before exit...", flush=True)
        self.interrupted = True
        self.interrupt_signal = sig_name
        if self._emergency_callback is not None:
            try:
                self._emergency_callback()
                print("[SlotGuard] Emergency checkpoint saved cleanly.", flush=True)
            except Exception as e:
                print(f"[SlotGuard] Error during emergency checkpoint save: {e}", flush=True)
        sys.exit(130)

    def elapsed_minutes(self):
        return (time.time() - self.start_time) / 60.0

    def remaining_minutes(self):
        rem = (self.slot_seconds - (time.time() - self.start_time)) / 60.0
        return max(0.0, rem)

    def should_stop_safely(self):
        """Returns True if interrupted or within the safety buffer before slot termination."""
        if self.interrupted:
            return True
        if self.slot_seconds > 0 and (self.slot_seconds - (time.time() - self.start_time)) <= self.warn_seconds:
            print(
                f"\n[SlotGuard] Slot safety threshold reached ({self.remaining_minutes():.1f} min remaining). "
                f"Pausing cleanly so checkpoint is safe before hard logout.",
                flush=True,
            )
            return True
        return False


def atomic_torch_save(obj, target_path):
    """
    Atomically saves a PyTorch object to `target_path` by writing to `target_path.tmp`
    and calling `os.replace()`. Prevents half-written corrupt `.pth`/`.pt` files if
    the CS3406 slot abruptly kills the process mid-write.
    """
    parent = os.path.dirname(os.path.abspath(target_path))
    os.makedirs(parent, exist_ok=True)
    tmp_path = f"{target_path}.tmp.{os.getpid()}"
    torch.save(obj, tmp_path)
    os.replace(tmp_path, target_path)
