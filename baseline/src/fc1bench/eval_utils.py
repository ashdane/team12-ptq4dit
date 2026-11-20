"""Evaluation utilities: separate measurement, diagnostics, and reproducibility."""
from contextlib import contextmanager
import csv
import json
from pathlib import Path
import random
import subprocess
import time

import torch
from .report import summarize


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2, allow_nan=False))
    tmp.replace(path)


def write_csv(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(dict.fromkeys(k for row in rows for k in row))
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def git_record(repo):
    def call(*args):
        return subprocess.check_output(["git", "-C", str(repo), *args], text=True)
    # Ignore untracked files: importing the upstream repo creates __pycache__/
    # there, which would otherwise make the first generated mode differ from the rest.
    # Edits to tracked files are still captured by status and diff.
    return dict(commit=call("rev-parse", "HEAD").strip(),
                status=call("status", "--porcelain", "--untracked-files=no"),
                diff=call("diff", "HEAD"))


@contextmanager
def precision(dtype):
    with torch.autocast("cuda", dtype=dtype if dtype != torch.float32 else torch.bfloat16,
                        enabled=dtype != torch.float32):
        yield


@contextmanager
def matched_rng(seed):
    # Restore both CPU and CUDA RNG states after each run. Initial latent and
    # every ancestral sampler noise draw repeat across all model modes.
    with torch.random.fork_rng(devices=[torch.cuda.current_device()]):
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        yield


def timed(fn, warmup=5, repeats=20):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    gpu, wall = [], []
    for _ in range(repeats):
        start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        t = time.perf_counter()
        start.record()
        fn()
        end.record()
        end.synchronize()
        wall.append((time.perf_counter()-t)*1000)
        gpu.append(start.elapsed_time(end))
    return dict(cuda_event=summarize(gpu), synchronized_wall=summarize(wall))


def memory(fn):
    torch.cuda.synchronize()
    torch.cuda.empty_cache()
    base = torch.cuda.memory_allocated()
    reserved = torch.cuda.memory_reserved()
    torch.cuda.reset_peak_memory_stats()
    output = fn()
    torch.cuda.synchronize()
    result = dict(baseline_allocated_bytes=base, baseline_reserved_bytes=reserved,
                  peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                  peak_reserved_bytes=torch.cuda.max_memory_reserved(),
                  incremental_peak_allocated_bytes=torch.cuda.max_memory_allocated()-base)
    del output
    return result


def model_storage(model):
    seen = set()
    total = 0
    for t in list(model.parameters()) + list(model.buffers()):
        storage = t.untyped_storage()
        ident = (str(t.device), storage.data_ptr())
        if ident not in seen:
            seen.add(ident)
            total += storage.nbytes()
    return total


def layer_timings(model, names, call):
    """CUDA event attribution in a separate diagnostic pass; not model latency."""
    handles, pending, pairs = [], {}, []
    for name in names:
        def before(module, inputs, name=name):
            event = torch.cuda.Event(enable_timing=True)
            event.record()
            pending.setdefault(name, []).append(event)

        def after(module, inputs, output, name=name):
            event = torch.cuda.Event(enable_timing=True)
            event.record()
            pairs.append((name, pending[name].pop(), event, list(inputs[0].shape), list(output.shape)))

        layer = model.get_submodule(name)
        handles += [layer.register_forward_pre_hook(before), layer.register_forward_hook(after)]
    try:
        call()
        torch.cuda.synchronize()
    finally:
        for h in handles:
            h.remove()
    return [dict(name=n, cuda_event_ms=a.elapsed_time(b), input_shape=str(x), output_shape=str(y))
            for n, a, b, x, y in pairs]
