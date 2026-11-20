#!/usr/bin/env python3
"""Summarise ncu raw-page CSVs written by scripts/ncu_profile.sh.

Usage: python3 scripts/ncu_summary.py <out_dir>   -> prints tables, writes <out_dir>/ncu_summary.csv

Per kernel: duration, compute/memory speed-of-light %, DRAM throughput % and bytes, tensor-pipe
utilisation, achieved occupancy, registers, L2 hit rate, top-3 warp stall reasons. For layer
targets also achieved TFLOPS/TOPS from the known FLOP count.
"""
import csv
import io
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

TIME_SCALE = {"nsecond": 1e-6, "usecond": 1e-3, "msecond": 1.0, "second": 1e3,      # -> ms
              "ns": 1e-6, "us": 1e-3, "ms": 1.0, "s": 1e3}
BYTE_SCALE = {"byte": 1, "Kbyte": 1e3, "Mbyte": 1e6, "Gbyte": 1e9,
              "B": 1, "KB": 1e3, "MB": 1e6, "GB": 1e9, "KiB": 1024, "MiB": 1024**2, "GiB": 1024**3}


def read(path):
    text = Path(path).read_text(errors="replace")
    start = text.find('"ID"')
    if start < 0:
        return []
    rows = list(csv.reader(io.StringIO(text[start:])))
    header, units, data = rows[0], rows[1], rows[2:]
    out = []
    for r in data:
        if len(r) != len(header):
            continue
        out.append({h: (v, u) for h, v, u in zip(header, r, units)})
    return out


def num(rec, key):
    if key not in rec:
        return None
    v, _ = rec[key]
    try:
        return float(v.replace(",", ""))
    except ValueError:
        return None


def ms(rec):
    v = num(rec, "gpu__time_duration.sum")
    if v is None:
        return None
    unit = rec["gpu__time_duration.sum"][1]
    if unit not in TIME_SCALE:
        raise ValueError(f"Unknown time unit {unit!r} for gpu__time_duration.sum")
    return v * TIME_SCALE[unit]


def nbytes(rec, key):
    v = num(rec, key)
    if v is None:
        return None
    unit = rec[key][1]
    if unit not in BYTE_SCALE:
        raise ValueError(f"Unknown byte unit {unit!r} for {key}")
    return v * BYTE_SCALE[unit]


def best(rec, pattern):
    vals = [num(rec, k) for k in rec if re.search(pattern, k)]
    vals = [v for v in vals if v is not None]
    return max(vals) if vals else None


def stalls(rec):
    pat = re.compile(r"smsp__average_warps_issue_stalled_(\w+?)_per_issue_active\.ratio$")
    items = [(m.group(1), num(rec, k)) for k in rec for m in [pat.search(k)] if m and num(rec, k)]
    return ", ".join(f"{n}:{v:.2f}" for n, v in sorted(items, key=lambda x: -x[1])[:3])


def summarise(rec):
    return dict(
        kernel=rec.get("Kernel Name", ("?",))[0][:70],
        ms=ms(rec),
        compute_sol_pct=num(rec, "sm__throughput.avg.pct_of_peak_sustained_elapsed"),
        memory_sol_pct=num(rec, "gpu__compute_memory_throughput.avg.pct_of_peak_sustained_elapsed"),
        dram_pct=num(rec, "gpu__dram_throughput.avg.pct_of_peak_sustained_elapsed"),
        dram_read_MB=(nbytes(rec, "dram__bytes_read.sum") or 0) / 1e6,
        dram_write_MB=(nbytes(rec, "dram__bytes_write.sum") or 0) / 1e6,
        # Tensor pipe busy cycles (covers HMMA fp16 and IMMA int8) while the SM is active.
        # (Ampere: sm__pipe_tensor_cycles_active; Ada: ..._v2)
        tensor_pipe_pct=next((v for v in (num(rec, "sm__pipe_tensor_cycles_active.avg.pct_of_peak_sustained_active"),
                                          num(rec, "sm__pipe_tensor_cycles_active_v2.avg.pct_of_peak_sustained_active"))
                              if v is not None), None),
        occupancy_pct=num(rec, "sm__warps_active.avg.pct_of_peak_sustained_active"),
        regs=num(rec, "launch__registers_per_thread"),
        l2_hit_pct=num(rec, "lts__t_sector_hit_rate.pct"),
        top_stalls=stalls(rec),
    )


def main():
    root = Path(sys.argv[1])
    rows = []
    for path in sorted(root.glob("*.csv")):
        if path.name in ("ncu_summary.csv", "gpu.csv"):
            continue
        tag = path.stem
        meta = {}
        log = root / f"{tag}.log"
        if log.exists():
            for line in log.read_text(errors="replace").splitlines():
                if line.startswith("{") and '"target"' in line:
                    meta = json.loads(line)
        recs = read(path)
        if tag.startswith("layer_"):
            total = sum(ms(r) or 0 for r in recs)
            for r in recs:
                s = summarise(r)
                s.update(tag=tag, kernels_in_target=len(recs), target_total_ms=total)
                if meta.get("flops") and total and "quant" not in tag:
                    s["target_TOPS"] = meta["flops"] / (total / 1e3) / 1e12
                rows.append(s)
        else:
            groups = defaultdict(lambda: dict(ms=0.0, n=0, dram_MB=0.0))
            for r in recs:
                s = summarise(r)
                key = re.sub(r"_\d+$", "", s["kernel"].split("<")[0].split("(")[0])[:50]
                g = groups[key]
                g["ms"] += s["ms"] or 0
                g["n"] += 1
                g["dram_MB"] += s["dram_read_MB"] + s["dram_write_MB"]
                g.setdefault("compute_sol", []).append(s["compute_sol_pct"] or 0)
                g.setdefault("memory_sol", []).append(s["memory_sol_pct"] or 0)
            for k, g in sorted(groups.items(), key=lambda kv: -kv[1]["ms"]):
                rows.append(dict(tag=tag, kernel=k, ms=g["ms"], kernels_in_target=g["n"],
                                 dram_read_MB=g["dram_MB"], compute_sol_pct=max(g["compute_sol"]),
                                 memory_sol_pct=max(g["memory_sol"])))
    if not rows:
        print("No ncu CSVs found")
        return
    keys = sorted({k for r in rows for k in r}, key=lambda k: (k != "tag", k != "kernel", k))
    with (root / "ncu_summary.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)
    fmt = lambda v, f="%.1f": (f % v) if isinstance(v, (int, float)) else "-"
    print(f"{'tag':34} {'kernel':38} {'ms':>8} {'TOPS':>6} {'SOL c%':>6} {'SOL m%':>6} {'DRAM%':>6} "
          f"{'tens%':>6} {'occ%':>5} {'regs':>4} {'L2hit':>5}  stalls")
    for r in rows:
        print(f"{r['tag'][:34]:34} {r['kernel'][:38]:38} {fmt(r.get('ms'), '%.4f'):>8} {fmt(r.get('target_TOPS')):>6} "
              f"{fmt(r.get('compute_sol_pct')):>6} {fmt(r.get('memory_sol_pct')):>6} {fmt(r.get('dram_pct')):>6} "
              f"{fmt(r.get('tensor_pipe_pct')):>6} {fmt(r.get('occupancy_pct')):>5} {fmt(r.get('regs'), '%.0f'):>4} "
              f"{fmt(r.get('l2_hit_pct')):>5}  {r.get('top_stalls', '')}")
    print(f"\nSaved {root/'ncu_summary.csv'}")


if __name__ == "__main__":
    main()
