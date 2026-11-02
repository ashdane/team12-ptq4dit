"""Dependency-free reporting helpers."""
import math
import statistics


def summarize(samples):
    if not samples or any(not math.isfinite(x) or x < 0 for x in samples):
        raise ValueError("Expected nonempty, finite, nonnegative timings")
    ordered = sorted(samples)

    def percentile(q):
        index = (len(ordered) - 1) * q
        lo = int(index)
        hi = min(lo + 1, len(ordered) - 1)
        return ordered[lo] + (ordered[hi] - ordered[lo]) * (index - lo)

    return dict(median_ms=statistics.median(samples), p10_ms=percentile(.1),
                p90_ms=percentile(.9), stdev_ms=statistics.pstdev(samples),
                samples_ms=samples)
