import unittest
from fc1bench.report import summarize


class ReportTests(unittest.TestCase):
    def test_percentiles(self):
        result = summarize([4., 1., 3., 2.])
        self.assertEqual(result["median_ms"], 2.5)
        self.assertAlmostEqual(result["p10_ms"], 1.3)
        self.assertAlmostEqual(result["p90_ms"], 3.7)
        self.assertEqual(result["samples_ms"], [4., 1., 3., 2.])

    def test_single_sample(self):
        self.assertEqual(summarize([2.])["stdev_ms"], 0.)

    def test_invalid(self):
        for samples in ([], [-1.], [float("nan")], [float("inf")]):
            with self.assertRaises(ValueError):
                summarize(samples)
