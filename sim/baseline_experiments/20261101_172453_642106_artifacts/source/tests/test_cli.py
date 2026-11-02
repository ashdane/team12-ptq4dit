import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest


@unittest.skipUnless(importlib.util.find_spec("torch"), "PyTorch is not installed")
class CliTests(unittest.TestCase):
    def test_cpu_smoke_artifacts(self):
        with tempfile.TemporaryDirectory() as out:
            subprocess.run([sys.executable, "-m", "fc1bench.benchmark", "--device", "cpu",
                            "--m", "5", "--k", "7", "--n", "9", "--modes", "fp32",
                            "--warmup", "1", "--repeats", "2", "--inner", "1", "--output", out], check=True)
            report = json.loads((Path(out) / "results.json").read_text())
            self.assertEqual(report["results"][0]["status"], "ok")
            self.assertEqual(report["results"][0]["correctness_vs_fp32"]["mse"], 0.)
            self.assertIsNone(report["results"][0]["core_timing"]["cuda_event"])
            self.assertTrue((Path(out) / "results.csv").exists())
