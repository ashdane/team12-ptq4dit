import base64
from contextlib import redirect_stderr
import copy
import hashlib
import io
import json
from pathlib import Path
import re
import tempfile
import unittest
import zipfile

import numpy as np
from fc1bench import image_quality
from fc1bench.quality_data import batch_ranges, deltas, npz_shape, pack_shards, paper_rows, sha256, validate_manifests, write_json


class QualityDataTests(unittest.TestCase):
    def test_streaming_npz_roundtrip_with_partial_final_batch(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp)
            expected=np.arange(5*4*4*3,dtype=np.uint8).reshape(5,4,4,3)
            for start,n in batch_ranges(5,2): np.save(root/f'{start:06d}.npy',expected[start:start+n])
            pack_shards(root,root/'samples.npz',5,2,4)
            self.assertEqual(npz_shape(root/'samples.npz'),(5,4,4,3))
            with np.load(root/'samples.npz') as z:
                np.testing.assert_array_equal(z['arr_0'],expected)

    def test_reject_corrupt_shard_and_preserve_existing_archive(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp)
            np.save(root/'000000.npy',np.zeros((2,4,4,3),dtype=np.float32))
            out=root/'samples.npz';out.write_bytes(b'previous')
            with self.assertRaises(ValueError):pack_shards(root,out,2,2,4)
            self.assertEqual(out.read_bytes(),b'previous')
            self.assertFalse((root/'samples.npz.tmp').exists())

    def test_reject_float_image_archive(self):
        with tempfile.TemporaryDirectory() as temp:
            path=Path(temp)/'images.npz'
            np.savez(path,arr_0=np.zeros((4,4,4,3),dtype=np.float32))
            with self.assertRaises(ValueError):npz_shape(path)

    def manifest(self):
        return dict(status='complete',checkpoint_sha256='abc',dit_commit='rev',source_sha256={},
                    image_size=256,num_samples=10000,batch_size=4,steps=50,cfg=1.5,seed=1,
                    vae='vae',vae_revision='rev',scope='all',attention='explicit')

    def test_mismatched_comparisons_fail_closed(self):
        base=self.manifest()
        for field in ['checkpoint_sha256','seed','steps','image_size','num_samples','batch_size','scope','vae_revision']:
            other=copy.deepcopy(base);other[field]='different'
            with self.subTest(field=field),self.assertRaises(ValueError):validate_manifests([base,other])
        bad=copy.deepcopy(base);bad['status']='running'
        with self.assertRaises(ValueError):validate_manifests([base,bad])
        self.assertEqual(validate_manifests([base,copy.deepcopy(base)]),base)

    def test_metric_deltas_and_published_values(self):
        rows=[dict(mode='fp32',FID=6.,sFID=20.,IS=240.,Precision=.8),
              dict(mode='int4',FID=10.,sFID=25.,IS=180.,Precision=.7)]
        result=deltas(rows)
        self.assertEqual(result[1]['FID_delta_vs_fp32'],4.)
        self.assertEqual(result[1]['IS_delta_vs_fp32'],-60.)
        self.assertAlmostEqual(result[1]['Precision_delta_vs_fp32'],-.1)
        with self.assertRaises(ValueError):deltas(rows[1:])
        self.assertEqual(paper_rows(256,50)[0]['FID'],6.02)
        self.assertEqual(paper_rows(512,50)[1]['num_samples'],5000)
        self.assertEqual(paper_rows(256,20),[])

    def test_image_quality_cli_gates_without_tensorflow(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            ref = root / 'ref.npz'
            np.savez(ref, arr_0=np.zeros((4, 256, 256, 3), dtype=np.uint8))
            for mode in ('fp32', 'int8'):
                mdir = root / mode
                mdir.mkdir()
                samples = mdir / 'samples.npz'
                np.savez(samples, arr_0=np.zeros((4, 256, 256, 3), dtype=np.uint8))
                m = self.manifest()
                m['num_samples'] = 4
                m['samples_sha256'] = sha256(samples)
                write_json(mdir / 'manifest.json', m)
            # Missing --allow-small-sample must fail via SystemExit
            with redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                image_quality.main(['--output', str(root), '--reference', str(ref),
                                    '--adm-repo', str(root), '--modes', 'fp32', 'int8'])
            # Missing fp32 baseline must fail via SystemExit
            with redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                image_quality.main(['--output', str(root), '--reference', str(ref),
                                    '--adm-repo', str(root), '--modes', 'int8', '--allow-small-sample'])
            # Corrupt sample checksum must raise ValueError before TensorFlow import
            bad_manifest = json.loads((root / 'int8' / 'manifest.json').read_text())
            bad_manifest['samples_sha256'] = 'deadbeef'
            write_json(root / 'int8' / 'manifest.json', bad_manifest)
            with self.assertRaises(ValueError):
                image_quality.main(['--output', str(root), '--reference', str(ref),
                                    '--adm-repo', str(root), '--modes', 'fp32', 'int8',
                                    '--allow-small-sample'])

    def test_colab_notebook_bundle_integrity(self):
        repo_root = Path(__file__).resolve().parents[1]
        nb_path = repo_root / 'Team12_Triton_Kernels_Colab.ipynb'
        if not nb_path.exists():
            self.skipTest('Colab notebook not present in extracted bundle')
        nb = json.loads(nb_path.read_text())
        code_sources = '\n'.join(
            ''.join(c['source']) if isinstance(c['source'], list) else c['source']
            for c in nb['cells'] if c.get('cell_type') == 'code'
        )
        payload_match = re.search(r"PAYLOAD\s*=\s*'([^']+)'", code_sources)
        sha_match = re.search(r"EXPECTED_SHA256\s*=\s*'([0-9a-f]{64})'", code_sources)
        self.assertIsNotNone(payload_match)
        self.assertIsNotNone(sha_match)
        archive_bytes = base64.b64decode(payload_match.group(1))
        self.assertEqual(hashlib.sha256(archive_bytes).hexdigest(), sha_match.group(1))
        with zipfile.ZipFile(io.BytesIO(archive_bytes)) as zf:
            names = set(zf.namelist())
            for expected_file in (
                'README.md',
                'requirements-colab.txt',
                'requirements-adm.txt',
                'scripts/setup_pretrained.py',
                'scripts/build_colab_notebook.py',
                'src/fc1bench/generate.py',
                'src/fc1bench/image_quality.py',
                'src/fc1bench/quality_data.py',
            ):
                self.assertIn(expected_file, names)
                self.assertEqual(zf.read(expected_file), (repo_root / expected_file).read_bytes())


if __name__=='__main__':unittest.main()
