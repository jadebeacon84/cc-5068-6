import sys
import tempfile
import pathlib
import unittest

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from vad_analyzer.audio import AudioBuffer
from vad_analyzer.exporter import export_range, verify_export


class ExportVerificationTests(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(0)
        self.samples = rng.integers(-30000, 30000, size=48000).astype(np.int16)
        self.audio = AudioBuffer(samples=self.samples, sample_rate=16000)

    def test_export_matches_source_sample_for_sample(self):
        with tempfile.TemporaryDirectory() as d:
            out = pathlib.Path(d) / "clip.wav"
            s0, s1 = 1234, 9876
            path, ver = export_range(self.audio, s0, s1, out)
            self.assertTrue(path.exists())
            self.assertIsNotNone(ver)
            self.assertTrue(ver.verified)
            self.assertTrue(ver.length_match)
            self.assertTrue(ver.all_samples_match)
            self.assertTrue(ver.edges_match)
            self.assertEqual(ver.expected_sample_count, s1 - s0)
            self.assertEqual(ver.actual_sample_count, s1 - s0)
            self.assertEqual(ver.first_sample_expected, int(self.samples[s0]))
            self.assertEqual(ver.first_sample_actual, int(self.samples[s0]))
            self.assertEqual(ver.last_sample_expected, int(self.samples[s1 - 1]))
            self.assertEqual(ver.last_sample_actual, int(self.samples[s1 - 1]))

            # Independently re-verify from the reloaded WAV path.
            ver2 = verify_export(path, self.audio, s0, s1)
            self.assertTrue(ver2.all_samples_match)
            self.assertTrue(ver2.verified)

    def test_first_and_last_samples_are_actually_kept(self):
        with tempfile.TemporaryDirectory() as d:
            out = pathlib.Path(d) / "edge.wav"
            s0, s1 = 100, 300
            _, ver = export_range(self.audio, s0, s1, out)
            self.assertEqual(ver.first_sample_actual, ver.first_sample_expected)
            self.assertEqual(ver.last_sample_actual, ver.last_sample_expected)
            self.assertNotEqual(ver.sha256_source_slice, "")


if __name__ == "__main__":
    unittest.main()
