import sys
import pathlib
import unittest

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from vad_analyzer.audio import AudioBuffer
from vad_analyzer.detector import FrameDetector
from vad_analyzer.pipeline import analyze_audio, config_for
from vad_analyzer.scenarios import (build_dialogue, build_fork_pair,
                                   build_interference)
from vad_analyzer.segments import merge_decision
from vad_analyzer.stream import run_streaming

DET = FrameDetector()
CFG = config_for(DET)


def buffer(x):
    return AudioBuffer(samples=np.asarray(x, dtype=np.int16), sample_rate=16000)


class ForkScenarioTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.fork = build_fork_pair()

    def test_prefixes_are_sample_identical(self):
        f = self.fork
        div = f.divergence_sample
        self.assertTrue(np.array_equal(f.audio_a[:div], f.audio_b[:div]))

    def test_branch_a_retracts_and_merges(self):
        st = run_streaming(buffer(self.fork.audio_a), CFG, DET)
        ks = [e.kind for e in st.events]
        self.assertEqual(ks.count("provisional_end"), 1)
        self.assertEqual(ks.count("end_retracted"), 1)
        self.assertEqual(len(st.confirmed), 1)
        self.assertGreaterEqual(st.confirmed[0].end_sample,
                                len(self.fork.audio_a) - 2048)

    def test_branch_b_never_retracts_and_ends_before_fork(self):
        f = self.fork
        st = run_streaming(buffer(f.audio_b), CFG, DET)
        ks = [e.kind for e in st.events]
        self.assertEqual(ks.count("end_retracted"), 0)
        self.assertGreaterEqual(ks.count("segment_confirmed"), 1)
        self.assertLessEqual(st.confirmed[0].end_sample,
                             f.divergence_sample + int(0.12 * 16000))


class DialoguePipelineTests(unittest.TestCase):
    def test_dialogue_produces_real_intervals(self):
        i16, ann = build_dialogue()
        res = analyze_audio(buffer(i16), CFG, DET, annotation=ann)
        self.assertGreaterEqual(len(res.intervals), 2)
        # Missed regions are always a list computed by set difference.
        self.assertIsInstance(res.missed, list)
        total = sum(iv.end_sample - iv.start_sample for iv in res.intervals)
        self.assertGreater(total, 16000)  # at least ~1s of activity


class InterferenceRejectionTests(unittest.TestCase):
    def test_loud_interferences_do_not_become_speech_intervals(self):
        for kind in ("wind", "keyboard", "music"):
            x, _ = build_interference(kind, 4.0, seed=5)
            d = DET.detect(buffer(x))
            ivs = merge_decision(d, CFG)
            self.assertEqual(
                ivs, [],
                msg=f"{kind} should yield no final intervals, got "
                    f"{[(round(v.start, 2), round(v.end, 2)) for v in ivs]}")


if __name__ == "__main__":
    unittest.main()
