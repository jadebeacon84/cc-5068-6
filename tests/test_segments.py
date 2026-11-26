import sys
import pathlib
import unittest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from vad_analyzer.segments import VadConfig, merge_probabilities

SR = 16000
HOP = 160
WIN = 400


def merge(ps, cfg, n_total=None):
    starts = [i * HOP for i in range(len(ps))]
    if n_total is None:
        n_total = len(ps) * HOP + WIN
    return merge_probabilities(ps, starts, SR, HOP, WIN, n_total, cfg)


class MinSpeechTests(unittest.TestCase):
    def test_isolated_short_blip_is_dropped(self):
        cfg = VadConfig(enter_threshold=0.6, exit_threshold=0.3,
                        min_speech_ms=130, min_strong_ratio=0.0,
                        gap_tolerance_ms=200, edge_pad_ms=0)
        ps = [0.0] * 5 + [0.95, 0.9, 0.95] + [0.0] * 20
        self.assertEqual(merge(ps, cfg), [])

    def test_sustained_event_survives(self):
        cfg = VadConfig(enter_threshold=0.6, exit_threshold=0.3,
                        min_speech_ms=130, min_strong_ratio=0.35,
                        gap_tolerance_ms=200, edge_pad_ms=0)
        ps = [0.9] * 20 + [0.05] * 20
        ivs = merge(ps, cfg)
        self.assertEqual(len(ivs), 1)
        self.assertGreaterEqual(
            ivs[0].core_end_sample - ivs[0].core_start_sample,
            int(SR * 0.13))

    def test_low_strong_density_transient_cluster_dropped(self):
        # Two sparse click clusters: long end-to-end span but few strong frames.
        cfg = VadConfig(enter_threshold=0.6, exit_threshold=0.3,
                        min_speech_ms=130, min_strong_ratio=0.5,
                        gap_tolerance_ms=400, edge_pad_ms=0)
        ps = [0.95, 0.95, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05,
              0.95, 0.95] + [0.05] * 20
        self.assertEqual(merge(ps, cfg), [])


class GapMergeTests(unittest.TestCase):
    def cfg(self, gap_ms):
        return VadConfig(enter_threshold=0.6, exit_threshold=0.3,
                         min_speech_ms=100, min_strong_ratio=0.0,
                         gap_tolerance_ms=gap_ms, edge_pad_ms=0)

    def test_short_gap_bridges_one_interval(self):
        # 16 strong, 12 low (< gap_frames=20), 16 strong -> one interval.
        ps = [0.9] * 16 + [0.05] * 12 + [0.9] * 16
        ivs = merge(ps, self.cfg(200))
        self.assertEqual(len(ivs), 1)
        self.assertGreaterEqual(ivs[0].bridged_gaps, 1)

    def test_long_gap_splits_two_intervals(self):
        # 30 low frames (>= gap_frames=20) -> close, then a new episode.
        ps = [0.9] * 16 + [0.05] * 30 + [0.9] * 16 + [0.05] * 5
        ivs = merge(ps, self.cfg(200))
        self.assertEqual(len(ivs), 2)
        self.assertLess(ivs[0].end_sample, ivs[1].start_sample)


class PadExactTests(unittest.TestCase):
    def test_pad_is_exact_and_clamped(self):
        cfg = VadConfig(enter_threshold=0.6, exit_threshold=0.3,
                        min_speech_ms=100, min_strong_ratio=0.0,
                        gap_tolerance_ms=200, edge_pad_ms=50)
        # 41 zero frames then 20 strong frames (41..60).
        ps = [0.05] * 41 + [0.9] * 20 + [0.05] * 20
        n_total = 20000
        ivs = merge(ps, cfg, n_total=n_total)
        self.assertEqual(len(ivs), 1)
        iv = ivs[0]
        # c0 = 41*160 = 6560 ; pad = 800 -> 5760
        self.assertEqual(iv.core_start_sample, 41 * HOP)
        self.assertEqual(iv.start_sample, 41 * HOP - 800)
        # last strong frame 60: alive_end = 60*160+400 = 10000 ; +800
        self.assertEqual(iv.core_end_sample, 60 * HOP + WIN)
        self.assertEqual(iv.end_sample, 60 * HOP + WIN + 800)

    def test_pad_clamps_to_zero_and_length(self):
        cfg = VadConfig(enter_threshold=0.6, exit_threshold=0.3,
                        min_speech_ms=100, min_strong_ratio=0.0,
                        gap_tolerance_ms=200, edge_pad_ms=120)
        ps = [0.9] * 18 + [0.05] * 10
        n_total = 2600
        ivs = merge(ps, cfg, n_total=n_total)
        self.assertEqual(len(ivs), 1)
        self.assertEqual(ivs[0].start_sample, 0)
        self.assertLessEqual(ivs[0].end_sample, n_total)


if __name__ == "__main__":
    unittest.main()
