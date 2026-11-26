import sys
import pathlib
import unittest
from types import SimpleNamespace

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from vad_analyzer.labels import subtract_detected, union_spans
from vad_analyzer.evaluate import (coverage_counts, frame_confusion, iou,
                                  match_events)


def iv(a, b, sr=16000):
    return SimpleNamespace(start_sample=a, end_sample=b, sample_rate=sr)


class SetDifferenceTests(unittest.TestCase):
    def test_subtract_known_gap(self):
        ref = [(0, 16000)]
        det = [iv(0, 8000)]
        miss = subtract_detected(ref, det, min_ms=120)
        self.assertEqual(len(miss), 1)
        self.assertEqual((miss[0].start_sample, miss[0].end_sample),
                         (8000, 16000))

    def test_min_ms_filters_tiny_residual(self):
        ref = [(0, 16000)]
        det = [iv(0, 15900)]  # 100 samples ~6ms residual
        self.assertEqual(subtract_detected(ref, det, min_ms=120), [])

    def test_union_detected_before_difference(self):
        ref = [(0, 20000)]
        det = [iv(0, 6000), iv(5900, 12000), iv(12000, 20000)]
        self.assertEqual(subtract_detected(ref, det, min_ms=1), [])

    def test_union_spans(self):
        self.assertEqual(union_spans([(0, 10), (8, 20), (30, 40)]),
                         [(0, 20), (30, 40)])
        self.assertEqual(union_spans([(0, 10), (12, 20)], join_gap_samples=3),
                         [(0, 20)])


class MetricKnownValuesTests(unittest.TestCase):
    def test_frame_confusion_perfect(self):
        y = np.array([0, 0, 1, 1])
        m = frame_confusion(y, y)
        self.assertEqual((m["tp"], m["tn"], m["fp"], m["fn"]), (2, 2, 0, 0))
        self.assertEqual(m["accuracy"], 1.0)

    def test_iou(self):
        self.assertEqual(iou(0, 10, 0, 10), 1.0)
        self.assertAlmostEqual(iou(0, 10, 5, 15), 5 / 15)
        self.assertEqual(iou(0, 4, 5, 9), 0.0)

    def test_match_events_greedy(self):
        ref = [(0, 100), (200, 300)]
        dets = [iv(0, 100), iv(200, 300)]
        matches, ur, ud, _, _ = match_events(ref, dets, iou_thr=0.5)
        self.assertEqual(len(matches), 2)
        self.assertEqual(ur, [])
        self.assertEqual(ud, [])

    def test_coverage_allows_merge(self):
        # One detection covering two reference events fully.
        ref = [(0, 100), (120, 220)]
        dets = [iv(0, 220)]
        covered, supported = coverage_counts(ref, dets, 0.5)
        self.assertEqual(covered, 2)
        self.assertEqual(supported, 1)


if __name__ == "__main__":
    unittest.main()
