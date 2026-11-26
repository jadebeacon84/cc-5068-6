import dataclasses
import sys
import pathlib
import unittest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from vad_analyzer.cuts import LockedCut, evaluate_locked_cuts
from vad_analyzer.segments import Interval


def interval():
    return Interval(
        start_sample=900, end_sample=4100, sample_rate=16000,
        core_start_sample=1000, core_end_sample=4000,
        confidence=0.8, index=1)


class LockedCutTests(unittest.TestCase):
    def test_cut_is_frozen(self):
        cut = LockedCut("C1", 2500, 16000)
        with self.assertRaises(dataclasses.FrozenInstanceError):
            cut.sample = 999  # type: ignore[misc]

    def test_core_hit_is_error_and_cut_unchanged(self):
        iv = interval()
        cut = LockedCut("C1", 2500, 16000)
        conflicts = evaluate_locked_cuts([cut], [iv])
        self.assertEqual(len(conflicts), 1)
        self.assertEqual(conflicts[0].severity, "error")
        self.assertEqual(conflicts[0].kind, "cut_through_voice_activity")
        self.assertEqual(cut.sample, 2500)

    def test_pad_hit_is_warning(self):
        iv = interval()
        cut = LockedCut("C2", 950, 16000)  # between start 900 and core 1000
        conflicts = evaluate_locked_cuts([cut], [iv])
        self.assertEqual(conflicts[0].severity, "warning")
        self.assertEqual(conflicts[0].kind, "cut_inside_safety_pad")

    def test_clean_gap_produces_no_conflict(self):
        iv = interval()
        cut = LockedCut("C3", 10000, 16000)
        self.assertEqual(evaluate_locked_cuts([cut], [iv]), [])


if __name__ == "__main__":
    unittest.main()
