import sys
import pathlib
import unittest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from vad_analyzer.segments import VadConfig
from vad_analyzer.stream import StreamingVad

SR = 16000


def make_vad():
    cfg = VadConfig(enter_threshold=0.6, exit_threshold=0.3,
                    min_speech_ms=60, min_strong_ratio=0.0,
                    gap_tolerance_ms=100, provisional_ms=40,
                    exit_tail_ms=50, edge_pad_ms=0)
    return StreamingVad(SR, cfg)


def drive(vad, probs):
    events = []
    for i, p in enumerate(probs):
        vad.probs.append(float(p))
        vad.start_samples.append(i * vad.hop)
        vad._processed_samples = (len(probs) + 8) * vad.hop + vad.win
        events.extend(vad._handle_prob(i, float(p)))
    return events


def kinds(events):
    return [e.kind for e in events]


class RetractionTests(unittest.TestCase):
    def test_late_evidence_retracts_provisional_single_segment(self):
        vad = make_vad()
        seq = [0.9] * 10 + [0.05] * 6 + [0.9] * 15
        events = drive(vad, seq)
        events += vad.finish()
        ks = kinds(events)

        self.assertEqual(ks.count("provisional_end"), 1)
        self.assertEqual(ks.count("end_retracted"), 1)
        # No archive confirmation may precede the retraction.
        first_confirm = next(i for i, k in enumerate(ks)
                             if k == "segment_confirmed")
        retract_at = ks.index("end_retracted")
        self.assertGreater(first_confirm, retract_at)
        self.assertEqual(ks.count("segment_confirmed"), 1)
        self.assertEqual(len(vad.confirmed), 1)

    def test_after_close_a_new_episode_is_not_a_retraction(self):
        vad = make_vad()
        # 6 strong, 10 low -> gap close; then more strong evidence opens a new
        # episode. The stale provisional flag must not produce a fake retract.
        seq = [0.9] * 6 + [0.05] * 11 + [0.9] * 9
        events = drive(vad, seq)
        ks_before_finish = kinds(events)
        events += vad.finish()
        ks = kinds(events)

        self.assertIn("segment_confirmed", ks_before_finish)
        self.assertEqual(ks.count("end_retracted"), 0)
        self.assertGreaterEqual(ks.count("speech_started"), 2)
        self.assertEqual(len(vad.confirmed), 2)


if __name__ == "__main__":
    unittest.main()
