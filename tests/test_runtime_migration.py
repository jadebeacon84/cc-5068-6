import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from vad_analyzer.audio import AudioBuffer, load_wav
from vad_analyzer.detector import FrameDetector
from vad_analyzer.kernel import VoiceIntervalKernel
from vad_analyzer.runtime_onnx import (
    PINNED_RUNTIME_VERSION,
    InputClock,
    legacy_center_to_new_start_sample,
    make_alignment,
)
from vad_analyzer.stream import StreamingVad

ROOT = Path(__file__).resolve().parent.parent
SHORT = ROOT / "data" / "labeled_short"


class RuntimeIdentityTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.detector = FrameDetector(runtime="onnx")
        cls.runtime = cls.detector.model

    def test_default_interval_path_uses_pinned_onnxruntime(self):
        self.assertEqual(self.detector.runtime, "onnx")
        self.assertIn("onnxruntime", self.detector.backend)
        self.assertEqual(self.runtime.runtime_version, PINNED_RUNTIME_VERSION)
        identity = self.runtime.identity
        self.assertEqual(identity["model_identity"], "local-logistic-v1")
        self.assertTrue(identity["model_sha256"])
        self.assertEqual(identity["weights_sha256"],
                         identity["source_weights_sha256_in_graph"])
        self.assertEqual(identity["output_timing"], "frame_start")

    def test_onnx_matches_reference_probabilities_and_first_frame(self):
        kernel = VoiceIntervalKernel(self.detector)
        for name in ("short_clean_A.wav", "wind_only.wav"):
            with self.subTest(name=name):
                audio = load_wav(SHORT / name)
                err = kernel.inference_error(audio)
                self.assertTrue(err["comparable"])
                self.assertLess(err["max_abs_error"], 1e-5)
                self.assertLess(err["mean_abs_error"], 1e-6)
                self.assertTrue(err["first_active_matches"])
                self.assertTrue(err["active_counts_match"])


class TimingAndAlignmentTests(unittest.TestCase):
    def test_old_center_maps_to_new_start_without_second_shift(self):
        hop, win = 160, 400
        for i in (0, 1, 66, 100):
            old_center = i * hop + win // 2
            new_start = legacy_center_to_new_start_sample(i, hop, win)
            self.assertEqual(new_start, i * hop)
            self.assertEqual(old_center - new_start, win // 2)

    def test_file_clock_is_zero_but_live_clock_is_retained(self):
        sr, hop = 16000, 160
        file_clock = InputClock.file_zero(sr, hop)
        live = InputClock.live_mid_stride(sr, 9_876_543, hop)
        self.assertEqual(file_clock.capture_origin_sample, 0)
        self.assertEqual(file_clock.phase_samples, 0)
        self.assertEqual(live.phase_samples, hop // 2)
        self.assertEqual(live.frame_start_local_sample(0), -hop // 2)
        self.assertEqual(live.frame_start_clock_sample(0),
                         9_876_543 - hop // 2)

    def test_alignment_compensation_has_single_owner(self):
        clock = InputClock.live_mid_stride(16000, 50_000, 160)
        padded = make_alignment(1001, clock, 512, "feature_pad")
        trimmed = make_alignment(1001, clock, 512, "output_trim")

        self.assertEqual((padded.feature_left_pad, padded.output_left_trim),
                         (80, 0))
        self.assertEqual((trimmed.feature_left_pad, trimmed.output_left_trim),
                         (0, 80))
        self.assertEqual(padded.feature_right_pad, 23)
        self.assertEqual(trimmed.feature_right_pad, 23)
        self.assertEqual(padded.output_to_source_sample(0), -80)
        self.assertEqual(trimmed.output_to_source_sample(0), 80)
        with self.assertRaises(ValueError):
            make_alignment(10, clock, 512, "feature_pad").__class__(
                clock=clock, feature_left_pad=80, output_left_trim=80)

    def test_live_events_keep_capture_clock_but_local_samples_drive_export(self):
        audio = load_wav(SHORT / "short_clean_A.wav")
        det = FrameDetector()
        from vad_analyzer.pipeline import config_for
        cfg = config_for(det)
        clock = InputClock.live_mid_stride(audio.sample_rate, 7_777_000, 160)
        vad = StreamingVad(audio.sample_rate, cfg, det, clock=clock)
        for a in range(0, len(audio.samples), 512):
            vad.feed(audio.samples[a:a + 512])
        vad.finish()
        starts = [e for e in vad.events if e.kind == "speech_started"]
        self.assertTrue(starts)
        for e in starts:
            self.assertEqual(e.clock_sample, 7_777_000 + e.sample)
        for iv in vad.confirmed:
            self.assertEqual(iv.clock_start_sample,
                             7_777_000 + iv.start_sample)
            self.assertEqual(iv.clock_end_sample,
                             7_777_000 + iv.end_sample)

    def test_editor_head_readback_has_no_double_offset(self):
        audio = load_wav(SHORT / "short_clean_A.wav")
        kernel = VoiceIntervalKernel()
        with tempfile.TemporaryDirectory() as td:
            head = kernel.export_and_read_back_head(audio, td)
            v = head.verification
            self.assertTrue(v.verified)
            self.assertEqual(v.requested_start_sample,
                             head.interval.start_sample)
            self.assertEqual(v.first_sample_expected,
                             int(audio.samples[head.interval.start_sample]))
            with __import__("wave").open(v.path, "rb") as wf:
                actual_head = int(np.frombuffer(
                    wf.readframes(1), dtype="<i2")[0])
            self.assertEqual(actual_head, v.first_sample_expected)


class BlockAndMaterialContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.kernel = VoiceIntervalKernel()

    def test_short_speech_and_noise_match_under_two_block_sizes(self):
        for name, expected_nonempty in (
            ("short_clean_A.wav", True),
            ("wind_only.wav", False),
        ):
            with self.subTest(name=name):
                audio = load_wav(SHORT / name)
                contract = self.kernel.block_contract(
                    audio, block_sizes=(256, 512))
                self.assertTrue(contract["all_match"], contract)
                ivs = contract["offline"]
                self.assertEqual(bool(ivs), expected_nonempty)

    def test_low_volume_and_non_whole_blocks_keep_semantic_interval(self):
        original = load_wav(SHORT / "short_clean_A.wav")
        x = original.samples.astype(np.float32) / 32768.0
        quiet = np.clip(np.round(x * 0.12 * 32767),
                        -32768, 32767).astype(np.int16)
        audio = AudioBuffer(quiet, original.sample_rate)
        err = self.kernel.inference_error(audio)
        self.assertLess(err["max_abs_error"], 1e-5)
        quiet_contract = self.kernel.block_contract(
            audio, block_sizes=(256, 512))
        self.assertTrue(quiet_contract["all_match"])
        self.assertTrue(quiet_contract["offline"])
        start, end, _, _ = quiet_contract["offline"][0]
        self.assertGreater((end - start) / original.sample_rate, 0.4)

        n = 80_000
        self.assertNotEqual(n % 256, 0)
        self.assertNotEqual(n % 512, 0)
        partial = AudioBuffer(original.samples[:n].copy(),
                              original.sample_rate)
        partial_contract = self.kernel.block_contract(
            partial, block_sizes=(256, 512))
        self.assertTrue(partial_contract["all_match"])
        self.assertTrue(partial_contract["offline"])
        self.assertLessEqual(partial_contract["offline"][0][1], n)


if __name__ == "__main__":
    unittest.main()
