import argparse
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from video_generation.batch import plan_batch, run_batch
from video_generation.manifest import fingerprint
from video_generation.pipeline import ROOT


class BatchTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.inputs = self.root / "clips"
        self.inputs.mkdir()
        for name in ("b.mp4", "a.mp4"):
            (self.inputs / name).write_bytes(name.encode())
        self.reference = self.root / "robot.png"
        self.reference.write_bytes(b"reference")
        self.args = argparse.Namespace(
            input_dir=self.inputs, output_dir=self.root / "runs", reference=self.reference,
            config=ROOT / "video_generation/configs/commu.json",
            prompt=ROOT / "video_generation/prompts/commu.txt", limit=None,
            dry_run=False, prepare_only=False, ffmpeg="ffmpeg", ffprobe="ffprobe")

    def saved_run(self, video_name, status="completed"):
        video = self.inputs / video_name
        directory = self.args.output_dir / video.stem
        directory.mkdir(parents=True, exist_ok=True)
        state = {"status": status, "job_id": "existing-job", "inputs": {
            "video": {"path": str(video), "sha256": fingerprint(video)},
            "reference": {"path": str(self.reference), "sha256": fingerprint(self.reference)}},
            "config": json.loads(self.args.config.read_text()), "prompt": self.args.prompt.read_text().strip()}
        (directory / "manifest.json").write_text(json.dumps(state))
        if status == "completed":
            (directory / "robot_with_audio.mp4").write_bytes(b"final")

    def test_dry_run_is_read_only_and_never_calls_pipeline(self):
        self.args.dry_run = True
        with patch("video_generation.batch.execute") as execute:
            run_batch(self.args)
        execute.assert_not_called()
        self.assertFalse(self.args.output_dir.exists())

    def test_new_existing_and_completed_runs_use_correct_resume_flags(self):
        self.saved_run("a.mp4", status="processing")
        with patch("video_generation.batch.execute") as execute:
            run_batch(self.args)
        calls = [call.args[0] for call in execute.call_args_list]
        self.assertEqual([call.video.name for call in calls], ["a.mp4", "b.mp4"])
        self.assertEqual([call.resume for call in calls], [True, False])
        self.saved_run("a.mp4")
        self.assertEqual(plan_batch(self.args)[0]["action"], "skip")

    def test_failure_stops_before_the_next_paid_job(self):
        with patch("video_generation.batch.execute", side_effect=TimeoutError("uncertain submission")) as execute:
            with self.assertRaisesRegex(RuntimeError, "Batch stopped at a.mp4"):
                run_batch(self.args)
        self.assertEqual(execute.call_count, 1)
        report = json.loads((self.args.output_dir / "batch.json").read_text())
        self.assertEqual(report["status"], "failed")
        self.assertEqual(report["clips"][0]["status"], "failed")

    def test_wrong_saved_video_rejected_before_any_jobs(self):
        self.saved_run("b.mp4")
        (self.inputs / "b.mp4").write_bytes(b"replacement")
        with patch("video_generation.batch.execute") as execute:
            with self.assertRaisesRegex(ValueError, "Saved input differs"):
                run_batch(self.args)
        execute.assert_not_called()

    def test_different_reference_requires_new_output_directory(self):
        self.saved_run("a.mp4")
        self.reference.write_bytes(b"another robot")
        with self.assertRaisesRegex(ValueError, "settings differ"):
            plan_batch(self.args)

    def test_limit_and_duplicate_stems(self):
        self.args.limit = 1
        self.assertEqual([p["video"].name for p in plan_batch(self.args)], ["a.mp4"])
        (self.inputs / "a.mov").write_bytes(b"duplicate name")
        with self.assertRaisesRegex(ValueError, "unique stems"):
            plan_batch(self.args)

    def test_prepare_only_passed_through(self):
        self.args.prepare_only = True
        with patch("video_generation.batch.execute") as execute:
            run_batch(self.args)
        self.assertTrue(all(call.args[0].prepare_only for call in execute.call_args_list))
