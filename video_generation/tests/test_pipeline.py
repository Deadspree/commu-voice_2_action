"""Regression tests for paid submission safety and media validation."""
import json
import tempfile
import unittest
from argparse import Namespace
from pathlib import Path
from unittest.mock import Mock, patch

from video_generation import media
from video_generation.pipeline import ROOT, run_locked


class PipelineTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.source = self.root / "human.mp4"
        self.reference = self.root / "robot.png"
        self.source.write_bytes(b"source")
        self.reference.write_bytes(b"reference")
        self.run = self.root / "run"
        self.run.mkdir()
        self.args = Namespace(resume=False, video=self.source, reference=self.reference,
                              config=ROOT / "video_generation/configs/commu.json",
                              prompt=ROOT / "video_generation/prompts/commu.txt",
                              start=0, duration=5, ffmpeg="ffmpeg", ffprobe="ffprobe",
                              prepare_only=True, job_id=None)

    def prepare(self, source, output, *args):
        output.write_bytes(b"prepared")
        return {"duration_seconds": 5, "has_audio": False}

    def initialize(self):
        with patch("video_generation.pipeline.media.prepare", side_effect=self.prepare):
            run_locked(self.args, self.run)
        self.args.resume = True
        self.args.prepare_only = False

    def test_prepare_only_never_constructs_api_client(self):
        with patch("video_generation.pipeline.DecartClient") as client:
            self.initialize()
            client.assert_not_called()

    def test_uncertain_submission_is_not_repeated(self):
        self.initialize()
        client = Mock()
        client.submit.side_effect = TimeoutError("uncertain POST")
        with patch.dict("os.environ", {"DECART_API_KEY": "test-only"}), \
                patch("video_generation.pipeline.DecartClient", return_value=client):
            with self.assertRaises(TimeoutError):
                run_locked(self.args, self.run)
            with self.assertRaisesRegex(RuntimeError, "uncertain outcome"):
                run_locked(self.args, self.run)
        self.assertEqual(client.submit.call_count, 1)

    def test_saved_job_resumes_without_submission(self):
        self.initialize()
        manifest = self.run / "manifest.json"
        state = json.loads(manifest.read_text())
        state.update(job_id="existing-job", status="processing")
        manifest.write_text(json.dumps(state))
        client = Mock()
        client.wait.side_effect = TimeoutError("still processing")
        with patch.dict("os.environ", {"DECART_API_KEY": "test-only"}), \
                patch("video_generation.pipeline.DecartClient", return_value=client):
            with self.assertRaises(TimeoutError):
                run_locked(self.args, self.run)
        client.submit.assert_not_called()
        self.assertEqual(client.wait.call_args.args[0], "existing-job")

    def test_changed_source_cannot_resume_old_job(self):
        self.initialize()
        self.source.write_bytes(b"changed")
        with self.assertRaisesRegex(ValueError, "input changed"):
            run_locked(self.args, self.run)

    def test_remote_completion_is_not_local_completion(self):
        self.initialize()
        client = Mock()
        client.submit.return_value = "new-job"
        client.wait.side_effect = lambda job, config, callback: callback("completed")
        client.download.side_effect = IOError("download interrupted")
        with patch.dict("os.environ", {"DECART_API_KEY": "test-only"}), \
                patch("video_generation.pipeline.DecartClient", return_value=client):
            with self.assertRaises(IOError):
                run_locked(self.args, self.run)
        state = json.loads((self.run / "manifest.json").read_text())
        self.assertEqual(state["status"], "remote_completed")

    def test_duration_mismatch_blocks_audio_attachment(self):
        with patch("video_generation.media.probe", return_value={"duration_seconds": 4}):
            with self.assertRaisesRegex(ValueError, "duration differs"):
                media.validate("robot.mp4", {"duration_seconds": 5}, "ffmpeg", "ffprobe")


if __name__ == "__main__":
    unittest.main()
