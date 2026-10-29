import csv
import json
from pathlib import Path
import shutil
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from video_generation import media
from video_generation.download_speech2gesture import (
    extract, load_candidates, run_download, seconds, youtube_url,
)


class SubsetTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        (self.root / "clips").mkdir()
        (self.root / "source_videos").mkdir()

    def test_timestamp_formats_and_invalid_values(self):
        self.assertEqual(seconds("0:01:02.5"), 62.5)
        self.assertEqual(seconds("1900-01-01 01:02:03.250"), 3723.25)
        self.assertEqual(seconds("12.3"), 12.3)
        for value in ("nan", "inf", "-1", "00:61:00", "missing"):
            with self.subTest(value=value), self.assertRaises(ValueError):
                seconds(value)

    def test_url_normalization_and_host_check(self):
        self.assertEqual(youtube_url("https://youtu.be/abcdefghijk?t=3"),
                         "https://www.youtube.com/watch?v=abcdefghijk")
        self.assertIsNone(youtube_url("https://youtube.com.evil.invalid/watch?v=abcdefghijk"))
        self.assertIsNone(youtube_url("https://www.youtube.com/watch?v=short"))

    def test_metadata_deduplicates_and_removes_overlaps(self):
        intervals = self.root / "intervals.csv"
        links = self.root / "links.csv"
        intervals.write_text("speaker,video_fn,start_time,end_time,interval_id,dataset\n"
                             "oliver,a.mp4,0:00:00,0:00:10,1,train\n"
                             "oliver,a.mp4,0:00:05,0:00:15,2,train\n"
                             "oliver,a.mp4,0:00:00,0:00:10,1,train\n"
                             "seth,b.mp4,0,10,3,test\n")
        links.write_text("speaker,video_fn,video_link\noliver,a.mp4,https://youtu.be/abcdefghijk\n"
                         "seth,b.mp4,https://youtu.be/zyxwvutsrqp\n")
        candidates, rejected = load_candidates(intervals, links, ["oliver"])
        self.assertEqual([(c["source_start"], c["source_end"]) for c in candidates], [(0, 10), (10, 15)])
        self.assertEqual(len(rejected), 1)

    def candidate(self, key, video="abcdefghijk", start=0, end=4):
        return dict(key=key, speaker="oliver", video_fn="a.mp4", source_start=start,
                    source_end=end, url="https://www.youtube.com/watch?v=" + video)

    def state(self, target=5):
        return dict(target_seconds=target, clips=[], failed_sources={}, failed_intervals={})

    def args(self):
        return SimpleNamespace(retry_failed=False, ffmpeg="ffmpeg", ffprobe="ffprobe",
                               max_height=720, cookies_file=None)

    def fake_extract(self, source, output, start, duration, *tools):
        output.write_bytes(b"encoded-video")
        return {"duration_seconds": duration, "has_audio": True}

    def test_budget_trims_last_clip_and_resume_skips_completed(self):
        state = self.state()
        candidates = [self.candidate("a"), self.candidate("b", start=4, end=8)]
        manifest = self.root / "manifest.json"
        with patch("video_generation.download_speech2gesture.download_source", return_value=self.root / "source_videos/a.mkv") as download, \
                patch("video_generation.download_speech2gesture.media.probe",
                      return_value={"duration_seconds": 20, "has_audio": True, "frame_rate": "30/1"}), \
                patch("video_generation.download_speech2gesture.extract", side_effect=self.fake_extract) as exporter:
            run_download(self.args(), candidates, state, manifest)
            self.assertEqual(state["completed_seconds"], 5)
            self.assertEqual(state["clips"][-1]["requested_duration_seconds"], 1)
            self.assertEqual(state["clips"][-1]["extracted_end_seconds"], 5)
            download.reset_mock()
            run_download(self.args(), candidates, state, manifest)
            download.assert_not_called()
            self.assertEqual(exporter.call_count, 2)

    def test_unavailable_source_replaced_and_attempted_once(self):
        candidates = [self.candidate("a"), self.candidate("b", start=4, end=8),
                      self.candidate("c", video="zyxwvutsrqp", end=6)]
        def download(url, *args):
            if "abcdefghijk" in url:
                raise RuntimeError("Video deleted")
            return self.root / "source_videos/ok.mkv"
        state = self.state()
        with patch("video_generation.download_speech2gesture.download_source", side_effect=download) as downloader, \
                patch("video_generation.download_speech2gesture.media.probe",
                      return_value={"duration_seconds": 20, "has_audio": True, "frame_rate": "30/1"}), \
                patch("video_generation.download_speech2gesture.extract", side_effect=self.fake_extract):
            run_download(self.args(), candidates, state, self.root / "manifest.json")
        self.assertEqual(downloader.call_count, 2)
        self.assertEqual(state["completed_seconds"], 5)

    def test_incomplete_dataset_is_not_marked_complete(self):
        state = self.state(target=10)
        with patch("video_generation.download_speech2gesture.download_source", side_effect=RuntimeError("Unavailable")):
            with self.assertRaisesRegex(RuntimeError, "Only 0.00/10.00"):
                run_download(self.args(), [self.candidate("a")], state, self.root / "manifest.json")
        self.assertEqual(state["status"], "incomplete")

    def test_bot_verification_stops_instead_of_skipping_entire_dataset(self):
        state = self.state()
        with patch("video_generation.download_speech2gesture.download_source",
                   side_effect=RuntimeError("Sign in to confirm you're not a bot")):
            with self.assertRaisesRegex(RuntimeError, "sign-in verification"):
                run_download(self.args(), [self.candidate("a")], state, self.root / "manifest.json")
        self.assertEqual(state["status"], "access_required")


FFMPEG = str(Path(sys.prefix) / "bin/ffmpeg")
FFPROBE = str(Path(sys.prefix) / "bin/ffprobe")


@unittest.skipUnless(shutil.which(FFMPEG) and shutil.which(FFPROBE), "Requires environment FFmpeg")
class ExtractionTests(unittest.TestCase):
    def test_real_mp4_has_audio_and_original_dimensions(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source, output = root / "input.mkv", root / "clip.mp4"
            media.run([FFMPEG, "-v", "error", "-y", "-f", "lavfi", "-i", "testsrc2=size=320x240:rate=30",
                       "-f", "lavfi", "-i", "sine=frequency=440:sample_rate=48000", "-t", "3",
                       "-c:v", "libx264", "-c:a", "aac", source])
            result = extract(source, output, .5, 1.5, FFMPEG, FFPROBE)
            self.assertTrue(result["has_audio"])
            self.assertEqual((result["width"], result["height"]), (320, 240))
            self.assertAlmostEqual(result["duration_seconds"], 1.5, places=2)
            self.assertFalse(output.with_name("clip.partial.mp4").exists())
