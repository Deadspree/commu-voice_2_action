"""Integration test: media normalization and AAC timing across remuxes."""
import sys
import shutil
import tempfile
import unittest
from pathlib import Path

from video_generation import media

FFMPEG = str(Path(sys.prefix) / "bin/ffmpeg")
FFPROBE = str(Path(sys.prefix) / "bin/ffprobe")


@unittest.skipUnless(shutil.which(FFMPEG) and shutil.which(FFPROBE), "FFmpeg/ffprobe not on PATH")
class MediaTests(unittest.TestCase):
    def test_trim_finalize_and_audio_timing(self):
        with tempfile.TemporaryDirectory() as tmp:
            for has_audio in (True, False):
                with self.subTest(has_audio=has_audio):
                    folder = Path(tmp) / str(has_audio)
                    folder.mkdir()
                    source = folder / "human.mp4"
                    command = [FFMPEG, "-v", "error", "-y", "-f", "lavfi", "-i",
                               "testsrc2=size=320x240:rate=30"]
                    if has_audio:
                        command += ["-f", "lavfi", "-i", "sine=frequency=440:sample_rate=48000"]
                    command += ["-t", "2", "-c:v", "libx264", "-pix_fmt", "yuv420p"]
                    if has_audio:
                        command += ["-c:a", "aac"]
                    media.run(command + [source])
                    prepared = folder / "prepared.mp4"
                    info = media.prepare(source, prepared, FFMPEG, FFPROBE, start=0.2, duration=1)
                    final = folder / "final.mp4"
                    media.finalize(prepared, folder / "source_audio.m4a", final, FFMPEG)
                    actual = media.validate(final, info, FFMPEG, FFPROBE)
                    self.assertAlmostEqual(actual["duration_seconds"], 1, places=1)
                    self.assertEqual(actual["has_audio"], has_audio)
                    if has_audio:
                        def decoded_audio(path):
                            return media.run([FFMPEG, "-v", "error", "-i", path,
                                              "-map", "0:a:0", "-f", "md5", "-"])
                        self.assertEqual(decoded_audio(prepared), decoded_audio(final))
