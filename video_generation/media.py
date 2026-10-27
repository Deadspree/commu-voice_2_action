"""FFmpeg processing preserves the prepared clip timeline from zero."""
import json
import math
import shutil
import subprocess
from pathlib import Path


def run(command):
    result = subprocess.run([str(x) for x in command], capture_output=True, text=True)
    if result.returncode:
        raise RuntimeError(f"{Path(command[0]).name} failed:\n{result.stderr[-2500:]}")
    return result.stdout


def require_tools(ffmpeg, ffprobe):
    for tool in (ffmpeg, ffprobe):
        if not shutil.which(tool):
            raise RuntimeError(f"Missing {tool}. Install FFmpeg or pass --ffmpeg and --ffprobe paths.")


def probe(path, ffprobe):
    info = json.loads(run([ffprobe, "-v", "error", "-show_streams", "-show_format", "-of", "json", path]))
    videos = [s for s in info["streams"] if s["codec_type"] == "video" and not s.get("disposition", {}).get("attached_pic")]
    if not videos:
        raise ValueError(f"No video stream in {path}")
    video = videos[0]
    duration = float(video.get("duration") or info["format"].get("duration", 0))
    if not math.isfinite(duration) or duration <= 0:
        raise ValueError(f"Invalid duration in {path}")
    return {"duration_seconds": duration, "width": video["width"], "height": video["height"],
            "frame_rate": video["avg_frame_rate"], "time_base": video["time_base"],
            "has_audio": any(s["codec_type"] == "audio" for s in info["streams"])}


def prepare(source, output, ffmpeg, ffprobe, start=0, duration=None):
    source_info = probe(source, ffprobe)
    if start >= source_info["duration_seconds"]:
        raise ValueError("--start must be before the end of the input video")
    width, height = (1280, 720) if source_info["width"] >= source_info["height"] else (720, 1280)
    filters = (f"scale={width}:{height}:force_original_aspect_ratio=decrease:force_divisible_by=2,"
               f"pad={width}:{height}:(ow-iw)/2:(oh-ih)/2,setsar=1,setpts=PTS-STARTPTS")
    command = [ffmpeg, "-hide_banner", "-loglevel", "error", "-y", "-ss", str(start), "-i", source]
    if duration is not None:
        command += ["-t", str(duration)]
    command += ["-map", "0:v:0", "-map", "0:a:0?", "-vf", filters,
                "-af", "asetpts=PTS-STARTPTS,aresample=async=1:first_pts=0",
                "-c:v", "libx264", "-preset", "fast", "-crf", "18", "-pix_fmt", "yuv420p",
                "-r", "30", "-c:a", "aac", "-b:a", "192k", "-movflags", "+faststart", output]
    run(command)
    if Path(output).stat().st_size > 200_000_000:
        raise ValueError("Prepared video exceeds 200 MB. Select a shorter clip with --duration.")
    info = probe(output, ffprobe)
    audio = Path(output).with_name("source_audio.m4a")
    if info["has_audio"]:
        # M4A retains AAC encoder-delay/edit-list metadata; Matroska does not.
        run([ffmpeg, "-hide_banner", "-loglevel", "error", "-y", "-i", output,
             "-map", "0:a:0", "-vn", "-c:a", "copy", audio])
    return info


def validate(path, expected, ffmpeg, ffprobe):
    actual = probe(path, ffprobe)
    tolerance = max(0.15, 2 / 30)
    if abs(actual["duration_seconds"] - expected["duration_seconds"]) > tolerance:
        raise ValueError("Generated duration differs from the prepared clip; audio alignment needs review.")
    # Decode every frame so truncated/corrupt downloads don't pass on metadata alone.
    run([ffmpeg, "-hide_banner", "-loglevel", "error", "-xerror", "-i", path,
         "-map", "0:v:0", "-f", "null", "-"])
    return actual


def finalize(raw, audio, output, ffmpeg):
    if Path(audio).is_file():
        run([ffmpeg, "-hide_banner", "-loglevel", "error", "-y", "-i", raw, "-i", audio,
             "-map", "0:v:0", "-map", "1:a:0", "-c", "copy", "-movflags", "+faststart", output])
    else:
        run([ffmpeg, "-hide_banner", "-loglevel", "error", "-y", "-i", raw,
             "-map", "0:v:0", "-c", "copy", "-movflags", "+faststart", output])
