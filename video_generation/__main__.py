"""Run with python -m video_generation --help."""
import argparse
import math
import sys
from pathlib import Path

from .pipeline import ROOT, execute


def positive(value):
    number = float(value)
    if not math.isfinite(number) or number <= 0:
        raise argparse.ArgumentTypeError("must be finite and positive")
    return number


def main():
    parser = argparse.ArgumentParser(description="Turn a human performance into a robot video using Lucy.")
    parser.add_argument("--video", type=Path, default=ROOT / "dataset/lucy/human_videos/me-test-3.mp4")
    parser.add_argument("--reference", type=Path, default=ROOT / "video_generation/robot_references/commu_front.png")
    parser.add_argument("--config", type=Path, default=ROOT / "video_generation/configs/commu.json")
    parser.add_argument("--prompt", type=Path, default=ROOT / "video_generation/prompts/commu.txt")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--start", type=float, default=0, help="Source clip start in seconds (new runs only)")
    parser.add_argument("--duration", type=positive, help="Process only this many seconds (new runs only)")
    parser.add_argument("--prepare-only", action="store_true", help="Prepare locally without calling the API")
    parser.add_argument("--resume", action="store_true", help="Use saved inputs/settings/job ID in output-dir")
    parser.add_argument("--job-id", help="Recover an uncertain submission using its dashboard job ID")
    parser.add_argument("--ffmpeg", default=str(Path(sys.prefix) / "bin/ffmpeg"),
                        help="Defaults to FFmpeg installed in the active Python environment")
    parser.add_argument("--ffprobe", default=str(Path(sys.prefix) / "bin/ffprobe"),
                        help="Defaults to ffprobe installed in the active Python environment")
    args = parser.parse_args()
    if not math.isfinite(args.start) or args.start < 0:
        parser.error("--start must be finite and nonnegative")
    try:
        execute(args)
    except (Exception, KeyboardInterrupt) as exc:
        # Do not print HTTP request objects or credentials in tracebacks.
        parser.exit(1, f"Pipeline stopped: {exc}\nResume with the same --output-dir and --resume.\n")


if __name__ == "__main__":
    main()
