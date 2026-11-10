"""Sequential, resumable Lucy generation for a folder of performance clips."""
import argparse
from collections import Counter
import fcntl
import json
from pathlib import Path
import sys

from .manifest import fingerprint, save
from .pipeline import ROOT, execute

VIDEO_EXTENSIONS = {".mp4", ".mov", ".mkv", ".webm", ".m4v"}


def plan_batch(args):
    """Check every selected run before submitting any new paid jobs."""
    source_dir, output_dir = args.input_dir.resolve(), args.output_dir.resolve()
    if not source_dir.is_dir():
        raise ValueError(f"Input directory does not exist: {source_dir}")
    if source_dir == output_dir or source_dir in output_dir.parents or output_dir in source_dir.parents:
        raise ValueError("Input and output directories must be separate, without nesting")
    videos = sorted((p for p in source_dir.iterdir() if p.is_file() and p.suffix.lower() in VIDEO_EXTENSIONS),
                    key=lambda p: p.name)
    if not videos:
        raise ValueError(f"No supported videos found in {source_dir}")
    names = [p.stem.casefold() for p in videos]
    if len(names) != len(set(names)):
        raise ValueError("Video filenames must have unique stems, including across extensions")
    if args.limit is not None:
        if args.limit < 1:
            raise ValueError("--limit must be at least 1")
        videos = videos[:args.limit]
    for path in (args.reference, args.config, args.prompt):
        if not path.is_file():
            raise FileNotFoundError(path)
    config = json.loads(args.config.read_text())
    prompt = args.prompt.read_text().strip()
    reference_hash = fingerprint(args.reference)
    plan = []
    for video in videos:
        directory = output_dir / video.stem
        manifest = directory / "manifest.json"
        action = "new"
        if manifest.is_file():
            state = json.loads(manifest.read_text())
            saved_video = state["inputs"]["video"]
            if Path(saved_video["path"]).resolve() != video.resolve() or fingerprint(video) != saved_video["sha256"]:
                raise ValueError(f"Saved input differs for {video.name}; choose a new --output-dir")
            if (state["config"] != config or state["prompt"] != prompt
                    or state["inputs"]["reference"]["sha256"] != reference_hash):
                raise ValueError(f"Saved generation settings differ for {video.name}; "
                                 "restore them or choose a new --output-dir")
            action = "skip" if state.get("status") == "completed" and (directory / "robot_with_audio.mp4").is_file() else "resume"
        elif directory.exists() and any(p.name != ".run.lock" for p in directory.iterdir()):
            raise ValueError(f"Run directory has files but no manifest: {directory}")
        plan.append({"video": video, "directory": directory, "action": action})
    return plan


def run_batch(args):
    plan = plan_batch(args)
    counts = Counter(item["action"] for item in plan)
    print(f"Batch: {len(plan)} clips — {counts['new']} new, {counts['resume']} to resume, "
          f"{counts['skip']} already completed.", flush=True)
    if args.dry_run:
        for item in plan:
            print(f"{item['action']:6} {item['video'].name}")
        print("Dry run only. No preparation, uploads, or paid API calls.", flush=True)
        return
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    with (output / ".batch.lock").open("w") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise RuntimeError("Another batch is already using this output directory") from None
        report_path = output / "batch.json"
        report = {"status": "running", "input_dir": str(args.input_dir.resolve()),
                  "prepare_only": args.prepare_only, "selected_clips": len(plan), "clips": []}
        save(report_path, report)
        for index, item in enumerate(plan, 1):
            print(f"[{index}/{len(plan)}] {item['video'].name} ({item['action']})", flush=True)
            entry = {"video": str(item["video"]), "output_dir": str(item["directory"])}
            try:
                # Reuse all single-clip checks and the saved job ID. Even completed
                # runs are validated locally before the pipeline skips them.
                execute(argparse.Namespace(
                    video=item["video"], output_dir=item["directory"], reference=args.reference,
                    config=args.config, prompt=args.prompt, start=0, duration=None,
                    prepare_only=args.prepare_only, resume=item["action"] != "new", job_id=None,
                    ffmpeg=args.ffmpeg, ffprobe=args.ffprobe,
                ))
                entry["status"] = "prepared" if args.prepare_only else "completed"
            except (Exception, KeyboardInterrupt) as error:
                entry.update(status="interrupted" if isinstance(error, KeyboardInterrupt) else "failed",
                             error=str(error))
                report["clips"].append(entry)
                report["status"] = entry["status"]
                save(report_path, report)
                raise RuntimeError(f"Batch stopped at {item['video'].name}: {error}. "
                                   "Rerun the same batch command to continue. For an uncertain submission, "
                                   "recover its job ID with the single-video CLI first.") from error
            report["clips"].append(entry)
            save(report_path, report)
        report["status"] = "prepared" if args.prepare_only else "completed"
        save(report_path, report)
        print(f"Batch {report['status']}. Results: {output}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=ROOT / "dataset/lucy/speech2gesture/clips")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "dataset/lucy/generation_runs/speech2gesture")
    parser.add_argument("--reference", type=Path, default=ROOT / "video_generation/robot_references/commu_front.png")
    parser.add_argument("--config", type=Path, default=ROOT / "video_generation/configs/commu.json")
    parser.add_argument("--prompt", type=Path, default=ROOT / "video_generation/prompts/commu.txt")
    parser.add_argument("--limit", type=int, help="Process only the first N filenames in sorted order")
    parser.add_argument("--dry-run", action="store_true", help="Show the plan without changing files or calling Lucy")
    parser.add_argument("--prepare-only", action="store_true", help="Prepare locally without calling Lucy")
    parser.add_argument("--ffmpeg", default=str(Path(sys.prefix) / "bin/ffmpeg"))
    parser.add_argument("--ffprobe", default=str(Path(sys.prefix) / "bin/ffprobe"))
    args = parser.parse_args()
    try:
        run_batch(args)
    except (Exception, KeyboardInterrupt) as error:
        parser.exit(1, f"{error}\n")


if __name__ == "__main__":
    main()
