"""Resumable Step 3 pipeline; independent of the pose estimation stack."""
import fcntl
import json
import math
import os
from pathlib import Path

from dotenv import load_dotenv

from .decart_client import DecartClient
from .manifest import fingerprint, save
from . import media

ROOT = Path(__file__).resolve().parents[1]


def execute(args):
    media.require_tools(args.ffmpeg, args.ffprobe)
    load_dotenv(ROOT / ".env", override=False)
    directory = args.output_dir.resolve()
    directory.mkdir(parents=True, exist_ok=True)
    # Kernel releases the lock after interruption; no stale lock cleanup needed.
    with (directory / ".run.lock").open("w") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise RuntimeError("Another process is already using this output directory.") from None
        return run_locked(args, directory)


def run_locked(args, directory):
    manifest_path = directory / "manifest.json"
    if args.resume:
        if not manifest_path.is_file():
            raise ValueError("No saved run to resume in this output directory.")
        state = json.loads(manifest_path.read_text())
        for label, info in state["inputs"].items():
            if fingerprint(info["path"]) != info["sha256"]:
                raise ValueError(f"The {label} input changed; use a new output directory.")
        config = state["config"]
        prompt = state["prompt"]
    else:
        if manifest_path.exists() or any(p.name != ".run.lock" for p in directory.iterdir()):
            raise ValueError("Output directory is not empty; use --resume or a new directory.")
        if args.job_id:
            raise ValueError("--job-id requires --resume.")
        source, reference = args.video.resolve(), args.reference.resolve()
        for path in (source, reference):
            if not path.is_file():
                raise FileNotFoundError(path)
        if reference.suffix.lower() not in {".png", ".jpg", ".jpeg", ".webp"}:
            raise ValueError("Reference image must be PNG, JPEG, or WebP.")
        config = json.loads(args.config.read_text())
        if config.get("model") not in {"lucy-2.6", "lucy-2.5", "lucy-2.1"}:
            raise ValueError("Choose a supported pinned Lucy editing model.")
        if config.get("resolution") != "720p":
            raise ValueError("Lucy batch editing currently requires 720p.")
        for key in ("poll_interval_seconds", "poll_timeout_seconds"):
            value = config[key]
            if not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
                raise ValueError(f"{key} must be finite and positive.")
        prompt = args.prompt.read_text().strip()
        state = {"schema_version": 1, "status": "initialized", "job_id": None,
                 "inputs": {label: {"path": str(path), "sha256": fingerprint(path)}
                            for label, path in (("video", source), ("reference", reference))},
                 "config": config, "prompt": prompt,
                 "timeline": {"source_start_seconds": args.start,
                              "requested_duration_seconds": args.duration,
                              "prepared_start_seconds": 0, "prepared_fps": 30,
                              "geometry": "fit and letterbox to 720p; no crop"}}
        save(manifest_path, state)
        save(directory / "config.json", config)
        (directory / "prompt.txt").write_text(prompt + "\n")

    prepared = directory / "prepared.mp4"
    raw = directory / "lucy_raw.mp4"
    final = directory / "robot_with_audio.mp4"
    if "prepared" not in state:
        print("Preparing video and preserving its audio...", flush=True)
        state["prepared"] = media.prepare(
            state["inputs"]["video"]["path"], prepared, args.ffmpeg, args.ffprobe,
            state["timeline"]["source_start_seconds"], state["timeline"]["requested_duration_seconds"])
        state["prepared_sha256"] = fingerprint(prepared)
        state["status"] = "prepared"
        save(manifest_path, state)
    elif not prepared.is_file() or fingerprint(prepared) != state["prepared_sha256"]:
        raise ValueError("Prepared video is missing or changed; use a new run directory.")

    if args.prepare_only:
        print(f"Prepared locally: {prepared}\nNo API request submitted.", flush=True)
        return
    if args.job_id:
        if state.get("job_id") and state["job_id"] != args.job_id:
            raise ValueError("This run already belongs to a different job.")
        state["job_id"] = args.job_id
        save(manifest_path, state)
    if not state.get("job_id") and state["status"] == "submission_started":
        raise RuntimeError("An earlier submission has an uncertain outcome. Check the dashboard, then "
                           "resume with --job-id ID. No automatic resubmission is allowed.")
    if state.get("status") == "completed" and final.is_file():
        print(f"Already completed: {final}", flush=True)
        return
    api_key = os.environ.get("DECART_API_KEY", "").strip()
    if not api_key:
        raise ValueError("Set DECART_API_KEY in the repository .env, then rerun with --resume.")
    client = DecartClient(api_key)
    try:
        if not state.get("job_id"):
            # Persist intent before networking: an interrupted POST must never be replayed blindly.
            state["status"] = "submission_started"
            save(manifest_path, state)
            print("Submitting one Lucy job...", flush=True)
            state["job_id"] = client.submit(prepared, state["inputs"]["reference"]["path"], prompt, config)
            state["status"] = "submitted"
            save(manifest_path, state)
            print(f"Saved job ID: {state['job_id']}", flush=True)

        def status_changed(status):
            if state.get("remote_status") != status:
                print(f"Lucy status: {status}", flush=True)
                state["remote_status"] = status
                state["status"] = f"remote_{status}"
                save(manifest_path, state)

        if not raw.is_file():
            client.wait(state["job_id"], config, status_changed)
            client.download(state["job_id"], raw)
        print("Validating downloaded video and attaching source audio...", flush=True)
        actual = media.validate(raw, state["prepared"], args.ffmpeg, args.ffprobe)
        media.finalize(raw, directory / "source_audio.m4a", final, args.ffmpeg)
        final_info = media.validate(final, state["prepared"], args.ffmpeg, args.ffprobe)
        if final_info["has_audio"] != state["prepared"]["has_audio"]:
            raise ValueError("Final audio stream does not match the prepared input.")
        save(directory / "validation.json", {"raw": actual, "final": final_info,
             "visual_review_required": True,
             "review": "Check robot identity, gestures, mouth timing, and audio synchronization."})
        state["status"] = "completed"
        state["output"] = str(final)
        save(manifest_path, state)
        print(f"Robot video ready for visual review: {final}", flush=True)
    finally:
        client.close()
