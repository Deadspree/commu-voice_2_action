"""Download a duration-limited set of annotated intervals, without spatial edits."""
import argparse
import csv
import fcntl
import hashlib
import json
import math
from pathlib import Path
import re
import shutil
import sys
from urllib.parse import parse_qs, urlparse

from . import media
from .manifest import fingerprint, save

ROOT = Path(__file__).resolve().parents[1]
DRIVE_FOLDER = "https://drive.google.com/drive/folders/1qvvnfGwas8DUBrwD4DoBnvj8anjSLldZ"
SHOW_BASE = "https://raw.githubusercontent.com/yhw-yhw/SHOW/main/"


def seconds(value):
    """Dataset HH:MM:SS[.fraction], pandas date/time, or numeric seconds."""
    text = str(value).strip()
    match = re.search(r"(?:^|[ T])(\d+):(\d{2}):(\d{2}(?:\.\d+)?)$", text)
    if match:
        hours, minutes, secs = map(float, match.groups())
        if minutes >= 60 or secs >= 60:
            raise ValueError(f"Invalid timestamp: {text}")
        result = hours * 3600 + minutes * 60 + secs
    else:
        result = float(text)
    if not math.isfinite(result) or result < 0:
        raise ValueError(f"Invalid timestamp: {text}")
    return result


def youtube_url(value):
    parsed = urlparse(str(value).strip())
    host = (parsed.hostname or "").lower()
    if host in {"youtube.com", "www.youtube.com", "m.youtube.com"}:
        video_id = parse_qs(parsed.query).get("v", [""])[0]
    elif host in {"youtu.be", "www.youtu.be"}:
        video_id = parsed.path.strip("/")
    else:
        return None
    if not re.fullmatch(r"[A-Za-z0-9_-]{11}", video_id):
        return None
    return f"https://www.youtube.com/watch?v={video_id}"


def read_csv(path, required):
    with Path(path).open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if not set(required).issubset(reader.fieldnames or []):
            raise ValueError(f"{path} needs columns: {', '.join(required)}")
        return list(reader)


def load_candidates(interval_path, links_path, speakers=None):
    links = {}
    for row in read_csv(links_path, ("speaker", "video_fn", "video_link")):
        key = (row["speaker"].strip(), row["video_fn"].strip())
        url = youtube_url(row["video_link"])
        if key in links and links[key] != url:
            raise ValueError(f"Conflicting URLs for {key}")
        links[key] = url
    rows = read_csv(interval_path, ("speaker", "video_fn", "start_time", "end_time"))
    candidates, rejected = [], []
    for number, row in enumerate(rows, 2):
        speaker, video_fn = row["speaker"].strip(), row["video_fn"].strip()
        if speakers and speaker not in speakers:
            continue
        try:
            start, end = seconds(row["start_time"]), seconds(row["end_time"])
            if end <= start:
                raise ValueError("Nonpositive interval")
            url = links.get((speaker, video_fn))
            if not url:
                raise ValueError("No supported YouTube URL")
            candidates.append(dict(speaker=speaker, video_fn=video_fn, url=url,
                                   source_start=start, source_end=end,
                                   interval_id=row.get("interval_id", ""),
                                   dataset_split=row.get("dataset", ""), csv_row=number))
        except ValueError as error:
            rejected.append(dict(csv_row=number, reason=str(error)))
    # Stable source ordering downloads each source once. Trim overlaps globally
    # by YouTube ID so repeated annotations do not count as extra footage.
    candidates.sort(key=lambda c: (c["url"], c["source_start"], c["source_end"], c["speaker"]))
    result, covered_until = [], {}
    for item in candidates:
        item["source_start"] = max(item["source_start"], covered_until.get(item["url"], 0))
        if item["source_start"] >= item["source_end"]:
            rejected.append(dict(csv_row=item["csv_row"], reason="Duplicate/covered interval"))
            continue
        covered_until[item["url"]] = item["source_end"]
        key = f"{item['url']}|{item['source_start']:.9f}|{item['source_end']:.9f}"
        item["key"] = hashlib.sha256(key.encode()).hexdigest()[:20]
        result.append(item)
    if not result:
        raise ValueError("No usable annotated intervals matched the requested speakers and source URLs")
    return result, rejected


def acquire_metadata(directory, source, interval_file=None, links_file=None):
    directory.mkdir(parents=True, exist_ok=True)
    interval_path, links_path = directory / "intervals_df.csv", directory / "video_links.csv"
    for supplied, dest in ((interval_file, interval_path), (links_file, links_path)):
        if supplied:
            if dest.exists() and fingerprint(supplied) != fingerprint(dest):
                raise ValueError(f"Existing metadata differs: {dest}. Use a new output directory.")
            if Path(supplied).resolve() != dest.resolve():
                shutil.copyfile(supplied, dest)
    if interval_path.exists() and links_path.exists():
        return interval_path, links_path
    if source == "show":
        import requests
        remote_names = {interval_path: "SHOW_intervals_subject4.csv", links_path: "videos_links.csv"}
        for dest, name in remote_names.items():
            if dest.exists():
                continue
            response = requests.get(SHOW_BASE + name, timeout=(15, 60))
            response.raise_for_status()
            partial = dest.with_suffix(".csv.partial")
            partial.write_bytes(response.content)
            partial.replace(dest)
        save(directory / "provenance.json", {
            "source": "SHOW/TalkSHOW derived Speech2Gesture subset (four speakers, intervals split up to 10s)",
            "urls": {p.name: SHOW_BASE + name for p, name in remote_names.items()},
            "reference": "https://github.com/yhw-yhw/SHOW"})
    else:
        import gdown
        # List public metadata only: never download pose archives or frame bundles.
        files = gdown.download_folder(url=DRIVE_FOLDER, skip_download=True,
                                     remaining_ok=True, quiet=True, use_cookies=False)
        if not files:
            raise RuntimeError("Original Speech2Gesture metadata is unavailable at its published Drive link. "
                               "Provide --intervals-csv and --links-csv, or explicitly choose "
                               "--metadata-source show for the related SHOW/TalkSHOW subset.")
        for file in files:
            name = Path(file.path).name
            dest = interval_path if name == "intervals_df.csv" else (
                links_path if name in {"video_links.csv", "videos_links.csv"} else None)
            if dest is not None and not dest.exists():
                partial = dest.with_suffix(".csv.partial")
                if not gdown.download(id=file.id, output=str(partial), quiet=True, use_cookies=False):
                    raise RuntimeError(f"Cannot download metadata: {name}")
                partial.replace(dest)
        save(directory / "provenance.json", {"source": "Original Speech2Gesture", "url": DRIVE_FOLDER})
    if not interval_path.is_file() or not links_path.is_file():
        raise RuntimeError("Metadata source did not supply both required CSV files")
    return interval_path, links_path


def download_source(url, directory, ffmpeg, max_height, cookies=None):
    from yt_dlp import YoutubeDL
    video_id = parse_qs(urlparse(url).query)["v"][0]
    destination = directory / f"{video_id}.mkv"
    if destination.is_file():
        return destination
    options = {"format": f"bv*[height<={max_height}]+ba/b[height<={max_height}]",
               "outtmpl": str(directory / f"{video_id}.%(ext)s"), "merge_output_format": "mkv",
               "postprocessors": [{"key": "FFmpegVideoRemuxer", "preferedformat": "mkv"}],
               "ffmpeg_location": str(Path(ffmpeg).parent), "noplaylist": True,
               "socket_timeout": 30, "retries": 2, "fragment_retries": 2,
               "continuedl": True, "cachedir": False, "quiet": True,
               "no_warnings": False, "overwrites": False, "noprogress": True}
    if cookies:
        options["cookiefile"] = str(cookies)
    node = Path(sys.prefix) / "bin/node"
    if node.is_file():
        options["js_runtimes"] = {"node": {"path": str(node)}}
    with YoutubeDL(options) as downloader:
        downloader.download([url])
    if not destination.is_file():
        raise RuntimeError("Source downloader did not produce the expected MKV cache file")
    return destination


def extract(source, output, start, duration, ffmpeg, ffprobe):
    """Re-encode time interval only. No crop, face selection, resize, or FPS filter."""
    partial = output.with_name(output.stem + ".partial.mp4")
    try:
        media.run([ffmpeg, "-v", "error", "-y", "-ss", f"{start:.9f}", "-i", source,
                   "-t", f"{duration:.9f}", "-map", "0:v:0", "-map", "0:a:0",
                   "-c:v", "libx264", "-crf", "18", "-preset", "fast", "-pix_fmt", "yuv420p",
                   "-c:a", "aac", "-b:a", "192k", "-movflags", "+faststart", partial])
        info = media.probe(partial, ffprobe)
        if not info["has_audio"] or abs(info["duration_seconds"] - duration) > 0.15:
            raise ValueError("Extracted clip has missing audio or an unexpected duration")
        media.run([ffmpeg, "-v", "error", "-xerror", "-i", partial, "-f", "null", "-"])
        partial.replace(output)
        return info
    finally:
        partial.unlink(missing_ok=True)


def run_download(args, candidates, state, manifest_path):
    root = manifest_path.parent
    total = sum(clip["duration_seconds"] for clip in state["clips"])
    done = {clip["key"] for clip in state["clips"]}
    failed_this_run = set()
    for clip in state["clips"]:
        path = root / clip["path"]
        if not path.is_file() or fingerprint(path) != clip["sha256"]:
            raise ValueError(f"Completed clip is missing or changed: {path}")
    for item in candidates:
        if total >= state["target_seconds"] - 0.05:
            break
        if item["key"] in done or (item["key"] in state["failed_intervals"] and not args.retry_failed):
            continue
        if item["url"] in state["failed_sources"] and not args.retry_failed:
            continue
        if item["url"] in failed_this_run:
            continue
        try:
            print(f"Source: {item['speaker']} / {item['video_fn']}", flush=True)
            source = download_source(item["url"], root / "source_videos", args.ffmpeg,
                                     args.max_height, args.cookies_file)
            info = media.probe(source, args.ffprobe)
            if not info["has_audio"]:
                raise ValueError("Source has no audio")
        except Exception as error:
            failed_this_run.add(item["url"])
            state["failed_sources"][item["url"]] = str(error)
            save(manifest_path, state)
            if "confirm you’re not a bot" in str(error) or "confirm you're not a bot" in str(error):
                state["status"] = "access_required"
                save(manifest_path, state)
                raise RuntimeError("YouTube requires sign-in verification. Supply your own --cookies-file "
                                   "and --retry-failed to resume. No completed clips were lost.") from error
            print(f"Skipping unavailable source: {error}", flush=True)
            continue
        start, end = item["source_start"], item["source_end"]
        try:
            if end > info["duration_seconds"] + 0.1:
                raise ValueError("Annotated end is beyond the available video; timestamp alignment is uncertain")
            numerator, denominator = map(float, info["frame_rate"].split("/"))
            fps = numerator / denominator
            if not math.isfinite(fps) or fps <= 0:
                raise ValueError("Source frame rate is invalid")
            requested = min(end - start, state["target_seconds"] - total)
            # Do not exceed the remaining budget when cutting on frame boundaries.
            duration = math.floor(requested * fps + 1e-6) / fps
            if duration < 1 / fps:
                continue
            filename = f"{len(state['clips']) + 1:05d}_{item['key']}.mp4"
            output = root / "clips" / filename
            result = extract(source, output, start, duration, args.ffmpeg, args.ffprobe)
            if total + result["duration_seconds"] > state["target_seconds"] + .05:
                output.unlink()
                raise ValueError("Clip exceeds target duration tolerance")
            clip = dict(item, path=f"clips/{filename}", requested_duration_seconds=duration,
                        duration_seconds=result["duration_seconds"], media=result,
                        source_cache=str(source.relative_to(root)), sha256=fingerprint(output),
                        extracted_end_seconds=start + duration)
            state["clips"].append(clip)
            state["failed_intervals"].pop(item["key"], None)
            state["failed_sources"].pop(item["url"], None)
            total += result["duration_seconds"]
            state["completed_seconds"] = total
            save(manifest_path, state)
            print(f"Saved {filename}; {total:.2f}/{state['target_seconds']:.2f} seconds", flush=True)
        except Exception as error:
            state["failed_intervals"][item["key"]] = str(error)
            save(manifest_path, state)
            print(f"Skipping interval: {error}", flush=True)
    state["completed_seconds"] = total
    state["status"] = "completed" if total >= state["target_seconds"] - .05 else "incomplete"
    save(manifest_path, state)
    if state["status"] != "completed":
        raise RuntimeError(f"Only {total:.2f}/{state['target_seconds']:.2f} seconds obtained. "
                           "See failed_sources/failed_intervals in manifest.json. "
                           "Rerun with --retry-failed after fixing availability/access.")
    print(f"Completed: {len(state['clips'])} clips, {total:.2f}s. Files: {root / 'clips'}", flush=True)


def execute(args):
    if not math.isfinite(args.target_hours) or args.target_hours <= 0:
        raise ValueError("--target-hours must be finite and positive")
    if args.max_height < 144:
        raise ValueError("--max-height must be at least 144")
    root = args.output_dir.resolve()
    root.mkdir(parents=True, exist_ok=True)
    with (root / ".download.lock").open("w") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise RuntimeError("Another downloader is already using this output directory") from None
        for folder in ("source_videos", "clips", "metadata"):
            (root / folder).mkdir(exist_ok=True)
        interval_path, links_path = acquire_metadata(root / "metadata", args.metadata_source,
                                                    args.intervals_csv, args.links_csv)
        candidates, rejected = load_candidates(interval_path, links_path, args.speakers)
        metadata = {p.name: fingerprint(p) for p in (interval_path, links_path)}
        config = dict(target_seconds=args.target_hours * 3600, metadata_source=args.metadata_source,
                      speakers=sorted(args.speakers) if args.speakers else [], max_height=args.max_height,
                      metadata_sha256=metadata)
        manifest_path = root / "manifest.json"
        if manifest_path.exists():
            state = json.loads(manifest_path.read_text())
            if any(state.get(key) != value for key, value in config.items()):
                raise ValueError("Run settings or metadata changed. Restore original flags or choose a new output directory.")
        else:
            state = dict(config, schema_version=1, status="planned", completed_seconds=0,
                         clips=[], failed_sources={}, failed_intervals={},
                         skipped_metadata=rejected, candidate_count=len(candidates),
                         candidate_seconds=sum(c["source_end"] - c["source_start"] for c in candidates))
            save(manifest_path, state)
        if args.dry_run:
            remaining = config["target_seconds"]
            plan = []
            for item in candidates:
                if remaining <= 1e-6:
                    break
                length = min(item["source_end"] - item["source_start"], remaining)
                plan.append(dict(item, requested_duration_seconds=length))
                remaining -= length
            save(root / "plan.json", dict(target_seconds=config["target_seconds"],
                                          planned_seconds=config["target_seconds"] - remaining,
                                          source_videos=len({c["url"] for c in plan}), clips=plan))
            print(f"Plan: {len(plan)} clips, {len({c['url'] for c in plan})} source videos, "
                  f"{config['target_seconds'] - remaining:.2f}s. No video downloads.", flush=True)
            return
        media.require_tools(args.ffmpeg, args.ffprobe)
        state["status"] = "downloading"
        save(manifest_path, state)
        try:
            run_download(args, candidates, state, manifest_path)
        except KeyboardInterrupt:
            state["status"] = "interrupted"
            save(manifest_path, state)
            raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target-hours", type=float, default=2)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "dataset/lucy/speech2gesture")
    parser.add_argument("--metadata-source", choices=("original", "show"), default="show",
                        help="SHOW is a related four-speaker subset, not the full original release")
    parser.add_argument("--intervals-csv", type=Path)
    parser.add_argument("--links-csv", type=Path, help="Accepts video_links.csv or videos_links.csv")
    parser.add_argument("--speakers", nargs="+", help="Optional speaker names, e.g. oliver seth")
    parser.add_argument("--max-height", type=int, default=720, help="Maximum source download resolution; no spatial crop")
    parser.add_argument("--cookies-file", type=Path, help="Optional user-supplied yt-dlp cookies file")
    parser.add_argument("--dry-run", action="store_true", help="Fetch/read metadata and plan only; no video downloads")
    parser.add_argument("--retry-failed", action="store_true")
    parser.add_argument("--ffmpeg", default=str(Path(sys.prefix) / "bin/ffmpeg"))
    parser.add_argument("--ffprobe", default=str(Path(sys.prefix) / "bin/ffprobe"))
    args = parser.parse_args()
    try:
        execute(args)
    except (Exception, KeyboardInterrupt) as error:
        parser.exit(1, f"Speech2Gesture extraction stopped: {error}\n")


if __name__ == "__main__":
    main()
