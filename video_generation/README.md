# Human performance → CommU video (Step 3)

Inputs: a recorded human performance and a robot reference image.
Output: a Lucy robot video with the input voice attached. This package does not
run pose estimation or generate joint angles.

## Setup

Use the project's existing Conda environment for both Python dependencies and
FFmpeg:

```bash
conda activate commu-voice-2-action
conda install -n commu-voice-2-action -c conda-forge ffmpeg
python -m pip install 'requests>=2.32.3,<3' 'python-dotenv>=1.0.1,<2'
```

The pipeline uses FFmpeg and ffprobe from the same environment as its Python
interpreter. No separate virtual environment or executable-path exports are
needed. Explicit `--ffmpeg` and `--ffprobe` flags remain available for diagnostics.
Old `FFMPEG_BINARY` and `FFPROBE_BINARY` exports are no longer used.

Create `.env` at the repository root using `.env.example` as a template:

```dotenv
DECART_API_KEY=your_key_here
```

The file is loaded automatically. An existing environment variable takes
precedence. Credentials are never saved in run manifests. `.env` and
`dataset/lucy/` are ignored by Git.

## Supplied test inputs

- `dataset/lucy/human_videos/me-test-3.mp4` — approximately 14 seconds, including audio.
- `video_generation/robot_references/commu_front.png` — supplied CommU image.

Run commands from the repository root. Defaults select these two inputs.

Prepare locally, without making an API request:

```bash
python -m video_generation \
  --output-dir dataset/lucy/generation_runs/my_run \
  --prepare-only
```

Submit the prepared run (uploads the video and reference; uses paid Decart API):

```bash
python -m video_generation \
  --output-dir dataset/lucy/generation_runs/my_run \
  --resume
```

Or prepare and generate in a single command, selecting your own inputs:

```bash
python -m video_generation \
  --video dataset/lucy/human_videos/me-test-3.mp4 \
  --reference video_generation/robot_references/commu_front.png \
  --output-dir dataset/lucy/generation_runs/another_run
```

Use `--start 2 --duration 5` on a new run to select a short performance segment.
Resume always uses the saved inputs, prompt, settings, and clip boundaries;
new input/config flags do not change an existing run.

The existing example run can be resumed using:

```bash
conda activate commu-voice-2-action
python -m video_generation \
  --output-dir dataset/lucy/generation_runs/me-test-3_run_001 --resume
```

## Run files

```text
dataset/lucy/generation_runs/my_run/
├── manifest.json          # Input hashes, timeline, settings, job ID, progress
├── config.json            # Snapshot of the configuration
├── prompt.txt             # Exact prompt sent to Lucy
├── prepared.mp4           # Human video normalized for upload
├── source_audio.m4a       # Audio preserved from prepared.mp4; absent if silent
├── lucy_raw.mp4           # Unmodified downloaded result
├── robot_with_audio.mp4   # Final video, using the input performance's voice
└── validation.json        # Technical checks; visual review still required
```

Preparation fits the video inside a 1280×720 or 720×1280 canvas, adding black
padding rather than cropping or stretching. It encodes H.264 video at 30 fps
and AAC audio, resets the clip timeline to zero, and retains the selected source
start offset in the manifest. Thus `source_audio.m4a` contains the prepared AAC
audio, not a bit-exact copy of the source's original audio encoding. No speech
synthesis or voice conversion is performed.
M4A is used instead of the initially proposed MKA because it preserves AAC
encoder-delay metadata during extraction and merging.

The API currently accepts files up to 200 MB. Oversize prepared clips are
rejected locally; use a shorter segment. Automatic chunking is intentionally
omitted because separate generations may produce visible boundary changes.

The output is checked for decodability, duration (within 150 ms), and the
expected presence/absence of audio. Visual identity, gesture accuracy, and
lip synchronization require manual review. Reattaching the voice cannot fix
incorrect generated mouth movements. The manifest records the preparation
timeline for future dataset alignment.

## Interruptions and errors

Use the same output directory with `--resume`. Job IDs are saved immediately;
polling and download failures do not create a new generation. Concurrent
processes cannot use the same run directory. A fresh generation needs a new
directory.

If a submission is interrupted before its job ID is saved, the outcome is
uncertain. The pipeline refuses to automatically repeat the POST. Look up
the job in the Decart dashboard, then recover it with:

```bash
python -m video_generation --output-dir dataset/lucy/generation_runs/my_run \
  --resume --job-id YOUR_EXISTING_JOB_ID
```

If no job was created, use a new output directory after verifying that fact in
the dashboard. A failed remote job is reported; it is never resubmitted
automatically. For a corrupt download, remove only `lucy_raw.mp4` and resume to
download the same job again. Preserve the manifest.

The run lock uses `fcntl`, supported on macOS and Linux.

## Download two hours of Speech2Gesture-derived footage

The public original Speech2Gesture Drive metadata link currently returns 404.
The project's maintainer points to the related SHOW/TalkSHOW release in
https://github.com/amirbar/speech2gesture/issues/23. With the user's approval,
the downloader defaults to the public SHOW four-speaker metadata, available at
https://github.com/yhw-yhw/SHOW. These are derived annotations (intervals split
into up to 10-second chunks), not a mirror of the complete original release.
Metadata provenance is recorded in `metadata/provenance.json`.

Setup in the existing project environment:

```bash
conda activate commu-voice-2-action
conda install -n commu-voice-2-action -c conda-forge ffmpeg nodejs
python -m pip install 'yt-dlp[default]>=2026.10.26' 'gdown>=5.2.2,<6'
```

Download and extract the two-hour subset:

```bash
python -m video_generation.download_speech2gesture --target-hours 2
```

Files are kept under the existing dataset directory:

```text
dataset/lucy/speech2gesture/
├── metadata/           # Interval/link CSVs and source provenance
├── source_videos/      # Only needed originals, cached as MKV
├── clips/              # H.264/AAC MP4 time intervals, original framing
├── manifest.json       # Settings, hashes, clips, total duration, failures
└── plan.json           # Written by --dry-run
```

The script downloads each needed source once, extracts annotated time intervals,
and counts actual exported video durations toward 7,200 seconds. It trims the
last interval to the remaining duration, with a 50 ms frame-boundary tolerance.
Overlapping annotations do not count the same source time twice. There is no
face detection, spatial cropping, gesture filtering, or motion estimation.
Original aspect ratio is retained. Downloads use an available format up to
720p by default; use `--max-height 1080` in a new output directory to change it.
MP4s are re-encoded, not lossless copies. Source video URLs and exact requested
start/end times remain in the manifest for future audio/joint alignment, along
with the source annotation's `dataset_split` and `interval_id` where present.
Frame boundaries can shift the effective cut by approximately one source frame.

The candidate order is deterministic and grouped by source URL to minimize
downloads. This is not balanced speaker sampling. Specify `--speakers oliver
seth` to restrict speakers. A plan is an estimate before availability checks:

```bash
python -m video_generation.download_speech2gesture --target-hours 2 --dry-run
```

Rerun the same command to resume; no `--resume` flag is needed. Completed clips
are hash-checked and skipped. Partial downloads can resume through yt-dlp;
partial MP4 extractions are regenerated. Keep the same settings and CSVs when
resuming; changed settings require a different `--output-dir`. Original videos
are cached and may occupy more disk space than the resulting two hours.

Unavailable sources are skipped and replaced by later candidates. Failures
are recorded in the manifest and skipped on future runs unless `--retry-failed`
is provided. If YouTube requests verification, the script stops and preserves
progress; supply your own Netscape-format `--cookies-file /path/to/cookies.txt`
with `--retry-failed` if needed. Cookies are not read from browsers automatically.
Node.js is resolved from the same Python/Conda environment for YouTube JS support.
The script reports `incomplete` instead of claiming two hours if available
footage is exhausted. No Lucy API requests are made by this downloader.

If you obtain the original release's CSVs, use them explicitly:

```bash
python -m video_generation.download_speech2gesture \
  --metadata-source original \
  --intervals-csv /path/to/intervals_df.csv \
  --links-csv /path/to/videos_links.csv \
  --output-dir dataset/lucy/speech2gesture_original \
  --target-hours 2
```

Both `video_links.csv` and `videos_links.csv` names are accepted. Required interval
columns are `speaker,video_fn,start_time,end_time`; link columns are
`speaker,video_fn,video_link`. Original metadata can also be fetched automatically
if its published Drive link becomes available again. Only metadata is fetched,
not pose archives or image bundles. Preserve the source dataset citations and
applicable source-video usage terms when using the downloaded material.

## Tests

```bash
python -m unittest discover -s video_generation/tests -v
```

API reference: https://docs.platform.decart.ai/models/video/video-editing
