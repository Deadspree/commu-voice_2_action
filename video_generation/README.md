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

## Tests

```bash
python -m unittest discover -s video_generation/tests -v
```

API reference: https://docs.platform.decart.ai/models/video/video-editing
