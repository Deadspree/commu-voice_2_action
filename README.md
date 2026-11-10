# commu-voice_2_action

## English conversation in Unity

The `robot_conversation` backend connects **Gemini Live 3.8** speech-to-speech
conversation and the trained speech-to-motion model to a new
`CommUConversation` Unity scene. It supports streamed microphone push-to-talk, text input,
synchronized audio/motion playback and interruption. See the
[conversation guide](robot_conversation/README.md) for setup, API-key configuration
and an audio-file preview that works without API calls.

## Speech → CommU motion

The `speech_to_motion` package trains a 512-wide Transformer motion decoder on
HuBERT speech features, then selectively unfreezes HuBERT's final layers. It
reuses the existing Lucy audio/joint CSV pairs, creates source-grouped splits
balanced by duration, and exports joint CSVs for Unity. See the
[training guide](speech_to_motion/README.md) for dataset preparation, A6000/H100
commands, evaluation, and inference.

## Setup

### 1. Install Miniforge3 (conda)

If you don't already have Miniforge3 installed, download and install it.

**Ubuntu / Linux (x86_64):**

```bash
# Download the installer
curl -L -O https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh

# Run the installer (accept the license and default install location ~/miniforge3)
bash Miniforge3-Linux-x86_64.sh
```

**Ubuntu / Linux (ARM64 / aarch64):**

```bash
curl -L -O https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-aarch64.sh
bash Miniforge3-Linux-aarch64.sh
```

**macOS (Apple Silicon / arm64):**

```bash
curl -L -O https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-MacOSX-arm64.sh
bash Miniforge3-MacOSX-arm64.sh
```

**macOS (Intel / x86_64):**

```bash
curl -L -O https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-MacOSX-x86_64.sh
bash Miniforge3-MacOSX-x86_64.sh
```

After installation, restart your terminal (or run `source ~/.bashrc` on Ubuntu/Linux, or `source ~/.zshrc` on macOS) so `conda` is available.

> **Note:** If you prefer not to auto-activate conda's base environment on every shell, run `conda config --set auto_activate_base false` after installing.

### 2. Create and activate the conda environment

```bash
conda create -n commu-voice-2-action python=3.11 -y
conda activate commu-voice-2-action
```

### 3. Install the dependencies

From the project root:

```bash
pip install -r requirements.txt
```

### 4. Clone the DINOv3 repository

The DINOv3 backbone code lives in `third_party/dinov3`:

```bash
git clone https://github.com/facebookresearch/dinov3.git third_party/dinov3
```

> **Note on pretrained weights:** The DINOv3 pretrained weights are gated. To use a pretrained backbone, request access at [https://ai.meta.com/resources/models-and-libraries/dinov3-downloads/](https://ai.meta.com/resources/models-and-libraries/dinov3-downloads/). Once approved, Meta emails you the download URLs. You can then either:
> - download the weights locally and pass the path via `backbone_weights`, or
> - pass the emailed URL directly via `backbone_weights`.
>
> Without access, the model falls back to a randomly initialized backbone.

## Usage

### Human video → Lucy CommU video (Step 3)

Run `python -m video_generation.batch` in `commu-voice-2-action` to process all
downloaded Speech2Gesture clips through Lucy sequentially. It resumes existing
jobs and skips completed outputs. Use `--dry-run` to preview without API calls,
or `--limit 3` to start with three clips. Results are saved under
`dataset/lucy/generation_runs/speech2gesture/`. Running generation submits paid
Decart jobs; rerunning the same command resumes without starting duplicate jobs.

For a two-hour subset of the public SHOW/TalkSHOW-derived Speech2Gesture
intervals, run `python -m video_generation.download_speech2gesture --target-hours 2`
in `commu-voice-2-action`. It saves original-framing MP4 clips with audio under
`dataset/lucy/speech2gesture/clips/` and resumes from a manifest. See the video
generation guide below for metadata provenance and setup.

The `video_generation` package converts a human performance video and a CommU
reference image into a robot video with the input voice. It reads
`DECART_API_KEY` from the root `.env`, supports local preparation and resumable
Decart jobs, and saves each attempt under `dataset/lucy/generation_runs/`.

See [the video generation guide](video_generation/README.md) for setup, the
supplied test inputs, commands, and output files. This stage runs independently
of pose estimation and does not require the CUDA training dependencies.

Train the model:

```bash
python pose_estimation/train.py --output-mode point --epochs 50
python pose_estimation/train.py --output-mode distribution --loss nll --epochs 50
```

Evaluate a trained checkpoint:

```bash
python pose_estimation/evaluate.py --checkpoint pose_estimation/checkpoints/best.pt
```

## Video pose inference → Unity playback

Run from the repository root using the `commu-voice-2-action` environment:

```bash
python pose_estimation/inference.py \
  --video dataset/commu_pose_dataset/test_video/video-test.mp4
```

This writes `video-test_joints.csv` and `video-test_joints.metadata.json` beside
the source video. Each CSV row contains `frame_index,timestamp_seconds` followed
by the 14 joint angles in degrees, in the same order as image inference. All
decoded frames are processed with the same image preprocessing; decoder timestamps
are preserved, with frame-rate fallback recorded in the metadata if necessary.
Predictions are raw model outputs; Unity clamps them to the robot's joint limits.
No temporal smoothing is applied, so the preview reveals frame-to-frame jitter.

In Unity, open the CommU scene and choose **CommU → Preview Video Joint CSV…**,
then select the generated CSV. This enters Play mode and starts the robot motion.
Alternatively, set **Pose Loader → CSV Path** on `CS_TestRotateBones`, enter Play
mode, and click **Play CSV**. Use **Playback Speed = 1** for real-time playback,
and **Stop CSV Playback** to stop. Timestamped CSVs use elapsed-time scheduling;
older image CSVs use **Legacy CSV Interval (s)**, including their first data row.
Disable external tracking or other pose controllers when previewing a CSV.
This plays joint motion; it does not play or synchronize the source audio/video.

Use `--batch-size 4` to reduce inference memory usage, `--device cpu` to force CPU,
`--output path/to/joints.csv` for a custom destination, and `--overwrite` to rerun
an existing export. Image inference with `--image` remains available.

### Predict joints for all completed Lucy videos

```bash
conda activate commu-voice-2-action
python pose_estimation/inference.py \
  --video-dir dataset/lucy/generation_runs/speech2gesture
```

This recursively finds `robot_with_audio.mp4`, ignores unfinished Lucy runs,
loads the checkpoint once, and processes all pending videos. It does not use
the Lucy API or spend credits. CSVs and metadata are saved beside each video:
`robot_with_audio_joints.csv` and `robot_with_audio_joints.metadata.json`.
Rerun the same command to skip existing CSV/metadata pairs with matching source
and checkpoint paths, valid columns, finite angles, increasing timestamps, and
the expected row count. An invalid existing CSV requires `--overwrite` to replace.
This check does not hash the source video/checkpoint; if either file's contents
change at the same path, use `--overwrite` or a new output directory.

Use `--dry-run` to preview without loading the model, or `--limit 3` to process
the first three completed videos in sorted order (including existing exports).
Use `--output-dir dataset/lucy/joint_predictions` to keep CSVs in a separate tree
that preserves the input subfolders. `--batch-size`, `--checkpoint`, `--device`,
and `--overwrite` work for batch inference too. The command stops on the first
error; completed clip exports remain available when you rerun it.

## Human motion tracking → Unity CommU

Use MediaPipe Face Landmarker for expressions/head rotation and Pose Landmarker
for torso/arms. The ready-to-run scene is
`simulation/unity/commu/Assets/Scenes/MediaPipeTracking.unity`.

```bash
conda activate commu-voice-2-action
# Already installed locally. On another machine, install tracking dependencies only:
python -m pip install mediapipe==1.1.1 opencv-contrib-python==5.0.0.93 numpy
python -m motion_tracking.download_models
python -m motion_tracking.track --camera 0
```

Open **MediaPipeTracking** in Unity and press **Play**, then run the tracker from
this repository root. Face the camera with shoulders, elbows, and hips visible;
hold a neutral posture for the first 20 valid frames. Head calibration requires
only your face; body calibration requires shoulders and hips. Use `--head-only`
to test the head and expressions with the body held still. Press **C** in the preview
to recalibrate or **Q** to stop. On macOS, grant your terminal camera permission
if prompted. Change `--camera 0` to another index if needed.

The tracking scene disables the original random-motion and dataset collectors;
`SampleScene` is unchanged. The receiver owns the existing joint controller while
active and disables its automatic eye-look-at behavior. Inspect the
**MediaPipeTracking** object's **Status** field to diagnose connection issues.

For recorded input:

```bash
python -m motion_tracking.track --video /absolute/path/performance.mp4
python -m motion_tracking.track --camera 0 --record motion_tracking/recordings/targets.jsonl
```

See [the tracking guide](motion_tracking/README.md) for calibration, motion tuning,
recording actual applied angles, limitations, and tests. The existing full
`requirements.txt` includes CUDA training wheels and is not a macOS setup command;
the tracking-only command above avoids reinstalling the training stack. OpenCV's
contrib package replaces `opencv-python`; do not install both into one environment.
