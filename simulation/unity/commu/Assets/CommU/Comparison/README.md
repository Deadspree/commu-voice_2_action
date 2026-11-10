# Original / Lucy / Unity comparison

In Unity choose **CommU → Open Video Comparison**, then press **Play**. The scene
starts paused; click the on-screen **Play** button. It shows three panels:

1. Original human video, resolved from the Lucy run manifest.
2. Generated `robot_with_audio.mp4`.
3. The actual Unity CommU robot driven by `robot_with_audio_joints.csv`.

The default run is `00271_6f35a901156dd4cd1fdf`. Select the `VideoPoseComparison`
object and change **Run Folder** in the Inspector to compare another completed
run. It accepts a repository-relative or absolute directory. The directory must
contain `manifest.json`, `robot_with_audio.mp4` and a timestamped pose-estimation
CSV, and the manifest must point to an available original video.

For run `00016_c61f28e4a67ee236e500`, open
`Assets/Scenes/VideoPoseComparison_00016.unity`, enter Play Mode, then click
**Play**. Its three-panel export is
`outputs/video_comparison/00016_c61f28e4a67ee236e500/comparison.mp4`
(88 frames at 30 FPS, 2.933333 seconds). The default scene retains the first run.

Both videos share play/pause, restart, seek, loop and mute controls. The generated
video's decoded time drives CSV interpolation and the original video's external
clock. The original includes `source_start_seconds` from the manifest, so trimmed
runs retain their source alignment. Playback stops at the shortest common duration
of the source video, generated video and motion timeline. Audio plays only from
the generated video. Native seeking may advance a frame or two while the decoder
presents the destination; the robot follows the actual displayed video time.

The CSV contains the 14 DINOv3 regressor outputs in **degrees**, not HuBERT motion.
The robot uses the existing external-control API and physical joint clamps. Random
motion, MediaPipe, manual operation and data collection are disabled in this new
scene. Source scenes and the robot prefab are unchanged.

For the default run, the existing export contains 296 frames at 30 FPS and names
`pose_estimation/checkpoints/pose_estimation.pt` as its inference checkpoint. It
has been reused rather than running the same inference again. To produce a CSV
for a different generated video, from the repository root:

```bash
conda activate commu-voice-2-action
python pose_estimation/inference.py --video path/to/robot_with_audio.mp4
```

## Export a comparison MP4

A ready-to-watch video for this run is saved at:

`outputs/video_comparison/00271_6f35a901156dd4cd1fdf/comparison.mp4`

It is 1920 × 424, 296 frames at 30 FPS, and 9.866667 seconds long. Each panel is
640 × 360 with a shared header. The third panel is rendered by Unity from the
actual CSV joint angles; it is not another generated video.

To regenerate it, close this Unity project first (batch rendering cannot share
an open project), then run from the repository root:

```bash
conda activate commu-voice-2-action
python -m video_generation.comparison \
  --run-dir dataset/lucy/generation_runs/speech2gesture/00271_6f35a901156dd4cd1fdf \
  --render
```

`--unity-editor` selects a different Unity executable and `--unity-project` selects
another project copy containing these scripts. Rendering requires graphics; do
not pass `-nographics`. FFmpeg and ffprobe must be available on PATH. No new Python
packages are required. PNG frames, metadata and exports are ignored under
`outputs/video_comparison/`.

To compose an export from already-rendered frames without opening Unity:

```bash
python -m video_generation.comparison \
  --run-dir dataset/lucy/generation_runs/speech2gesture/00271_6f35a901156dd4cd1fdf
```

The export checks the CSV fingerprint against Unity's render metadata, checks for
missing render frames and verifies encoded duration. It includes one audio track
from the generated video and trims all panels to a common 30 FPS timeline.

## Validation

Unity batch entry points:

- `VideoComparisonSetup.RunBatch`: scene references, video paths, CSV joint order,
  296 poses, timestamps and interpolation.
- `VideoComparisonPlaybackValidation.RunBatch`: actual video decoding, clock
  alignment, joint replay, pause, seek and resume, with audio muted.

Python composition checks:

```bash
python -m pytest video_generation/tests/test_comparison.py -q
```
