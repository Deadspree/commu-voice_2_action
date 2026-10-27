# CommU human motion tracking

One person, one camera, local Python → UDP → Unity. Run commands from the repository
root with `conda activate commu-voice-2-action`.

## Start

1. Open `simulation/unity/commu` in Unity 6000.5.5f1.
2. Open `Assets/Scenes/MediaPipeTracking.unity` and press Play.
3. Run `python -m motion_tracking.track --camera 0`.
4. Face the camera and hold your head neutral for 20 valid face frames.
   The preview shows head calibration progress, live tracking status, and sent angles.
   C resets calibration; Q/Escape quits. Body tracking separately needs shoulders/hips.

Models are downloaded once by `python -m motion_tracking.download_models` into
`motion_tracking/models/` (ignored by Git). Both models use Google's version-1
float16 task bundles. Paths can be changed with `--model-dir` on both commands.
MediaPipe runs on the CPU by default. Face and pose use blocking VIDEO mode even
for webcam frames so results always correspond to the same frame and timestamp.
The preview is mirrored for convenience; inference and robot left/right are not.

`--video /path/to/video.mp4` replaces the webcam. Playback is paced to source video
timestamps (FPS fallback); slow inference can slow playback. Audio is not played or
recorded. `--no-preview` and `--max-frames N` support headless checks. Ctrl+C exits.

## Mapping and tuning

The mapping outputs degrees in the same 14-joint order as `CommUJoint`:

| Indices | Motion | Source |
| --- | --- | --- |
| 0–1 | Body yaw/pitch | Shoulder/hip frame, relative to neutral |
| 2–3 / 4–5 | Right/left arm pitch/roll | Upper-arm direction relative to torso |
| 6–8 | Head pitch/yaw/roll | Face matrix relative to calibrated camera-facing neutral |
| 9–11 | Eye pitch, right/left yaw | Approximate eye-look blendshape differences |
| 12 | Shared eyelid | Mean blink score, threshold mapping and brief hold |
| 13 | Mouth | Jaw-opening score |

Pose coordinates are converted from x-right/y-down/z-away to x-right/y-up/z-near
before combining with the face matrix. The retargeter clamps all 14 outputs to
the Unity joint limits: body pitch ±90°, arm pitch ±180°, eye pitch ±30°,
eyelid 0–45°, mouth ±20°, and face yaw ±10°. Arm pitch currently
uses an elevation mapping from -180° to 0°; direct controller input can use its
full -180° to 180° range. Arm roll remains within CommU's joint limits. This is
expressive retargeting, not anatomical inverse kinematics. Elbow bending, brows,
independent eyelid motion, and detailed lip shapes cannot be reproduced by this
rig. Eye-look scores are an approximation, not calibrated gaze tracking.

Head calibration uses the first 20 valid face orientations; body calibration uses
20 valid torso orientations independently. Arms use absolute body-relative
directions; expressions use blendshape scores directly. A completed calibration
is retained even when tracking is lost; the preview now shows current tracking
status separately instead of continuing to display only “calibrated.”

To test the head independently:

```bash
python -m motion_tracking.track --camera 0 --head-only
```

This runs Face Landmarker only and holds body/arms at zero, while head, eyes,
eyelids, and mouth remain active. Face the camera, press C, hold neutral until
**Head: Tracking**, then turn left/right, nod, and tilt. Shoulders and hips are
not required. Watch the sent pitch/yaw/roll values in the preview.

The default `head_reference` is `camera`: head movement is measured relative to
your calibrated orientation in the camera frame. It includes movement caused by
turning your whole torso. Keep your torso still when testing the head. Optional
`head_reference: "torso"` restores torso-relative neck mapping for full-body work;
that mode requires reliable shoulders/hips and reports when they block the head.
`--head-only` always uses the camera reference. No automatic switching between
reference frames occurs when landmarks disappear.

Copy `motion_tracking/config.example.json`, edit it, and pass `--config path.json`.
Configuration fields are documented in `retarget.Config`; omitted fields use the
shown defaults. `gains` and `offsets` follow the table's joint order. Gains include
signs; offsets are robot degrees. Existing Unity joint limits always apply.
If a particular rig axis moves backwards, reverse that gain. Tune mouth/eyelid
ranges using the existing Unity joint sliders before changing the mapping.

Default smoothing is 100 ms for body/arms/head, 35 ms for expressions. Unreliable
channels hold their last target for 350 ms, then ease toward configured offsets.
After 600 ms without valid packets, Unity eases all joints toward zero (the rig's
neutral pose), independently of Python. Disconnect timing is configurable in the
receiver Inspector. Increasing smoothing reduces jitter but adds latency.

## Eyelid tracking

CommU has **one shared eyelid joint**: it cannot wink one eye independently.
Test by closing both eyes. The preview shows left/right blink scores and the
actual eyelid angle sent. Press **B** in the preview to command 45 degrees for
one second, bypassing detection; both robot lids should move together. This
manual diagnostic is also included in recordings, so avoid B during dataset capture.

The average blink score maps from `blink_open_score` (default 0.10, open) to
`blink_closed_score` (default 0.55, full configured travel). These thresholds
are adjustable in the JSON config. A detected closure is held for at least
`blink_hold_seconds` (0.12 seconds) so brief detections remain visible in Unity.
This cannot recover a blink that happens entirely between captured frames.
The 45-degree joint limit allows more closure; on the current mesh it produces a
partial visible closure rather than completely covering the eyeball.

If B moves the lids but closing both eyes does not, inspect the L/R scores:
low values indicate a detection/sensitivity issue. Missing scores are displayed
as “not detected.” Eyelids do not depend on head or body calibration.

## Recordings for the next dataset step

`--record motion_tracking/recordings/session.jsonl` saves SENT Python targets,
source frame timestamps, validity masks, session ID, and sequence number. Existing
files are never overwritten. These are not guaranteed to be applied by Unity.

To save actual applied poses, enable **Record Applied Angles** on the Unity
**MediaPipeTracking** receiver before pressing Play. Unity prints the new CSV path
under `Application.persistentDataPath` in its Console. Each rendered update records
the clamped controller angles, Unity time, source timestamp, session/sequence,
validity mask, and a stale flag. Source timestamps may repeat when Unity runs faster
than the tracker. Stale rows reflect interpolation toward neutral, not a new human
observation. Logs close and flush when the receiver stops.

This preserves alignment information for a later human-frame/robot-render dataset.
It does not capture synchronized images or audio; old `DataCollect` image capture
remains disabled in this scene to avoid mixing live observations into the existing
pose-estimation training dataset.

## Unity integration and transport

`CS_MediaPipeReceiver` listens only on `127.0.0.1:5055`; `--port` and the Inspector
port must match. A single datagram is JSON with `version: 1`, UUID `session`,
nonnegative `sequence`, source `timestamp_ms`, 14 float `angles`, and 14 bool `valid`.
The validity mask describes fresh measurements; invalid channels may still contain
held or fading targets. Unity validates finite values, clamps angles, rejects
reordered/duplicate frames within a session, and applies the latest accepted pose
on its main thread. A new session ID permits restarting Python. Run one sender.

`CS_TestRotateBones.BeginExternalControl/ApplyExternalPose/EndExternalControl`
provide exclusive control, stop CSV playback, and invalidate queued gesture moves.
Automatic eye-look-at is restored on receiver shutdown. For another scene, add the
receiver and assign its controller; disable random-motion and automatic collectors.
The provided tracking scene already does this.

The **MediaPipeTracking** receiver also accepts a PS4-compatible gamepad through
Unity's Input System. The left stick controls the left arm and the right stick the
right arm (up/down = pitch, horizontal roll direction reversed on both sticks).
Stick direction moves the arm at a rate; releasing the stick holds the current
arm position. The D-pad left/right increases/decreases body pitch; R1/R2
increase/decrease body yaw. Releasing a control holds the current torso angle.
The stick deadzone defaults to 0.15,
arm speeds to 121.68°/s pitch and 46.8°/s roll, and torso speeds to 50.7°/s yaw and
60.84°/s pitch. Adjust these values on the receiver component in the Inspector. If no
gamepad is connected, MediaPipe continues to control the available joints as usual.

## Verify

```bash
python -m pytest motion_tracking/tests -q
python -m pip check
```

Unity's **CommU → Validate MediaPipe Receiver** menu runs a temporary-object UDP
smoke test outside Play mode: clamping, actual bone application, duplicate rejection,
new-frame delivery, sender restart, malformed/NaN input, and control release.

Manual check: neutral → raise each arm → turn torso with head aligned → turn head
alone → blink → open mouth. Then cover the camera and stop Python. Motion should
hold briefly, then return smoothly to neutral. Confirm left/right and range scaling
on your own camera before collecting training data.
