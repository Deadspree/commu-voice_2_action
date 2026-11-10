# Gemini Live 3.8 conversation with CommU

Unity streams your microphone to a Python backend connected to **Gemini Live
3.8**. Gemini understands speech and returns spoken English directly. The backend
feeds that exact response audio to your trained HuBERT motion model and sends
paired audio/joint segments to Unity for synchronized playback.

The previous separate recognition/chat/TTS adapters have been removed. Live
conversation uses Google's cloud API; motion inference runs locally. No new
motion training or normalization changes are required.

## Start on the Mac

Run these commands from the repository root:

```bash
conda activate commu-voice-2-action
```

The Google SDK has been installed in this laptop's project environment. For
another machine with the project's Torch dependencies already installed:

```bash
python -m pip install 'google-genai>=2.25,<3' 'websockets>=14,<18' \
  'scipy>=1.11,<2' 'PyYAML>=6,<7' python-dotenv
```

Add your key to the **existing root `.env`**, keeping its other entries:

```dotenv
GEMINI_API_KEY=your_key_here
```

Only the Python backend reads the key. Never put it into Unity assets or source
code. No OpenAI key is required. Your API key must have access to the configured
Gemini Developer API model. This adapter uses the API-key endpoint, rather than
Vertex AI service-account authentication.

Check authentication and model access without loading HuBERT or sending audio:

```bash
python -m robot_conversation.server --check-gemini
```

Then start the backend:

```bash
python -m robot_conversation.server
```

Wait for `CommU listening on ws://127.0.0.1:8765`. In the Unity project at
`simulation/unity/commu`, choose **CommU → Open Conversation Scene**, press
**Play**, and wait for **Listening**. A Gemini Live connection opens when Unity
connects. There is one independent Gemini session for each Unity connection.

Hold **Space** to speak and release to finish. Microphone audio streams in
100 ms packets while held; it does not wait for a full recording. You can also
use **Start speaking / Finish speaking**, or type and click **Send text**.
Space types into a focused text field; click outside it before using the shortcut.
The first recording may ask for microphone permission. If denied, enable Unity
Editor in macOS **Privacy & Security → Microphone**.

**Stop / interrupt**, or starting another recording, immediately stops audio and
motion and clears pending packets. Cancelled inference results cannot restart
playback. A cancelled or failed Gemini session is closed; the next request opens
a new one seeded only with the last fully played conversation transcripts.
Completed turns share a persistent Live session. Reconnecting Unity starts a
fresh conversation. Session expiry/network failures display an error; reconnect
to continue. There is no automatic paid retry loop.

Microphone input and conversation content are sent to Google and use your lab's
API quota/billing. The UI labels the generated voice. Backend errors do not echo
SDK response bodies, audio or keys.

## Configuration

Edit `robot_conversation/configs/conversation.yaml`:

| Setting | Default |
|---|---|
| Gemini model | `gemini-3.8-live` |
| Voice | `Aoede` |
| Motion checkpoint | `speech_to_motion/checkpoints/base_all_last2/last.pt` |
| Motion device | CPU, 4 threads |
| Response segment | 2 seconds |
| Motion context / lookahead | 0.3 / 0.2 seconds |
| Joint boundary blend | 0.12 seconds |
| User recording limit | 20 seconds |
| Response audio limit | 30 seconds total per turn |
| Response timeout | 90 seconds after finishing input |
| Context retained after cancellation | Last 10 acknowledged transcript turns |

The model ID is configurable so the lab can select an endpoint it has access to.
There is no silent fallback to another model or provider.

## Streaming, synchronization and latency

Input is mono PCM16 at 16 kHz. Push-to-talk explicitly sends Gemini
`ActivityStart` and `ActivityEnd`, with automatic VAD disabled. It does **not**
send `audio_stream_end` in this manual-VAD mode. Text input uses an explicit
completed user turn. Input/output transcriptions are enabled for subtitles.

Output is mono PCM16 at 24 kHz. The backend reads every audio part exactly once
and splits it into contiguous segments. It retains surrounding audio for motion
inference, resamples that context to 16 kHz, then crops/interpolates the predicted
angles onto each segment's 30 FPS timeline. Context is never played twice. A
short correction blends joints across boundaries; the audio samples are neither
crossfaded nor modified. Very short final tails are padded only for inference.

Unity's two alternating AudioSources schedule the segments on its DSP timeline.
Joint interpolation uses the playing source's sample position. Audio clips are
decoded when packets arrive, ahead of their scheduling deadlines. Existing Unity
physical joint limits still clamp the checkpoint's denormalized angles.

The Unity player now starts with **4 seconds of ready audio and motion** (normally
two segments). Short replies and final tails flush as soon as the backend confirms
all of their audio and motion are ready. The 0.15-second DSP scheduling lead is
separate from this reserve. Subsequent ready segments are scheduled contiguously.
If the reserve runs out, the player holds the last spoken pose, waits for another
4 seconds of ready content (or completion), and then resumes. It returns toward
its initial pose only after playback completes or is cancelled. Buffering remains
part of the active turn, so Unity cannot acknowledge a temporarily empty stream
as a finished reply. The UI displays **Buffering speech** or **Buffering more
speech** while waiting.

To tune the tradeoff, select the `CommUConversation` object in Unity and edit the
**Speech Motion Player** component:

| Inspector field | Default | Meaning |
|---|---|---|
| Startup Buffer Seconds | 4 | Ready audio/motion required before starting |
| Rebuffer Seconds | 4 | Reserve required to resume after an underrun |
| Prebuffer Seconds | 0.15 | AudioSource scheduling lead time |
| Log Timing | Enabled | Per-packet, scheduling and underrun diagnostics |

This adds initial response latency. For a longer streamed reply, preparing the
two initial segments normally needs 4.2 seconds of **generated audio**, plus
inference and transport; a short completed reply can start sooner. Gemini may
produce audio faster than real time, so this is not a measured 4.2-second wall-clock
delay. A larger reserve absorbs more delivery jitter, but no finite reserve can
hide an arbitrarily long stall or sustained inference slower than playback. The
checkpoint remains noncausal; a causal model would be a later extension.

### Timing diagnostics

The backend prints `[CommU timing]` lines for each segment. Disable them with
`diagnostics.timing_logs: false` in the YAML configuration. Unity prints
`[CommU playback]` lines in the Console and Editor log; disable these with
**Log Timing** in the Inspector. These contain turn IDs and timings, without
transcripts, audio samples or credentials.

| Field/event | What it measures |
|---|---|
| `ready_ms` | Input-finish to a segment having enough Gemini audio/context |
| `gemini_gap_ms` | Largest received-audio interval feeding that segment |
| `queue_ms` | Time waiting for the serialized motion worker |
| `motion_ms` | Context resampling, motion inference and cropping |
| `backend_ms` | Input-finish to the paired segment being ready to send |
| `send_ms` | Backend WebSocket send time |
| `arrival_gap_ms` | Interval between segment arrivals at the Unity player |
| `buffer_s` | Remaining ready audio/motion in Unity |
| `event=underrun` | Playback exhausted its ready content before completion |
| `event=resume` | Playback restarted after rebuilding its reserve |

For offline previews, elapsed timings start at the beginning of the demo turn.
Gemini arrival gaps include server generation and network delivery; they do not
by themselves distinguish those two sources. Backend and Unity use independent
monotonic clocks, so compare the reported durations rather than absolute clocks.

This version uses streaming **push-to-talk**, rather than always-on listening.
Full-duplex automatic interruption/VAD and acoustic echo cancellation are future
extensions. Use headphones if testing a future always-listening microphone path.

## Offline audio/motion preview

The local generated sample remains available at
`outputs/robot_conversation/demo_speech.wav` on this laptop:

```bash
python -m robot_conversation.server \
  --demo-audio outputs/robot_conversation/demo_speech.wav
```

Open the scene, press Play, and choose **Preview audio + predicted motion**.
This runs the actual trained checkpoint through the same segmented output path,
without a Gemini key or paid requests. Another server-local file can be supplied
(up to 30 seconds). FFmpeg must be on PATH for decoding. Stop the preview server
with Ctrl+C before starting the live server on port 8765.

## Optional lab GPU

Copy the updated conversation package, motion runtime and checkpoint to the lab,
install dependencies there and configure its own root `.env`. Run:

```bash
CUDA_VISIBLE_DEVICES=3 python -m robot_conversation.server --device cuda
```

Forward the loopback service from the Mac:

```bash
ssh -N -J lab-jump -L 127.0.0.1:8765:127.0.0.1:8765 minhle@10.0.1.203
```

Unity continues using `ws://127.0.0.1:8765`. This implementation is installed on
the laptop; it does not modify the lab training environment automatically.

## Verification

```bash
python -m pytest robot_conversation/tests speech_to_motion/tests -q
```

Tests use the real Google SDK types with fake sessions: configuration/history
seeding, streamed microphone packets, every audio part, PCM preservation,
segmentation/context/tiny tails, acknowledgement, cancellation during inference,
interruption, redacted errors, bounded messages and a real local WebSocket preview.
They do not make paid calls. `CommU → Validate Conversation` checks scene wiring,
protocol, interpolation, ownership and joint clamping. A muted Play Mode test can
run against an already-running preview server using Unity's
`-executeMethod ConversationPlaybackValidation.RunBatch`. To point that test at a
separate preview backend, add `--conversation-backend ws://127.0.0.1:8766`.
`SpeechBufferingValidation.RunBatch` runs a muted, offline Play Mode check with
intentionally delayed packets: startup reserve, continuous adjacent segments,
underrun counting, pose holding, refill, short replies, cancellation and late
completion. It needs no backend or API key.

A Gemini API key was not configured during implementation. Live model access,
voice quality, microphone capture and end-to-end cloud latency still require a
live check with the lab's key; mocked tests and offline playback do not establish
those results.

Official references:
[Gemini 3.8 Live model](https://ai.google.dev/gemini-api/docs/models/gemini-3.8-live),
[SDK setup](https://ai.google.dev/gemini-api/docs/live-api/get-started-sdk),
[Live capabilities and activity handling](https://ai.google.dev/gemini-api/docs/live-api/capabilities).
