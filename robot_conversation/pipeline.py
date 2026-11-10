"""One Live session per Unity connection, with independent receive/motion tasks."""
import asyncio
import contextlib
import time
from dataclasses import dataclass, field

import numpy as np
from google.genai import types
from websockets.exceptions import ConnectionClosed

from .motion.response_buffer import ResponseBuffer, predict_segment
from .protocol import decode_pcm, packet, parse_request, response_chunk
from .session import ConversationSession


class LiveResponseError(ValueError):
    """Safe validation messages generated here, never provider exception bodies."""


@dataclass
class Turn:
    id: str
    buffer: ResponseBuffer
    queue: asyncio.Queue = field(default_factory=lambda: asyncio.Queue(maxsize=32))
    worker: object = None
    watchdog: object = None
    recording: bool = False
    input_samples: int = 0
    output_bytes: int = 0
    user: str = ''
    assistant: str = ''
    ended: bool = False
    complete: bool = False
    started_at: float = field(default_factory=time.monotonic)
    input_finished_at: object = None
    last_audio_at: object = None
    audio_gap_ms: float = 0


class LiveConversation:
    def __init__(self, provider, predictor, cfg, demo=None):
        self.provider, self.predictor, self.cfg, self.demo = provider, predictor, cfg, demo
        self.history = ConversationSession(cfg['session']['history_turns'])
        self.turn = self.live = self.receiver = self.context = None
        self.send_lock = asyncio.Lock()
        self.dirty = False

    async def send(self, kind, turn='', **values):
        import json
        async with self.send_lock:
            await self.connection.send(json.dumps(packet(kind, turn, **values), allow_nan=False))

    async def run(self, connection):
        self.connection = connection
        try:
            if self.provider:
                await self.open_live()
            await self.send('ready', state='listening', demo=self.demo is not None,
                            text='CommU uses an AI-generated Gemini voice.')
            async for raw in connection:
                try:
                    request = parse_request(raw)
                    # Bound network sends and session setup even if the provider stalls.
                    await asyncio.wait_for(self.request(request), self.cfg['gemini']['connect_timeout_seconds'])
                except (ValueError, TypeError):
                    await self.fail('Invalid request or Gemini rejected the input. Check audio format, model access and quota.')
                except asyncio.TimeoutError:
                    await self.fail('Gemini connection timed out. Reconnect and try again.')
                except ConnectionClosed:
                    raise
                except Exception:
                    await self.fail('Gemini Live request failed. Check model access, quota and network, then reconnect.')
        except ConnectionClosed:
            pass
        except Exception:
            with contextlib.suppress(ConnectionClosed):
                await self.send('error', text='Cannot connect to Gemini Live. Check GEMINI_API_KEY, model access, quota and network.')
        finally:
            await self.cancel_turn()
            await self.close_live()

    async def open_live(self):
        self.context = self.provider.connect(self.history.history)
        try:
            self.live = await asyncio.wait_for(self.context.__aenter__(), self.cfg['gemini']['connect_timeout_seconds'])
        except BaseException:
            self.context = None
            raise
        self.receiver = asyncio.create_task(self.receive_live())

    async def close_live(self):
        task, self.receiver = self.receiver, None
        if task and task is not asyncio.current_task():
            task.cancel()
            with contextlib.suppress(asyncio.CancelledError, Exception):
                await task
        self.live = None
        context, self.context = self.context, None
        if context:
            with contextlib.suppress(Exception):
                await asyncio.wait_for(context.__aexit__(None, None, None), 5)

    async def cancel_turn(self):
        turn, self.turn = self.turn, None
        if not turn:
            return
        self.history.cancel(turn.id)
        for task in (turn.worker, turn.watchdog):
            if task and task is not asyncio.current_task():
                task.cancel()
                with contextlib.suppress(asyncio.CancelledError, Exception):
                    await task

    async def fail(self, message):
        # Reset the Live connection after failure; never continue a partially sent turn.
        turn = self.turn.id if self.turn else ''
        await self.cancel_turn()
        await self.close_live()
        await self.send('error', turn, text=message)

    async def begin(self, turn_id):
        if self.turn or self.dirty:
            await self.cancel_turn()
            # A cancelled reply may already be in Gemini's context but unheard in Unity.
            # Reconnect and seed only completed/acknowledged transcript turns.
            await self.close_live()
        if self.provider and self.live is None:
            await self.open_live()
        self.dirty = False
        self.history.begin(turn_id)
        self.turn = Turn(turn_id, ResponseBuffer(self.cfg['motion']))
        self.turn.worker = asyncio.create_task(self.motion_worker(self.turn))
        return self.turn

    async def request(self, message):
        kind, identity = message['type'], message['turn_id']
        if kind == 'playback_done':
            if self.turn and self.turn.id == identity and self.turn.complete and self.history.commit(identity):
                await self.cancel_turn()
                await self.send('state', identity, state='listening')
            return
        if kind == 'cancel':
            if self.turn and self.turn.id == identity:
                await self.cancel_turn()
                await self.close_live()
                await self.send('cancelled', identity)
            return
        if kind in ('audio_start', 'text', 'demo'):
            if kind == 'demo' and self.demo is None:
                raise ValueError('Preview requires --demo-audio')
            if kind != 'demo' and not self.provider:
                raise ValueError('Preview mode: use Preview audio + predicted motion')
            turn = await self.begin(identity)
            if kind == 'demo':
                turn.user, turn.assistant = 'Preview audio', 'Audio-file preview'
                raw = np.clip(np.rint(self.demo * 32768), -32768, 32767).astype('<i2').tobytes()
                self.enqueue_audio(turn, raw)
                self.finish_output(turn)
            elif kind == 'text':
                turn.user = message['text'].strip()
                await self.send('transcript', identity, role='user', text=turn.user)
                turn.input_finished_at = time.monotonic()
                await self.live.send_client_content(turns=types.Content(role='user', parts=[types.Part(text=turn.user)]), turn_complete=True)
                self.start_timeout(turn)
                await self.send('state', identity, state='thinking')
            else:
                turn.recording = True
                await self.live.send_realtime_input(activity_start=types.ActivityStart())
                await self.send('state', identity, state='recording')
            return
        turn = self.turn
        if not turn or turn.id != identity or not turn.recording:
            return  # Late packets after cancellation cannot revive a turn.
        if kind == 'audio_chunk':
            if message.get('sample_rate') != 16000:
                raise ValueError('Microphone audio must be mono PCM16 at 16000 Hz')
            audio = decode_pcm(message.get('audio_pcm16'), 16000, .1, min_seconds=0)
            turn.input_samples += len(audio)
            if turn.input_samples > self.cfg['session']['max_recording_seconds'] * 16000:
                raise ValueError('Recording exceeds the configured duration limit')
            raw = np.rint(audio * 32768).astype('<i2').tobytes()
            await self.live.send_realtime_input(audio=types.Blob(data=raw, mime_type='audio/pcm;rate=16000'))
        elif kind == 'audio_end':
            if turn.input_samples < 1600:
                raise ValueError('Speak for at least 0.1 seconds')
            turn.recording = False
            turn.input_finished_at = time.monotonic()
            await self.live.send_realtime_input(activity_end=types.ActivityEnd())
            self.start_timeout(turn)
            await self.send('state', identity, state='thinking')

    def start_timeout(self, turn):
        async def wait():
            await asyncio.sleep(self.cfg['gemini']['response_timeout_seconds'])
            if self.turn is turn and not turn.complete:
                await self.fail('Gemini response timed out. Please try again.')
        turn.watchdog = asyncio.create_task(wait())

    def enqueue_audio(self, turn, raw):
        now = time.monotonic()
        if turn.last_audio_at is not None:
            turn.audio_gap_ms = max(turn.audio_gap_ms, (now - turn.last_audio_at) * 1000)
        turn.last_audio_at = now
        turn.output_bytes += len(raw)
        if turn.output_bytes > self.cfg['session']['max_reply_seconds'] * 24000 * 2:
            raise LiveResponseError('Gemini reply exceeded the configured audio duration limit')
        self.queue_segments(turn, turn.buffer.append(raw))

    def queue_segments(self, turn, segments):
        now = time.monotonic()
        origin = turn.input_finished_at if turn.input_finished_at is not None else turn.started_at
        for segment in segments:
            turn.queue.put_nowait((segment, now, turn.audio_gap_ms, (now - origin) * 1000))
        if segments:
            turn.audio_gap_ms = 0

    def finish_output(self, turn):
        if turn.ended:
            return
        turn.ended = True
        if not turn.output_bytes:
            raise LiveResponseError('Gemini returned no speech audio; try a different prompt')
        self.queue_segments(turn, turn.buffer.append(b'', final=True))
        turn.queue.put_nowait(None)

    async def receive_live(self):
        try:
            while self.live is not None:
                async for event in self.live.receive():
                    if getattr(event, 'go_away', None):
                        raise LiveResponseError('Gemini session is ending. Reconnect to continue.')
                    turn = self.turn
                    content = event.server_content
                    if not turn or not content:
                        continue
                    if content.interrupted:
                        identity = turn.id
                        self.dirty = True
                        await self.cancel_turn()
                        await self.send('cancelled', identity, text='Gemini interrupted this response')
                        continue
                    if turn.ended:
                        continue
                    for field, role in (('input_transcription', 'user'), ('output_transcription', 'assistant')):
                        transcript = getattr(content, field, None)
                        if transcript and transcript.text:
                            if role == 'user':
                                turn.user += transcript.text
                                text = turn.user
                            else:
                                turn.assistant += transcript.text
                                text = turn.assistant
                            if len(text) > 16000:
                                raise LiveResponseError('Gemini transcript exceeded the limit')
                            await self.send('transcript', turn.id, role=role, text=text)
                    if content.model_turn:
                        # Process every inline part exactly once; event.data is an aggregate alias.
                        for part in content.model_turn.parts or []:
                            blob = part.inline_data
                            if blob and blob.data:
                                if not blob.mime_type.startswith('audio/pcm') or ('rate=' in blob.mime_type and 'rate=24000' not in blob.mime_type):
                                    raise LiveResponseError('Gemini returned an unsupported audio format')
                                self.enqueue_audio(turn, blob.data)
                    if content.turn_complete:
                        self.finish_output(turn)
                await asyncio.sleep(0)
        except asyncio.CancelledError:
            raise
        except Exception as error:
            # Do not echo SDK exception strings: they may include credentials or transcripts.
            message = str(error) if isinstance(error, LiveResponseError) else 'Gemini Live disconnected. Check model access, quota and network, then reconnect.'
            await self.fail(message)

    async def motion_worker(self, turn):
        sequence, previous = 0, None
        try:
            while True:
                work = await turn.queue.get()
                if work is None:
                    break
                segment, queued_at, source_gap_ms, ready_ms = work
                started = time.monotonic()
                times, angles = await predict_segment(self.predictor, segment, previous, self.cfg['motion'])
                finished = time.monotonic()
                if self.turn is not turn:
                    return
                previous = angles[-1].copy()
                message = response_chunk(turn.id, sequence, '', segment.audio, 24000, times, angles)
                origin = turn.input_finished_at if turn.input_finished_at is not None else turn.started_at
                message.update(backend_elapsed_ms=(finished - origin) * 1000,
                               gemini_ready_ms=ready_ms, gemini_gap_ms=source_gap_ms,
                               motion_queue_ms=(started - queued_at) * 1000,
                               motion_inference_ms=(finished - started) * 1000)
                send_started = time.monotonic()
                await self.send('response_chunk', turn.id, **{k: v for k, v in message.items() if k not in ('version', 'type', 'turn_id')})
                if self.cfg.get('diagnostics', {}).get('timing_logs', True):
                    print(f"[CommU timing] turn={turn.id} seq={sequence} "
                          f"audio_s={len(segment.audio) / 24000:.3f} ready_ms={ready_ms:.1f} "
                          f"gemini_gap_ms={source_gap_ms:.1f} queue_ms={message['motion_queue_ms']:.1f} "
                          f"motion_ms={message['motion_inference_ms']:.1f} "
                          f"backend_ms={message['backend_elapsed_ms']:.1f} "
                          f"send_ms={(time.monotonic() - send_started) * 1000:.1f}", flush=True)
                sequence += 1
            if self.turn is turn:
                self.history.pending = (turn.user, turn.assistant)
                turn.complete = True
                if turn.watchdog:
                    turn.watchdog.cancel()
                await self.send('response_complete', turn.id, chunks=sequence)
        except asyncio.CancelledError:
            raise
        except Exception:
            await self.fail('Motion prediction failed. Check the backend checkpoint and device.')
