import asyncio
import base64
from contextlib import asynccontextmanager
import json
from types import SimpleNamespace

import numpy as np
import pytest
from google.genai import types

from robot_conversation.pipeline import LiveConversation
from robot_conversation.protocol import decode_pcm, encode_pcm, packet, parse_request, response_chunk
from robot_conversation.session import ConversationSession
from robot_conversation.settings import load_settings
from robot_conversation.speech.gemini_live import GeminiLiveProvider
from robot_conversation.motion.response_buffer import ResponseBuffer, predict_segment


def config():
    return load_settings('robot_conversation/configs/conversation.yaml')


class FakePredictor:
    def __init__(self):
        self.audio = []
    async def predict(self, audio, rate):
        self.audio.append((audio.copy(), rate))
        times = np.arange(0, len(audio) / rate - 1e-10, 1 / 30)
        return times, np.tile(np.arange(len(times))[:, None], (1, 14)).astype(float)


class FakeConnection:
    def __init__(self):
        self.messages = []
    async def send(self, raw):
        self.messages.append(json.loads(raw))


class FakeLive:
    def __init__(self):
        self.sent = []
        self.events = asyncio.Queue()
    async def send_realtime_input(self, **kwargs):
        self.sent.append(kwargs)
    async def send_client_content(self, **kwargs):
        self.sent.append(kwargs)
    async def receive(self):
        while True:
            event = await self.events.get()
            if isinstance(event, Exception):
                raise event
            yield event
            if event.server_content and event.server_content.turn_complete:
                return


class FakeProvider:
    def __init__(self):
        self.sessions = []
        self.histories = []
        self.closed = 0
    @asynccontextmanager
    async def connect(self, history=()):
        live = FakeLive()
        self.sessions.append(live)
        self.histories.append(list(history))
        try:
            yield live
        finally:
            self.closed += 1


def event(raw=b'', final=False, **values):
    content = dict(values, turn_complete=final)
    if raw:
        # Multiple inline parts must be read, without double-counting event.data.
        mid = len(raw) // 4 * 2
        content['model_turn'] = types.Content(role='model', parts=[
            types.Part(inline_data=types.Blob(data=raw[:mid], mime_type='audio/pcm;rate=24000')),
            types.Part(inline_data=types.Blob(data=raw[mid:], mime_type='audio/pcm;rate=24000'))])
    return types.LiveServerMessage(server_content=types.LiveServerContent(**content))


async def setup():
    provider, predictor = FakeProvider(), FakePredictor()
    conversation = LiveConversation(provider, predictor, config())
    conversation.connection = FakeConnection()
    await conversation.open_live()
    return conversation, provider, predictor


async def cleanup(conversation):
    await conversation.cancel_turn()
    await conversation.close_live()


async def wait_for(predicate):
    async with asyncio.timeout(3):
        while not predicate():
            await asyncio.sleep(.001)


def test_protocol_and_timeline_bounds():
    samples = np.arange(-2000, 2000, dtype=np.int16)
    encoded = base64.b64encode(samples.astype('<i2').tobytes()).decode()
    assert encode_pcm(decode_pcm(encoded, 16000)) == encoded
    for bad in ['not base64!', '', 'YQ==']:
        with pytest.raises(ValueError): decode_pcm(bad, 16000)
    with pytest.raises(ValueError):
        parse_request(json.dumps({'version': True, 'type': 'text', 'turn_id': 'x', 'text': 'hello'}))
    with pytest.raises(ValueError): parse_request(json.dumps(packet('audio', 'x')))
    with pytest.raises(ValueError): parse_request(json.dumps(packet('text', '../bad', text='hello')))
    assert parse_request(json.dumps(packet('audio_start', 'x')))['type'] == 'audio_start'
    with pytest.raises(ValueError):
        response_chunk('x', 0, '', samples, 16000, np.array([0, .1]), np.zeros((3, 14)))


def test_segment_buffer_preserves_every_pcm_sample_and_context():
    cfg = config()['motion']
    source = np.arange(24000 * 5 + 1, dtype=np.int16)
    buffer, result = ResponseBuffer(cfg), []
    raw = source.astype('<i2').tobytes()
    for index in range(0, len(raw), 4800):
        result.extend(buffer.append(raw[index:index + 4800]))
    result.extend(buffer.append(b'', final=True))
    assert len(result) == 3
    np.testing.assert_array_equal(np.concatenate([s.audio for s in result]), source.astype(np.float32) / 32768)
    assert result[0].offset == 0 and result[1].offset == .3
    assert len(result[0].context) == round(2.2 * 24000)
    assert len(buffer.pending) == 0
    with pytest.raises(ValueError): buffer.append(b'1')


def test_tiny_final_tail_and_boundary_blending():
    async def run():
        cfg = config()['motion']
        buffer = ResponseBuffer(cfg)
        segment = buffer.append(b'\0\0', final=True)[0]
        times, angles = await predict_segment(FakePredictor(), segment, np.ones(14) * 12, cfg)
        assert len(times) == 1 and times[0] == 0
        assert np.all(angles == 12) and len(segment.audio) == 1
    asyncio.run(run())


def test_gemini_sdk_setup_and_history_without_cloud_calls():
    async def run():
        captured, live = {}, FakeLive()
        @asynccontextmanager
        async def connect(**kwargs):
            captured.update(kwargs)
            yield live
        client = SimpleNamespace(aio=SimpleNamespace(live=SimpleNamespace(connect=connect)))
        provider = GeminiLiveProvider(config()['gemini'], client)
        history = [{'role': 'user', 'content': 'Hi'}, {'role': 'assistant', 'content': 'Hello'}]
        async with provider.connect(history): pass
        cfg = captured['config']
        assert captured['model'] == 'gemini-3.8-live'
        assert cfg.response_modalities == ['AUDIO']
        assert cfg.realtime_input_config.automatic_activity_detection.disabled
        assert cfg.history_config.initial_history_in_client_content
        assert cfg.input_audio_transcription and cfg.output_audio_transcription
        assert cfg.speech_config.voice_config.prebuilt_voice_config.voice_name == 'Aoede'
        assert [turn.role for turn in live.sent[0]['turns']] == ['user', 'model']
        assert live.sent[0]['turn_complete'] is False
    asyncio.run(run())


def test_live_output_pairs_audio_motion_and_reuses_session_after_ack():
    async def run():
        conversation, provider, predictor = await setup()
        try:
            await conversation.request(packet('text', 'first', text='Hello'))
            raw = np.arange(24000 * 5, dtype=np.int16).astype('<i2').tobytes()
            await provider.sessions[-1].events.put(event(raw, final=True,
                output_transcription=types.Transcription(text='Hello there!')))
            await wait_for(lambda: conversation.turn.complete)
            chunks = [m for m in conversation.connection.messages if m['type'] == 'response_chunk']
            assert len(chunks) == 3 and [c['sequence'] for c in chunks] == [0, 1, 2]
            received = np.concatenate([decode_pcm(c['audio_pcm16'], 24000, min_seconds=0) for c in chunks])
            assert received.tobytes() == (np.frombuffer(raw, dtype='<i2').astype(np.float32) / 32768).tobytes()
            assert len(predictor.audio) == 3 and all(rate == 24000 for _, rate in predictor.audio)
            assert not conversation.history.history
            await conversation.request(packet('playback_done', 'first'))
            assert conversation.history.history[-1]['content'] == 'Hello there!'
            await conversation.request(packet('text', 'second', text='How are you?'))
            assert len(provider.sessions) == 1  # Persistent Gemini context across completed turns.
            await conversation.request(packet('cancel', 'second'))
            await conversation.request(packet('text', 'third', text='New question'))
            assert len(provider.sessions) == 2
            assert provider.histories[-1] == conversation.history.history
            assert provider.histories[-1][-1]['content'] == 'Hello there!'
        finally:
            await cleanup(conversation)
    asyncio.run(run())


def test_microphone_packets_and_explicit_activity_boundaries():
    async def run():
        conversation, provider, _ = await setup()
        try:
            await conversation.request(packet('audio_start', 'mic'))
            audio = np.arange(1600, dtype=np.float32) / 32768
            await conversation.request(packet('audio_chunk', 'mic', sample_rate=16000, audio_pcm16=encode_pcm(audio)))
            await conversation.request(packet('audio_end', 'mic'))
            sent = provider.sessions[-1].sent
            assert isinstance(sent[0]['activity_start'], types.ActivityStart)
            assert sent[1]['audio'].mime_type == 'audio/pcm;rate=16000'
            assert sent[1]['audio'].data == np.arange(1600, dtype='<i2').tobytes()
            assert isinstance(sent[2]['activity_end'], types.ActivityEnd)
            assert all('audio_stream_end' not in item for item in sent)
            await conversation.request(packet('audio_chunk', 'mic', sample_rate=16000, audio_pcm16=encode_pcm(audio)))
            assert len(sent) == 3  # Data after audio_end is discarded.
        finally: await cleanup(conversation)
    asyncio.run(run())


def test_cancellation_discards_inflight_motion_and_unplayed_context():
    async def run():
        conversation, provider, _ = await setup()
        gate, started = asyncio.Event(), asyncio.Event()
        class SlowPredictor(FakePredictor):
            async def predict(self, audio, rate):
                started.set()
                await gate.wait()
                return await super().predict(audio, rate)
        conversation.predictor = SlowPredictor()
        try:
            await conversation.request(packet('text', 'old', text='Old prompt'))
            await provider.sessions[-1].events.put(event(np.zeros(72000, dtype='<i2').tobytes(), final=True))
            await started.wait()
            await conversation.request(packet('cancel', 'old'))
            gate.set()
            assert not conversation.history.history and not conversation.turn
            assert not any(m['type'] == 'response_chunk' for m in conversation.connection.messages)
            await conversation.request(packet('text', 'new', text='New prompt'))
            assert len(provider.sessions) == 2 and provider.histories[-1] == []
            await conversation.request(packet('audio_end', 'old'))
            assert conversation.turn.id == 'new'
        finally: await cleanup(conversation)
    asyncio.run(run())


def test_server_interruption_clears_turn_and_resets_before_next_request():
    async def run():
        conversation, provider, _ = await setup()
        try:
            await conversation.request(packet('text', 'old', text='hello'))
            await provider.sessions[-1].events.put(event(interrupted=True))
            await wait_for(lambda: conversation.turn is None)
            assert conversation.connection.messages[-1]['type'] == 'cancelled'
            await conversation.request(packet('text', 'new', text='hello again'))
            assert len(provider.sessions) == 2
        finally: await cleanup(conversation)
    asyncio.run(run())


@pytest.mark.parametrize('error_type', [RuntimeError, ValueError])
def test_limits_fail_closed_and_provider_errors_are_redacted(error_type):
    async def run():
        conversation, provider, _ = await setup()
        try:
            await conversation.request(packet('text', 'bad', text='hello'))
            await provider.sessions[-1].events.put(error_type('secret-api-key-must-not-leak'))
            await wait_for(lambda: conversation.connection.messages[-1]['type'] == 'error')
            assert 'secret-api-key' not in json.dumps(conversation.connection.messages)
            assert not conversation.turn and not conversation.live
            await conversation.request(packet('audio_start', 'new'))
            with pytest.raises(ValueError):
                await conversation.request(packet('audio_chunk', 'new', sample_rate=24000, audio_pcm16=encode_pcm(np.zeros(2400))))
        finally: await cleanup(conversation)
    asyncio.run(run())


def test_history_requires_playback_acknowledgement():
    session = ConversationSession(1)
    session.begin('old'); session.pending = ('hello', 'reply'); session.cancel('old')
    assert not session.commit('old') and not session.history
    session.begin('new'); session.pending = ('new hello', 'new reply')
    assert session.commit('new') and not session.commit('new')


def test_websocket_offline_preview_and_acknowledgement():
    async def run():
        from websockets.asyncio.server import serve
        from websockets.asyncio.client import connect
        from robot_conversation.server import handle_connection
        cfg = config()
        async with serve(lambda ws: handle_connection(ws, None, FakePredictor(), cfg, np.zeros(72000)), '127.0.0.1', 0) as server:
            port = server.sockets[0].getsockname()[1]
            async with connect(f'ws://127.0.0.1:{port}') as client:
                assert json.loads(await client.recv())['type'] == 'ready'
                await client.send(json.dumps(packet('demo', 'first')))
                messages = []
                while not messages or messages[-1]['type'] != 'response_complete':
                    messages.append(json.loads(await asyncio.wait_for(client.recv(), 5)))
                assert len([m for m in messages if m['type'] == 'response_chunk']) == 2
                await client.send(json.dumps(packet('playback_done', 'first')))
                assert json.loads(await client.recv())['state'] == 'listening'
    asyncio.run(run())


def test_runtime_resamples_before_predicting(monkeypatch):
    import threading
    import torch
    import speech_to_motion.runtime as module
    runtime = object.__new__(module.MotionRuntime)
    runtime.device, runtime.model, runtime.config, runtime.normalization = torch.device('cpu'), object(), {}, {}
    runtime.lock = threading.Lock()
    def predict(model, audio, cfg, normalization, device):
        assert len(audio) == 16000 and np.isfinite(audio).all()
        return np.arange(30) / 30, np.zeros((30, 14))
    monkeypatch.setattr(module, 'predict_audio', predict)
    assert runtime.predict(np.zeros(24000), 24000)[1].shape == (30, 14)


def test_cancelling_prediction_does_not_overlap_model_execution():
    async def run():
        import time
        from robot_conversation.motion.predictor import MotionPredictor
        class Runtime:
            active = maximum = 0
            def predict(self, audio, rate):
                self.active += 1; self.maximum = max(self.maximum, self.active)
                time.sleep(.04)
                self.active -= 1
                return rate
        runtime, predictor = Runtime(), None
        predictor = MotionPredictor(runtime)
        try:
            first = asyncio.create_task(predictor.predict([], 16000))
            while not runtime.active: await asyncio.sleep(.001)
            first.cancel()
            with pytest.raises(asyncio.CancelledError): await first
            assert await predictor.predict([], 24000) == 24000 and runtime.maximum == 1
        finally: predictor.close()
    asyncio.run(run())


def test_motion_streams_before_gemini_finishes_the_reply():
    async def run():
        conversation, provider, _ = await setup()
        try:
            await conversation.request(packet('text', 'stream', text='hello'))
            await provider.sessions[-1].events.put(event(np.zeros(52800, dtype='<i2').tobytes()))
            await wait_for(lambda: any(m['type'] == 'response_chunk' for m in conversation.connection.messages))
            assert not conversation.turn.complete
            assert not any(m['type'] == 'response_complete' for m in conversation.connection.messages)
            await provider.sessions[-1].events.put(event(final=True))
            await wait_for(lambda: conversation.turn.complete)
            assert conversation.connection.messages[-1]['chunks'] == 2
        finally: await cleanup(conversation)
    asyncio.run(run())


def test_response_timeout_closes_session_without_automatic_retry():
    async def run():
        conversation, provider, _ = await setup()
        conversation.cfg['gemini']['response_timeout_seconds'] = .02
        try:
            await conversation.request(packet('text', 'timeout', text='hello'))
            await wait_for(lambda: conversation.connection.messages[-1]['type'] == 'error')
            assert conversation.turn is None and conversation.live is None
            assert provider.closed == 1 and len(provider.sessions) == 1
            assert not conversation.history.history
        finally: await cleanup(conversation)
    asyncio.run(run())


def test_segment_diagnostics_separate_source_delay_queue_and_motion(capsys):
    async def run():
        conversation, provider, _ = await setup()
        class SlowPredictor(FakePredictor):
            async def predict(self, audio, rate):
                await asyncio.sleep(.02)
                return await super().predict(audio, rate)
        conversation.predictor = SlowPredictor()
        try:
            await conversation.request(packet('text', 'timing', text='private transcript sentinel'))
            await provider.sessions[-1].events.put(event(np.zeros(2400, dtype='<i2').tobytes()))
            await wait_for(lambda: conversation.turn.last_audio_at is not None)
            await asyncio.sleep(.025)
            await provider.sessions[-1].events.put(event(np.zeros(117600, dtype='<i2').tobytes(), final=True,
                output_transcription=types.Transcription(text='private response sentinel')))
            await wait_for(lambda: conversation.turn.complete)
            chunks = [m for m in conversation.connection.messages if m['type'] == 'response_chunk']
            assert len(chunks) == 3
            for chunk in chunks:
                for key in ('backend_elapsed_ms', 'gemini_ready_ms', 'gemini_gap_ms', 'motion_queue_ms', 'motion_inference_ms'):
                    assert np.isfinite(chunk[key]) and chunk[key] >= 0
                assert chunk['motion_inference_ms'] >= 15
            assert chunks[0]['gemini_gap_ms'] >= 20
            assert chunks[0]['gemini_ready_ms'] >= 20
            assert chunks[1]['motion_queue_ms'] >= 15
        finally: await cleanup(conversation)
    asyncio.run(run())
    log = capsys.readouterr().out
    assert '[CommU timing]' in log and 'send_ms=' in log
    assert 'private transcript' not in log and 'private response' not in log
