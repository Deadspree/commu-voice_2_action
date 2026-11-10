"""A persistent Google Gemini Live session; credentials never leave the backend."""
from contextlib import asynccontextmanager
import os

from google import genai
from google.genai import types


class GeminiLiveProvider:
    def __init__(self, cfg, client=None):
        self.cfg = cfg
        key = os.getenv('GEMINI_API_KEY')
        if client is None and not key:
            raise ValueError('Set GEMINI_API_KEY in the existing root .env before starting Gemini Live')
        self.client = client or genai.Client(api_key=key, http_options={'api_version': 'v1beta'})

    @asynccontextmanager
    async def connect(self, history=()):
        config = types.LiveConnectConfig(
            response_modalities=['AUDIO'],
            system_instruction=self.cfg['instructions'],
            max_output_tokens=self.cfg['max_output_tokens'],
            speech_config=types.SpeechConfig(voice_config=types.VoiceConfig(
                prebuilt_voice_config=types.PrebuiltVoiceConfig(voice_name=self.cfg['voice']))),
            input_audio_transcription=types.AudioTranscriptionConfig(),
            output_audio_transcription=types.AudioTranscriptionConfig(),
            # Explicit activity boundaries match push-to-talk. Never send audio_stream_end
            # with manual VAD; releasing Space sends ActivityEnd instead.
            realtime_input_config=types.RealtimeInputConfig(
                automatic_activity_detection=types.AutomaticActivityDetection(disabled=True)),
            history_config=types.HistoryConfig(initial_history_in_client_content=bool(history)),
        )
        async with self.client.aio.live.connect(model=self.cfg['model'], config=config) as session:
            # connect() waits for setup_complete before returning. Seed only fully played turns.
            if history:
                turns = [types.Content(role='model' if item['role'] == 'assistant' else 'user',
                                       parts=[types.Part(text=item['content'])]) for item in history]
                await session.send_client_content(turns=turns, turn_complete=False)
            yield session

    async def close(self):
        await self.client.aio.aclose()
        self.client.close()
