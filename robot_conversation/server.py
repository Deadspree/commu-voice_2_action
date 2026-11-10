"""Gemini Live + trained motion backend: python -m robot_conversation.server."""
import argparse
import asyncio
from pathlib import Path

from .pipeline import LiveConversation
from .protocol import MAX_MESSAGE_BYTES


async def handle_connection(connection, provider, predictor, cfg, demo=None):
    await LiveConversation(provider, predictor, cfg, demo).run(connection)


async def serve(provider, predictor, cfg, demo=None):
    from websockets.asyncio.server import serve as websocket_serve
    async with websocket_serve(lambda ws: handle_connection(ws, provider, predictor, cfg, demo),
                              cfg['server']['host'], cfg['server']['port'],
                              max_size=MAX_MESSAGE_BYTES, max_queue=4, ping_interval=20,
                              origins=[None, 'http://localhost', 'http://127.0.0.1']):
        print(f"CommU listening on ws://{cfg['server']['host']}:{cfg['server']['port']}", flush=True)
        print('Offline audio preview' if demo is not None else f"Gemini model: {cfg['gemini']['model']} (connect Unity to open a Live session)", flush=True)
        await asyncio.Future()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', default='robot_conversation/configs/conversation.yaml')
    parser.add_argument('--device')
    parser.add_argument('--demo-audio', help='Local audio-file preview without Gemini calls')
    parser.add_argument('--check-gemini', action='store_true', help='Check Live session setup without loading the motion model or sending audio')
    args = parser.parse_args()
    from dotenv import load_dotenv
    from .settings import load_settings
    from .motion.predictor import MotionPredictor
    from .speech.gemini_live import GeminiLiveProvider
    load_dotenv(Path(__file__).resolve().parents[1] / '.env')
    try:
        cfg = load_settings(args.config)
        provider = None if args.demo_audio and not args.check_gemini else GeminiLiveProvider(cfg['gemini'])
    except ValueError as error:
        parser.error(str(error))
    if args.check_gemini:
        async def check():
            try:
                async with asyncio.timeout(cfg['gemini']['connect_timeout_seconds']):
                    async with provider.connect():
                        print(f"Gemini Live setup succeeded: {cfg['gemini']['model']}")
            finally:
                await provider.close()
        try:
            asyncio.run(check())
        except Exception:
            parser.exit(1, 'Gemini setup failed. Check GEMINI_API_KEY, model access, quota and network.\n')
        return
    from speech_to_motion.runtime import MotionRuntime
    from speech_to_motion.audio import decode_audio
    print('Loading speech-to-motion checkpoint...', flush=True)
    runtime = MotionRuntime(cfg['motion']['checkpoint'], args.device or cfg['motion']['device'], cfg['motion']['cpu_threads'])
    predictor = MotionPredictor(runtime)
    demo = decode_audio(args.demo_audio, 24000) if args.demo_audio else None
    if demo is not None and not .1 <= len(demo) / 24000 <= cfg['session']['max_reply_seconds']:
        predictor.close()
        parser.error('Demo audio must be 0.1 seconds to max_reply_seconds long')
    async def run():
        try:
            await serve(provider, predictor, cfg, demo)
        finally:
            if provider:
                await provider.close()
    try:
        asyncio.run(run())
    except KeyboardInterrupt:
        pass
    finally:
        predictor.close()


if __name__ == '__main__':
    main()
