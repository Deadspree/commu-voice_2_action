import asyncio
from concurrent.futures import ThreadPoolExecutor


class MotionPredictor:
    def __init__(self, runtime):
        self.runtime = runtime
        # Cancelling an asyncio task cannot stop an executing GPU kernel.
        # One worker guarantees no overlapping model execution after cancellation.
        self.executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="commu-motion")

    async def predict(self, audio, sample_rate):
        return await asyncio.get_running_loop().run_in_executor(
            self.executor, self.runtime.predict, audio, sample_rate)

    def close(self):
        self.executor.shutdown(wait=True, cancel_futures=True)
