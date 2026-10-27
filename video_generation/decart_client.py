"""Small REST adapter. POST is deliberately never retried automatically."""
import os
import time
from pathlib import Path
from urllib.parse import quote

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry


class DecartClient:
    BASE = "https://api.decart.ai/v1/jobs"

    def __init__(self, api_key):
        self.session = requests.Session()
        self.session.headers["x-api-key"] = api_key
        # Only read operations are safe to retry without creating another job.
        self.session.mount("https://", HTTPAdapter(max_retries=Retry(
            total=3, backoff_factor=1, status_forcelist=[429, 500, 502, 503, 504],
            allowed_methods={"GET"}, respect_retry_after_header=False,
        )))

    def close(self):
        self.session.close()

    @staticmethod
    def check(response):
        if not response.ok:
            # Response bodies may echo submitted content; don't log them.
            raise RuntimeError(f"Decart HTTP {response.status_code}; inspect the job in the Decart dashboard.")

    def submit(self, video, reference, prompt, config):
        with Path(video).open("rb") as vf, Path(reference).open("rb") as rf:
            response = self.session.post(
                f"{self.BASE}/{quote(config['model'], safe='')}",
                files={"data": ("input.mp4", vf, "video/mp4"),
                       "reference_image": (Path(reference).name, rf)},
                data={"prompt": prompt, "resolution": config["resolution"]},
                timeout=(15, 300), allow_redirects=False,
            )
        with response:
            self.check(response)
            job_id = response.json().get("job_id")
            if not isinstance(job_id, str) or not job_id:
                raise RuntimeError("Submission returned no job ID; check the dashboard before retrying.")
            return job_id

    def wait(self, job_id, config, on_status):
        deadline = time.monotonic() + config["poll_timeout_seconds"]
        while time.monotonic() < deadline:
            with self.session.get(f"{self.BASE}/{quote(job_id, safe='')}", timeout=(15, 60)) as response:
                self.check(response)
                status = response.json().get("status")
            if status not in {"pending", "processing", "completed", "failed"}:
                raise RuntimeError(f"Unexpected Decart job status: {status!r}")
            on_status(status)
            if status == "completed":
                return
            if status == "failed":
                raise RuntimeError("Lucy generation failed. Check the Decart dashboard for details.")
            time.sleep(min(config["poll_interval_seconds"], max(0, deadline - time.monotonic())))
        raise TimeoutError("Polling deadline reached. Resume this run to check the same job again.")

    def download(self, job_id, output):
        output = Path(output)
        partial = output.with_suffix(".mp4.partial")
        try:
            with self.session.get(f"{self.BASE}/{quote(job_id, safe='')}/content",
                                  timeout=(15, 120), stream=True) as response:
                self.check(response)
                with partial.open("wb") as handle:
                    for block in response.iter_content(1024 * 1024):
                        handle.write(block)
            if partial.stat().st_size == 0:
                raise RuntimeError("Decart returned an empty video.")
            os.replace(partial, output)
        finally:
            partial.unlink(missing_ok=True)
