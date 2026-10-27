"""Download Google's versioned MediaPipe task bundles, atomically."""
import argparse
from pathlib import Path
import urllib.request

MODEL_DIR = Path(__file__).parent / 'models'
MODELS = {
    'face_landmarker.task': 'https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task',
    'pose_landmarker_lite.task': 'https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/1/pose_landmarker_lite.task',
}

def download(directory=MODEL_DIR):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    for name, url in MODELS.items():
        destination = directory / name
        if destination.is_file():
            print(f'Already available: {destination}')
            continue
        temporary = destination.with_suffix('.part')
        try:
            print(f'Downloading {name}')
            with urllib.request.urlopen(url, timeout=60) as response, temporary.open('wb') as out:
                while chunk := response.read(1024*1024):
                    out.write(chunk)
            temporary.replace(destination)
        finally:
            temporary.unlink(missing_ok=True)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model-dir', type=Path, default=MODEL_DIR)
    download(parser.parse_args().model_dir)
