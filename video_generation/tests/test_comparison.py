import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys

from PIL import Image
import pytest

from video_generation import comparison


def test_render_validation_checks_frames_and_csv_fingerprint(tmp_path):
    csv = tmp_path / 'joints.csv'
    csv.write_text('test pose data')
    folder = tmp_path / 'frames'
    folder.mkdir()
    metadata = dict(fps=30, frame_count=1, csv=str(csv), csv_sha256=hashlib.sha256(csv.read_bytes()).hexdigest())
    (folder / 'render.json').write_text(json.dumps(metadata))
    with pytest.raises(ValueError, match='Missing Unity'): comparison.validate_render(folder, csv)
    Image.new('RGB', (64, 36)).save(folder / '000000.png')
    assert comparison.validate_render(folder, csv)['frame_count'] == 1
    csv.write_text('different data')
    with pytest.raises(ValueError, match='CSV changed'): comparison.validate_render(folder, csv)


def test_input_manifest_selects_original_and_honors_source_offset(tmp_path):
    original = tmp_path / 'original video.mp4'
    original.touch()
    (tmp_path / 'robot_with_audio.mp4').touch()
    (tmp_path / 'robot_with_audio_joints.csv').touch()
    manifest = dict(status='completed', inputs=dict(video=dict(path=str(original))), timeline=dict(source_start_seconds=1.25))
    (tmp_path / 'manifest.json').write_text(json.dumps(manifest))
    assert comparison.inputs(tmp_path)[0] == original
    assert comparison.inputs(tmp_path)[-1] == 1.25
    manifest['status'] = 'pending'
    (tmp_path / 'manifest.json').write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match='completed'): comparison.inputs(tmp_path)


def test_composition_keeps_30fps_timing_and_one_audio_track(tmp_path, monkeypatch):
    ffmpeg, ffprobe = shutil.which('ffmpeg'), shutil.which('ffprobe')
    if not ffmpeg or not ffprobe: pytest.skip('FFmpeg/ffprobe required')
    run = tmp_path / 'run with spaces'
    run.mkdir()
    original = run / 'original.mp4'
    generated = run / 'robot_with_audio.mp4'
    subprocess.run([ffmpeg, '-v', 'error', '-y', '-f', 'lavfi', '-i', 'color=red:s=64x36:r=30000/1001:d=0.3',
                    '-c:v', 'libx264', str(original)], check=True)
    subprocess.run([ffmpeg, '-v', 'error', '-y', '-f', 'lavfi', '-i', 'color=blue:s=64x36:r=30:d=0.3',
                    '-f', 'lavfi', '-i', 'sine=frequency=440:duration=0.3', '-c:v', 'libx264', '-c:a', 'aac',
                    '-shortest', str(generated)], check=True)
    csv = run / 'robot_with_audio_joints.csv'
    csv.write_text('test pose data')
    (run / 'manifest.json').write_text(json.dumps(dict(status='completed', inputs=dict(video=dict(path=str(original))))))
    frames = tmp_path / 'unity frames'
    frames.mkdir()
    for index in range(9): Image.new('RGB', (64, 36), '#24a143').save(frames / f'{index:06d}.png')
    (frames / 'render.json').write_text(json.dumps(dict(fps=30, frame_count=9, csv=str(csv),
        csv_sha256=hashlib.sha256(csv.read_bytes()).hexdigest())))
    output = tmp_path / 'output'
    monkeypatch.setattr(sys, 'argv', ['comparison', '--run-dir', str(run), '--unity-frames', str(frames), '--output-dir', str(output)])
    comparison.main()
    result = json.loads(subprocess.check_output([ffprobe, '-v', 'error', '-show_streams', '-of', 'json', str(output / 'comparison.mp4')]))
    video = [s for s in result['streams'] if s['codec_type'] == 'video'][0]
    assert video['r_frame_rate'] == '30/1' and int(video['nb_frames']) == 9
    assert abs(float(video['duration']) - .3) < .001
    assert video['width'] == 1920 and video['height'] == 424
    assert len([s for s in result['streams'] if s['codec_type'] == 'audio']) == 1
