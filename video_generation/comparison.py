"""Compose original/Lucy/Unity panels from a completed run and rendered Unity frames."""
from __future__ import annotations
import argparse
import hashlib
import json
import math
from pathlib import Path
import shutil
import subprocess

from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_EDITOR = '/Applications/Unity/Hub/Editor/6000.5.5f1/Unity.app/Contents/MacOS/Unity'


def probe_duration(path: Path, ffprobe: str) -> float:
    result = subprocess.run([ffprobe, '-v', 'error', '-select_streams', 'v:0', '-show_entries',
                             'stream=duration', '-of', 'json', str(path)], check=True, capture_output=True, text=True)
    duration = float(json.loads(result.stdout)['streams'][0]['duration'])
    if not math.isfinite(duration) or duration <= 0:
        raise ValueError(f'Invalid video duration: {path}')
    return duration


def inputs(run: Path):
    manifest = json.loads((run / 'manifest.json').read_text())
    if manifest['status'] != 'completed':
        raise ValueError('Select a completed Lucy run')
    original = Path(manifest['inputs']['video']['path'])
    if not original.is_file() and '/dataset/' in str(original):
        original = ROOT / ('dataset/' + str(original).split('/dataset/', 1)[1])
    generated, joints = run / 'robot_with_audio.mp4', run / 'robot_with_audio_joints.csv'
    for path in (original, generated, joints):
        if not path.is_file(): raise FileNotFoundError(path)
    offset = float(manifest.get('timeline', {}).get('source_start_seconds', 0))
    if not math.isfinite(offset) or offset < 0: raise ValueError('Invalid source offset')
    return original, generated, joints, offset


def validate_render(folder: Path, csv_path: Path):
    data = json.loads((folder / 'render.json').read_text())
    if data['fps'] != 30 or type(data['frame_count']) is not int or data['frame_count'] < 1:
        raise ValueError('Expected 30 FPS Unity render metadata')
    if Path(data['csv']).resolve() != csv_path.resolve():
        raise ValueError('Unity frames were rendered from a different CSV')
    if data['csv_sha256'] != hashlib.sha256(csv_path.read_bytes()).hexdigest():
        raise ValueError('CSV changed after Unity rendering; render again')
    for index in range(data['frame_count']):
        if not (folder / f'{index:06d}.png').is_file():
            raise ValueError(f'Missing Unity render frame {index}')
    return data


def make_header(path: Path, clip: str):
    image = Image.new('RGB', (1920, 64), '#101820')
    draw = ImageDraw.Draw(image)
    font_path = Path('/System/Library/Fonts/Supplemental/Arial.ttf')
    font = ImageFont.truetype(str(font_path), 23) if font_path.exists() else ImageFont.load_default(size=23)
    small = ImageFont.truetype(str(font_path), 14) if font_path.exists() else ImageFont.load_default(size=14)
    for i, label in enumerate(['Original video', 'Lucy 2.6 generated video', 'Unity robot: DINOv3 joint angles']):
        draw.text((i * 640 + 320, 7), label, fill='white', font=font, anchor='mt')
    draw.text((960, 42), f'{clip} | Shared timeline | Audio from generated video', fill='#aebdca', font=small, anchor='mt')
    image.save(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path)
    parser.add_argument('--unity-frames', type=Path)
    parser.add_argument('--render', action='store_true', help='Render Unity frames first; requires a closed Unity project')
    parser.add_argument('--unity-editor', default=DEFAULT_EDITOR)
    parser.add_argument('--unity-project', type=Path, default=ROOT / 'simulation/unity/commu')
    parser.add_argument('--ffmpeg', default=shutil.which('ffmpeg'))
    parser.add_argument('--ffprobe', default=shutil.which('ffprobe'))
    args = parser.parse_args()
    if not args.ffmpeg or not args.ffprobe: parser.error('FFmpeg and ffprobe must be on PATH')
    run = args.run_dir.resolve()
    original, generated, joints, offset = inputs(run)
    output = (args.output_dir or ROOT / 'outputs/video_comparison' / run.name).resolve()
    output.mkdir(parents=True, exist_ok=True)
    frames = (args.unity_frames or output / 'unity_frames').resolve()
    if args.render:
        subprocess.run([args.unity_editor, '-batchmode', '-projectPath', str(args.unity_project.resolve()),
                        '-executeMethod', 'VideoComparisonSetup.RenderBatch', '--comparison-run', str(run),
                        '--comparison-output', str(frames), '-logFile', str(output / 'unity-render.log')], check=True)
    metadata = validate_render(frames, joints)
    duration = min(metadata['frame_count'] / 30, probe_duration(generated, args.ffprobe),
                   probe_duration(original, args.ffprobe) - offset)
    frame_count = math.floor(duration * 30 + .001)
    if frame_count < 1: raise ValueError('No shared video timeline')
    header = output / 'header.png'
    make_header(header, run.name)
    filters = []
    for index in range(3):
        filters.append(f'[{index}:v]setpts=PTS-STARTPTS,fps=30,scale=640:360:force_original_aspect_ratio=decrease,'
                       f'pad=640:360:(ow-iw)/2:(oh-ih)/2:color=black,setsar=1[v{index}]')
    filters.extend(['[v0][v1][v2]hstack=inputs=3[panels]', '[3:v]fps=30,setpts=PTS-STARTPTS[header]',
                    '[header][panels]vstack=inputs=2,fps=30,format=yuv420p[video]'])
    result = output / 'comparison.mp4'
    partial = output / 'comparison.partial.mp4'
    subprocess.run([args.ffmpeg, '-hide_banner', '-loglevel', 'warning', '-y', '-ss', str(offset), '-i', str(original),
                    '-i', str(generated), '-framerate', '30', '-i', str(frames / '%06d.png'),
                    '-loop', '1', '-i', str(header), '-filter_complex', ';'.join(filters),
                    '-map', '[video]', '-map', '1:a:0?', '-frames:v', str(frame_count), '-t', str(frame_count / 30),
                    '-r', '30', '-c:v', 'libx264', '-crf', '18', '-preset', 'medium', '-c:a', 'aac', '-b:a', '160k',
                    '-movflags', '+faststart', str(partial)], check=True)
    rendered_duration = probe_duration(partial, args.ffprobe)
    if abs(rendered_duration - frame_count / 30) > .01:
        raise ValueError('Encoded comparison duration does not match the shared timeline')
    partial.replace(result)
    (output / 'comparison.json').write_text(json.dumps(dict(
        original=str(original), generated=str(generated), joints=str(joints), unity_frames=str(frames),
        frames=frame_count, fps=30, duration_seconds=frame_count / 30, original_offset_seconds=offset,
        audio_source='generated_video', csv_sha256=metadata['csv_sha256'], output=str(result)), indent=2) + '\n')
    print(result)


if __name__ == '__main__':
    main()
