"""Run: python -m motion_tracking.track --camera 0 (or --video path.mp4)."""
import argparse
from contextlib import ExitStack
import json
from pathlib import Path
import socket
import time
import uuid
import numpy as np
from .download_models import MODEL_DIR
from .retarget import Config, Retargeter, JOINTS


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    source = p.add_mutually_exclusive_group()
    source.add_argument('--camera', type=int, default=0)
    source.add_argument('--video', type=Path)
    p.add_argument('--port', type=int, default=5055)
    p.add_argument('--model-dir', type=Path, default=MODEL_DIR)
    p.add_argument('--config', type=Path, help='JSON overrides for retarget.Config')
    p.add_argument('--record', type=Path, help='Optional JSONL of SENT targets (not Unity applied poses)')
    p.add_argument('--no-preview', action='store_true')
    p.add_argument('--head-only', action='store_true', help='Track head and expressions only; send neutral body/arms')
    p.add_argument('--max-frames', type=int, default=0)
    return p


def main():
    args = parser().parse_args()
    if not 1 <= args.port <= 65535 or args.max_frames < 0:
        raise SystemExit('Invalid port or max-frames')
    config = Config(**json.loads(args.config.read_text())) if args.config else Config()
    if args.head_only:
        config.head_reference = 'camera'
    for name in (('face_landmarker.task',) if args.head_only else ('face_landmarker.task','pose_landmarker_lite.task')):
        if not (args.model_dir/name).is_file():
            raise SystemExit('Missing task models. Run: python -m motion_tracking.download_models')
    import cv2
    import mediapipe as mp
    from mediapipe.tasks.python import vision
    mapper = Retargeter(config)
    session = str(uuid.uuid4())
    with ExitStack() as stack:
        capture = cv2.VideoCapture(str(args.video) if args.video else args.camera)
        stack.callback(capture.release)
        stack.callback(cv2.destroyAllWindows)
        if not capture.isOpened():
            raise SystemExit('Cannot open input. Check path/camera index and macOS camera permission for your terminal.')
        fps = capture.get(cv2.CAP_PROP_FPS)
        fps = fps if np.isfinite(fps) and fps > 0 else 30.0
        base = mp.tasks.BaseOptions
        face = stack.enter_context(vision.FaceLandmarker.create_from_options(vision.FaceLandmarkerOptions(
            base_options=base(model_asset_path=str(args.model_dir/'face_landmarker.task')),
            running_mode=vision.RunningMode.VIDEO, num_faces=1,
            output_face_blendshapes=True, output_facial_transformation_matrixes=True)))
        pose = None if args.head_only else stack.enter_context(vision.PoseLandmarker.create_from_options(vision.PoseLandmarkerOptions(
            base_options=base(model_asset_path=str(args.model_dir/'pose_landmarker_lite.task')),
            running_mode=vision.RunningMode.VIDEO, num_poses=1)))
        sock = stack.enter_context(socket.socket(socket.AF_INET, socket.SOCK_DGRAM))
        recording = None
        if args.record:
            args.record.parent.mkdir(parents=True, exist_ok=True)
            recording = stack.enter_context(args.record.open('x'))
        start, previous_ms, sequence = time.monotonic(), -1, 0
        test_blink_until = -1.0
        print('Tracking to 127.0.0.1:%d. Face camera; hold your head neutral briefly. C: recalibrate; B: test eyelids; Q: quit.' % args.port)
        while not args.max_frames or sequence < args.max_frames:
            ok, frame = capture.read()
            if not ok:
                if not args.video:
                    raise RuntimeError('Camera stopped delivering frames')
                break
            source_ms = capture.get(cv2.CAP_PROP_POS_MSEC) if args.video else (time.monotonic()-start)*1000
            if not np.isfinite(source_ms) or (args.video and sequence > 0 and source_ms <= previous_ms):
                source_ms = sequence*1000/fps
            timestamp_ms = max(previous_ms+1, int(source_ms))
            previous_ms = timestamp_ms
            if args.video:
                time.sleep(max(0, start+timestamp_ms/1000-time.monotonic()))
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image = mp.Image(image_format=mp.ImageFormat.SRGB, data=np.ascontiguousarray(rgb))
            # Blocking VIDEO calls share exactly the same frame/timestamp, also for a webcam.
            f = face.detect_for_video(image, timestamp_ms)
            p = pose.detect_for_video(image, timestamp_ms) if pose else None
            landmarks = p.pose_world_landmarks[0] if p and p.pose_world_landmarks else None
            points = [[v.x,v.y,v.z] for v in landmarks] if landmarks else None
            visibility = [min(v.visibility or 0, v.presence or 0) for v in landmarks] if landmarks else None
            matrix = f.facial_transformation_matrixes[0] if len(f.facial_transformation_matrixes) else None
            blends = {v.category_name:v.score for v in f.face_blendshapes[0]} if f.face_blendshapes else None
            angles, valid = mapper.update(timestamp_ms/1000, points, visibility, matrix, blends)
            if args.head_only:
                angles[:6] = 0
                valid[:6] = False
            # A manual diagnostic uses the same packet/controller path as detected blinks.
            if time.monotonic() < test_blink_until:
                angles[12] = 45.0
                valid[12] = True
            packet = dict(version=1, session=session, sequence=sequence, timestamp_ms=timestamp_ms,
                          angles=angles.tolist(), valid=valid.tolist())
            encoded = json.dumps(packet, allow_nan=False, separators=(',',':'))
            sock.sendto(encoded.encode(), ('127.0.0.1', args.port))
            if recording:
                recording.write(encoded+'\n')
            if not args.no_preview:
                preview = cv2.flip(frame,1)
                for label, values in (('pose',p.pose_landmarks[0] if p and p.pose_landmarks else []),
                                      ('face',f.face_landmarks[0] if f.face_landmarks else [])):
                    for v in values:
                        cv2.circle(preview,(int((1-v.x)*frame.shape[1]),int(v.y*frame.shape[0])),
                                   2 if label=='pose' else 1,(0,255,0),-1)
                lines = [f"Head: {mapper.head_status}",
                         f"Sent head degrees: pitch {angles[6]:+.1f} yaw {angles[7]:+.1f} roll {angles[8]:+.1f}"]
                blink = mapper.blink_scores
                blink_label = f"L {blink[0]:.2f} R {blink[1]:.2f}" if blink is not None else "not detected"
                lines.append(f"Blink {blink_label} -> lid {angles[12]:.1f} deg | B: test")
                if not args.head_only:
                    lines.append("Body: " + ("tracking" if valid[:2].all() else "not tracked / calibrating"))
                for index, line in enumerate(lines):
                    cv2.putText(preview,line,(15,30+25*index),cv2.FONT_HERSHEY_SIMPLEX,0.5,(0,255,255),1)
                cv2.imshow('CommU tracking | C calibrate | Q quit',preview)
                key = cv2.waitKey(1) & 0xff
                if key in (ord('q'),27): break
                if key == ord('c'): mapper.reset()
                if key == ord('b'): test_blink_until = time.monotonic() + 1.0
            sequence += 1
        print(f'Finished; {sequence} frames processed. Joint order: {", ".join(JOINTS)}')

if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        pass
