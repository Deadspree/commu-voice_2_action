"""Pure, testable mapping into the existing CommU joint order (degrees).

Pose world points use x right/y down/z away. Convert to x right/y up/z
near before combining with FaceLandmarker's canonical-face rotation.
The arm mapping is an expressive approximation of a two-axis robot arm.
"""
from dataclasses import dataclass, field
import math
import numpy as np

JOINTS = ('body_yaw', 'body_pitch', 'right_arm_pitch', 'right_arm_roll',
          'left_arm_pitch', 'left_arm_roll', 'face_pitch', 'face_yaw',
          'face_roll', 'eye_pitch', 'right_eye_yaw', 'left_eye_yaw', 'eyelid', 'mouth')
LIMITS = np.array([[-20,20],[-90,90],[-180,180],[-30,15],[-180,180],[-15,30],
                   [-12,12],[-10,10],[-20,20],[-30,30],[-30,30],[-30,30],[0,45],[-20,20]], dtype=float)
NEUTRAL = np.zeros(14)

def unit(v):
    length = np.linalg.norm(v)
    if not np.isfinite(length) or length < 1e-6:
        raise ValueError('Degenerate landmark geometry')
    return v / length

def rotation_angles(r):
    """XYZ intrinsic components for Rz(roll) Ry(yaw) Rx(pitch)."""
    return np.degrees([math.atan2(r[2,1], r[2,2]),
                       math.asin(float(np.clip(-r[2,0], -1, 1))),
                       math.atan2(r[1,0], r[0,0])])

def torso_frame(points):
    left, right = points[11], points[12]
    up = unit((left + right - points[23] - points[24]) / 2)
    x = unit(left - right)
    z = unit(np.cross(x, up))
    y = unit(np.cross(z, x))
    return np.column_stack((x, y, z))

@dataclass
class Config:
    head_reference: str = "camera"  # "torso" requires visible shoulders and hips.
    calibration_frames: int = 20
    visibility: float = 0.6
    hold_seconds: float = 0.35
    smoothing_seconds: float = 0.10
    expression_smoothing_seconds: float = 0.035
    blink_open_score: float = 0.10
    blink_closed_score: float = 0.55
    blink_hold_seconds: float = 0.12
    # Human degrees -> robot degrees; eye, lid, and mouth expressions use joint ranges.
    gains: list = field(default_factory=lambda: [0.5,0.7,1,1,1,1,0.4,0.3,0.7,1,1,1,1,1])
    offsets: list = field(default_factory=lambda: [0.0] * 14)

    def __post_init__(self):
        if not (0 <= self.blink_open_score < self.blink_closed_score <= 1):
            raise ValueError("Blink thresholds must satisfy 0 <= open < closed <= 1")
        if self.head_reference not in ("camera", "torso"):
            raise ValueError("head_reference must be camera or torso")
        if self.calibration_frames < 1 or not 0 <= self.visibility <= 1:
            raise ValueError('Invalid calibration_frames or visibility')
        if any(not np.isfinite(v) or v < 0 for v in (self.hold_seconds, self.smoothing_seconds, self.expression_smoothing_seconds, self.blink_hold_seconds)):
            raise ValueError('Filter times must be finite and nonnegative')
        for values in (self.gains, self.offsets):
            if len(values) != 14 or not np.isfinite(values).all():
                raise ValueError('gains and offsets must each contain 14 finite numbers')

class Retargeter:
    def __init__(self, config=None):
        self.config = config or Config()
        self.reset()

    def reset(self):
        self.baselines = {}
        self.samples = {}
        self.current = NEUTRAL.copy()
        self.target = NEUTRAL.copy()
        self.last_seen = np.full(14, -np.inf)
        self.last_time = None
        self.head_status = "Waiting for face"
        self.blink_scores = None
        self.blink_until = -np.inf

    def calibrated(self, name):
        return name in self.baselines

    def _relative(self, name, values):
        if name not in self.baselines:
            samples = self.samples.setdefault(name, [])
            samples.append(np.asarray(values).copy())
            if len(samples) < self.config.calibration_frames:
                return None
            # Circular mean avoids +/-180-degree neutral discontinuities.
            radians = np.radians(samples)
            self.baselines[name] = np.degrees(np.arctan2(np.sin(radians).mean(0), np.cos(radians).mean(0)))
        return (values - self.baselines[name] + 180) % 360 - 180

    def update(self, timestamp, points=None, visibility=None, face_matrix=None, blends=None):
        if not math.isfinite(timestamp) or (self.last_time is not None and timestamp <= self.last_time):
            raise ValueError('Timestamps must be finite and strictly increasing')
        dt = 1/30 if self.last_time is None else timestamp-self.last_time
        self.last_time = timestamp
        raw = np.zeros(14)
        valid = np.zeros(14, dtype=bool)
        torso = None
        p = None
        if points is not None:
            p = np.asarray(points, dtype=float)
            vis = np.asarray(visibility, dtype=float) if visibility is not None else np.ones(33)
            if p.shape != (33,3) or vis.shape != (33,):
                raise ValueError('Expected 33 xyz points and 33 visibility scores')
            p = p * np.array([1,-1,-1])
            good = np.isfinite(p).all(1) & np.isfinite(vis) & (vis >= self.config.visibility)
            if good[[11,12,23,24]].all():
                try:
                    torso = torso_frame(p)
                    angles = self._relative('body', rotation_angles(torso))
                    if angles is not None:
                        raw[:2] = angles[[1,0]]
                        valid[:2] = True
                    for shoulder, elbow, index, side in ((12,14,2,-1),(11,13,4,1)):
                        if good[[shoulder,elbow]].all():
                            arm = unit(torso.T @ (p[elbow]-p[shoulder]))
                            raw[index] = -math.degrees(math.atan2(math.hypot(arm[0],arm[2]), -arm[1]))
                            raw[index+1] = side * 30 * max(0, side * arm[0])
                            valid[index:index+2] = True
                except ValueError:
                    torso = None
        self.head_status = "Face lost" if self.calibrated('head') else "Waiting for face"
        if face_matrix is not None and (self.config.head_reference == 'camera' or torso is not None):
            matrix = np.asarray(face_matrix, dtype=float)
            if matrix.shape == (4,4) and np.isfinite(matrix).all():
                # Remove any scale from the estimated rigid transform.
                u, _, vt = np.linalg.svd(matrix[:3,:3])
                face = u @ np.diag([1,1,np.linalg.det(u @ vt)]) @ vt
                reference = face if self.config.head_reference == 'camera' else torso.T @ face
                relative = self._relative('head', rotation_angles(reference))
                self.head_status = f"Calibrating {len(self.samples.get('head', []))}/{self.config.calibration_frames}"
                if relative is not None:
                    raw[6:9] = relative
                    valid[6:9] = True
                    self.head_status = "Tracking"
        elif face_matrix is not None:
            self.head_status = "Head blocked: torso not visible"
        self.blink_scores = None
        if blends:
            def score(name):
                return float(np.clip(blends[name],0,1))
            groups = [((12,), ('eyeBlinkLeft','eyeBlinkRight')),
                      ((13,), ('jawOpen',)),
                      ((9,), ('eyeLookUpLeft','eyeLookUpRight','eyeLookDownLeft','eyeLookDownRight')),
                      ((10,11), ('eyeLookInRight','eyeLookOutRight','eyeLookInLeft','eyeLookOutLeft'))]
            for indices, keys in groups:
                if not all(k in blends and math.isfinite(blends[k]) for k in keys):
                    continue
                if indices == (12,):
                    self.blink_scores = (score(keys[0]), score(keys[1]))
                    mean = sum(self.blink_scores) / 2
                    closure = np.clip((mean - self.config.blink_open_score) /
                                      (self.config.blink_closed_score - self.config.blink_open_score), 0, 1)
                    if mean >= self.config.blink_closed_score:
                        self.blink_until = timestamp + self.config.blink_hold_seconds
                    if timestamp < self.blink_until:
                        closure = 1.0
                    raw[12] = 45 * closure
                elif indices == (13,): raw[13] = 20 * score('jawOpen')
                elif indices == (9,): raw[9] = 20 * (score(keys[2])+score(keys[3])-score(keys[0])-score(keys[1]))/2
                else:
                    raw[10] = 30*(score('eyeLookInRight')-score('eyeLookOutRight'))
                    raw[11] = 30*(score('eyeLookOutLeft')-score('eyeLookInLeft'))
                valid[list(indices)] = True
        desired = raw*np.asarray(self.config.gains)+np.asarray(self.config.offsets)
        self.target[valid] = np.clip(desired, LIMITS[:,0], LIMITS[:,1])[valid]
        self.last_seen[valid] = timestamp
        expired = timestamp-self.last_seen > self.config.hold_seconds
        self.target[expired] = np.clip(np.asarray(self.config.offsets), LIMITS[:,0], LIMITS[:,1])[expired]
        tau = np.full(14, self.config.smoothing_seconds)
        tau[9:] = self.config.expression_smoothing_seconds
        alpha = 1-np.exp(-dt / np.maximum(tau, 1e-8))
        self.current += alpha*(self.target-self.current)
        return self.current.copy(), valid
