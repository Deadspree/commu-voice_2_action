"""Physical-limit scaling matching pose_estimation, with legacy checkpoint support."""
import numpy as np

from .common import JOINT_LIMITS, JOINT_NAMES


def joint_limit_normalization(training_mean):
    limits = np.asarray(JOINT_LIMITS, dtype=float)
    return {"method": "joint_limits", "offset": limits.mean(1).tolist(),
            "scale": ((limits[:, 1] - limits[:, 0]) / 2).tolist(),
            "joint_limits": limits.tolist(), "joint_names": JOINT_NAMES,
            "angle_units": "degrees", "training_mean": list(training_mean)}


def affine_parameters(normalization):
    method = normalization.get("method", "standardization")
    if method == "joint_limits":
        offset, scale = normalization["offset"], normalization["scale"]
    elif method == "standardization":
        offset, scale = normalization["mean"], normalization["std"]
    else:
        raise ValueError(f"Unknown normalization: {method}")
    offset, scale = np.asarray(offset, dtype=float), np.asarray(scale, dtype=float)
    if offset.shape != (14,) or scale.shape != (14,) or not np.isfinite(offset).all() or not np.isfinite(scale).all() or np.any(scale <= 0):
        raise ValueError("Invalid 14-joint normalization offset/scale")
    return offset, scale


def normalize_angles(angles, normalization):
    offset, scale = affine_parameters(normalization)
    return (angles - offset) / scale


def denormalize_angles(angles, normalization):
    offset, scale = affine_parameters(normalization)
    return angles * scale + offset


def training_mean(normalization):
    if "training_mean" in normalization:
        return np.asarray(normalization["training_mean"])
    return np.asarray(normalization["mean"])
