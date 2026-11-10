"""Masked joint regression metrics, expressed in degrees against pose pseudo-labels."""
import numpy as np

from .common import JOINT_LIMITS, JOINT_NAMES
from .normalization import affine_parameters

TOLERANCES = (1, 2, 5, 10)


class AngleMetrics:
    def __init__(self, normalization):
        self.mean, self.std = affine_parameters(normalization)
        self.count = 0
        self.absolute = np.zeros(14)
        self.squared = np.zeros(14)
        self.correct = np.zeros((14, len(TOLERANCES)), dtype=np.int64)
        self.violations = 0

    def update(self, prediction, target, mask):
        valid = mask.detach().cpu().numpy().astype(bool)
        prediction = prediction.detach().float().cpu().numpy()[valid]
        target = target.detach().float().cpu().numpy()[valid]
        error = np.abs((prediction - target) * self.std)
        self.absolute += error.sum(0)
        self.squared += np.square(error).sum(0)
        self.correct += (error[:, :, None] <= np.asarray(TOLERANCES)).sum(0)
        degrees = prediction * self.std + self.mean
        limits = np.asarray(JOINT_LIMITS)
        self.violations += int(((degrees < limits[:, 0]) | (degrees > limits[:, 1])).sum())
        self.count += len(error)

    def compute(self):
        if self.count == 0:
            raise ValueError("No valid joint frames to evaluate")
        matrix = self.correct * (100.0 / self.count)
        per_joint_rmse = np.sqrt(self.squared / self.count)
        limits = np.asarray(JOINT_LIMITS)
        percent = per_joint_rmse / (limits[:, 1] - limits[:, 0]) * 100
        return {"frames": self.count, "mae_degrees": float(self.absolute.mean() / self.count),
                "rmse_degrees": float(np.sqrt(self.squared.mean() / self.count)),
                "per_joint_mae_degrees": dict(zip(JOINT_NAMES, (self.absolute / self.count).tolist())),
                "per_joint_rmse_degrees": dict(zip(JOINT_NAMES, per_joint_rmse.tolist())),
                "per_joint_rmse_percent_of_range": dict(zip(JOINT_NAMES, percent.tolist())),
                "tolerances_degrees": list(TOLERANCES), "accuracy_matrix_percent": matrix.tolist(),
                "accuracy_percent": {str(t): float(matrix[:, i].mean()) for i, t in enumerate(TOLERANCES)},
                "joint_limit_violation_fraction": self.violations / (self.count * 14)}
