"""Config for safe servo-style Panda teleoperation with hand landmarks."""
from __future__ import annotations

from dataclasses import dataclass, field
import numpy as np


@dataclass
class WorkspaceConfig:
    x_min: float = 0.20
    x_max: float = 0.62
    y_min: float = -0.35
    y_max: float = 0.35
    z_min: float = 0.16
    z_max: float = 0.78


@dataclass
class IKConfig:
    max_iter: int = 70
    tol: float = 1e-3
    damping: float = 0.06
    step_size: float = 0.65
    neutral_weight: float = 0.035
    limit_avoidance_weight: float = 0.045
    posture_weight: float = 0.12


@dataclass
class PandaGeometry:
    a: np.ndarray = field(default_factory=lambda: np.array(
        [0.0, 0.0, 0.0, 0.0825, -0.0825, 0.0, 0.088], dtype=float
    ))
    d: np.ndarray = field(default_factory=lambda: np.array(
        [0.333, 0.0, 0.316, 0.0, 0.384, 0.0, 0.0], dtype=float
    ))
    alpha: np.ndarray = field(default_factory=lambda: np.array(
        [0.0, -np.pi / 2.0, np.pi / 2.0, np.pi / 2.0,
         -np.pi / 2.0, np.pi / 2.0, np.pi / 2.0], dtype=float
    ))
    flange_offset: np.ndarray = field(default_factory=lambda: np.array([0.0, 0.0, 0.107], dtype=float))


JOINT_LIMITS = np.array([
    [-2.8973,  2.8973],
    [-1.7628,  1.7628],
    [-2.8973,  2.8973],
    [-3.0718, -0.0698],
    [-2.8973,  2.8973],
    [-0.0175,  3.7525],
    [-2.8973,  2.8973],
], dtype=float)

# Keep the user's preferred safe starting pose.
HOME_POSE = np.array([0.0, -0.4, 0.0, -1.9, -1.57, 1.57, 0.0], dtype=float)