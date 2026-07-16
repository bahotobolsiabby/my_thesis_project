"""Safe damped-least-squares IK for Franka Panda.

Position-dominant IK plus optional nullspace wrist posture. The posture term
helps the gripper stay usable for pick/place without fighting the XYZ target.
"""
from __future__ import annotations

import numpy as np
from panda_config import PandaGeometry, IKConfig


def _dh_transform(a: float, alpha: float, d: float, theta: float) -> np.ndarray:
    ca, sa = np.cos(alpha), np.sin(alpha)
    ct, st = np.cos(theta), np.sin(theta)
    return np.array([
        [ct,       -st,       0.0,      a],
        [st * ca,   ct * ca, -sa, -sa * d],
        [st * sa,   ct * sa,  ca,  ca * d],
        [0.0,       0.0,      0.0,      1.0],
    ], dtype=float)


def clamp_joint_step(q_new: np.ndarray, q_prev: np.ndarray, max_delta: float = 0.03) -> np.ndarray:
    delta = np.clip(q_new - q_prev, -max_delta, max_delta)
    return q_prev + delta


def forward_kinematics_full(q: np.ndarray, geo: PandaGeometry) -> tuple[np.ndarray, np.ndarray]:
    T = np.eye(4)
    for i in range(7):
        T = T @ _dh_transform(geo.a[i], geo.alpha[i], geo.d[i], q[i])
    T_flange = np.eye(4)
    T_flange[:3, 3] = geo.flange_offset
    T = T @ T_flange
    return T[:3, 3].copy(), T[:3, :3].copy()


def forward_kinematics(q: np.ndarray, geo: PandaGeometry) -> np.ndarray:
    pos, _ = forward_kinematics_full(q, geo)
    return pos


def _numeric_jacobian(q: np.ndarray, geo: PandaGeometry, eps: float = 1e-5) -> np.ndarray:
    jac = np.zeros((3, 7), dtype=float)
    for i in range(7):
        dq = np.zeros(7, dtype=float)
        dq[i] = eps
        p_plus = forward_kinematics(q + dq, geo)
        p_minus = forward_kinematics(q - dq, geo)
        jac[:, i] = (p_plus - p_minus) / (2.0 * eps)
    return jac


def _joint_limit_avoidance_gradient(q: np.ndarray, joint_limits: np.ndarray, buffer_fraction: float = 0.20) -> np.ndarray:
    lo = joint_limits[:, 0]
    hi = joint_limits[:, 1]
    mid = (lo + hi) / 2.0
    half = (hi - lo) / 2.0

    normalized = (q - mid) / half
    mag = np.abs(normalized)
    start = 1.0 - buffer_fraction

    ramp = np.where(mag > start, (mag - start) / buffer_fraction, 0.0)
    return -np.sign(normalized) * (ramp ** 3)


def solve_panda_ik(
    ee_target: np.ndarray,
    q_init: np.ndarray,
    joint_limits: np.ndarray,
    neutral_q: np.ndarray,
    geo: PandaGeometry,
    cfg: IKConfig,
    *,
    posture_target: np.ndarray | None = None,
    posture_weight: float = 0.0,
    posture_mask: np.ndarray | None = None,
) -> np.ndarray:
    q = q_init.astype(float).copy()
    best_q = q.copy()
    best_err = float(np.linalg.norm(ee_target - forward_kinematics(q, geo)))
    max_internal_delta = 0.08

    for _ in range(cfg.max_iter):
        ee_pos = forward_kinematics(q, geo)
        err_vec = ee_target - ee_pos
        err = float(np.linalg.norm(err_vec))

        if err < best_err:
            best_err = err
            best_q = q.copy()

        if err < cfg.tol:
            break

        jac = _numeric_jacobian(q, geo)
        jj_t = jac @ jac.T
        j_pinv = jac.T @ np.linalg.solve(jj_t + cfg.damping ** 2 * np.eye(3), np.eye(3))

        dq_task = j_pinv @ err_vec

        dq_posture = np.zeros(7, dtype=float)
        if posture_target is not None and posture_weight > 0.0:
            mask = np.ones(7, dtype=float) if posture_mask is None else posture_mask.astype(float)
            posture_err = mask * (posture_target - q)
            nullspace = np.eye(7) - j_pinv @ jac
            dq_posture = nullspace @ (posture_weight * posture_err)

        neutral_bias = cfg.neutral_weight * (neutral_q - q)
        limit_bias = cfg.limit_avoidance_weight * _joint_limit_avoidance_gradient(q, joint_limits)

        dq = cfg.step_size * dq_task + dq_posture + neutral_bias + limit_bias
        dq = np.clip(dq, -max_internal_delta, max_internal_delta)

        q_next = np.clip(q + dq, joint_limits[:, 0], joint_limits[:, 1])
        if not np.all(np.isfinite(q_next)):
            break
        q = q_next

    final_err = float(np.linalg.norm(ee_target - forward_kinematics(q, geo)))
    if final_err < best_err:
        best_q = q.copy()

    return best_q


def joints_at_limit(q: np.ndarray, joint_limits: np.ndarray, margin: float = 0.01) -> list[int]:
    lo = joint_limits[:, 0]
    hi = joint_limits[:, 1]
    return [i for i in range(len(q)) if q[i] <= lo[i] + margin or q[i] >= hi[i] - margin]


def ik_is_stuck(ee_target: np.ndarray, q_result: np.ndarray, joint_limits: np.ndarray, geo: PandaGeometry, cfg: IKConfig) -> bool:
    err = float(np.linalg.norm(ee_target - forward_kinematics(q_result, geo)))
    return bool(err > 0.025 and len(joints_at_limit(q_result, joint_limits)) > 0)