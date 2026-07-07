"""
panda_position_ik.py

External position-based inverse kinematics helper for the Franka Panda arm.

Purpose:
- Convert a desired end-effector target position [x, y, z] into 7 Panda joint values.
- This is a lightweight numerical IK solver using damped least squares.
- It uses only Python's standard library, so it does not require numpy/scipy.

Important:
- This is position-only IK. It controls where the gripper goes, not the exact wrist orientation.
- If the real Webots Panda model differs from the DH approximation, some tuning may be needed.
"""

import math


JOINT_LIMITS = [
    (-2.60, 2.60),    # panda_joint1
    (-1.70, 1.70),    # panda_joint2
    (-2.60, 2.60),    # panda_joint3
    (-3.00, -0.20),   # panda_joint4
    (-2.60, 2.60),    # panda_joint5
    (-0.10, 3.60),    # panda_joint6
    (-2.60, 2.60),    # panda_joint7
]

# Common approximate DH parameters for Franka Panda.
# These are used for numerical IK estimation only.
A = [0.0, 0.0, 0.0, 0.0825, -0.0825, 0.0, 0.088]
D = [0.333, 0.0, 0.316, 0.0, 0.384, 0.0, 0.107]
ALPHA = [0.0, -math.pi / 2, math.pi / 2, math.pi / 2, -math.pi / 2, math.pi / 2, math.pi / 2]


def clamp(value, minimum, maximum):
    return max(minimum, min(maximum, value))


def clamp_joints(q):
    return [clamp(q[i], JOINT_LIMITS[i][0], JOINT_LIMITS[i][1]) for i in range(7)]


def matmul4(a, b):
    out = [[0.0 for _ in range(4)] for _ in range(4)]
    for i in range(4):
        for j in range(4):
            out[i][j] = (
                a[i][0] * b[0][j]
                + a[i][1] * b[1][j]
                + a[i][2] * b[2][j]
                + a[i][3] * b[3][j]
            )
    return out


def dh_transform(a, alpha, d, theta):
    ct = math.cos(theta)
    st = math.sin(theta)
    ca = math.cos(alpha)
    sa = math.sin(alpha)
    return [
        [ct, -st * ca, st * sa, a * ct],
        [st, ct * ca, -ct * sa, a * st],
        [0.0, sa, ca, d],
        [0.0, 0.0, 0.0, 1.0],
    ]


def panda_fk_position(q):
    """Approximate forward kinematics. Returns estimated end-effector [x, y, z]."""
    transform = [
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ]
    for i in range(7):
        transform = matmul4(transform, dh_transform(A[i], ALPHA[i], D[i], q[i]))
    return [transform[0][3], transform[1][3], transform[2][3]]


def vector_sub(a, b):
    return [a[i] - b[i] for i in range(len(a))]


def vector_norm(v):
    return math.sqrt(sum(x * x for x in v))


def compute_position_jacobian(q, epsilon=1e-4):
    """Numerical position Jacobian, output size 3 x 7."""
    base_position = panda_fk_position(q)
    jacobian = [[0.0 for _ in range(7)] for _ in range(3)]
    for j in range(7):
        q_perturbed = q[:]
        q_perturbed[j] += epsilon
        perturbed_position = panda_fk_position(q_perturbed)
        for axis in range(3):
            jacobian[axis][j] = (perturbed_position[axis] - base_position[axis]) / epsilon
    return jacobian


def mat3_inverse(m):
    a, b, c = m[0]
    d, e, f = m[1]
    g, h, i = m[2]
    determinant = a * (e * i - f * h) - b * (d * i - f * g) + c * (d * h - e * g)
    if abs(determinant) < 1e-9:
        return None
    inv_det = 1.0 / determinant
    return [
        [(e * i - f * h) * inv_det, (c * h - b * i) * inv_det, (b * f - c * e) * inv_det],
        [(f * g - d * i) * inv_det, (a * i - c * g) * inv_det, (c * d - a * f) * inv_det],
        [(d * h - e * g) * inv_det, (b * g - a * h) * inv_det, (a * e - b * d) * inv_det],
    ]


def damped_least_squares_step(jacobian, error, damping=0.08):
    """dq = J^T (J J^T + lambda^2 I)^-1 error"""
    jj_t = [[0.0 for _ in range(3)] for _ in range(3)]
    for r in range(3):
        for c in range(3):
            total = 0.0
            for k in range(7):
                total += jacobian[r][k] * jacobian[c][k]
            jj_t[r][c] = total
    for idx in range(3):
        jj_t[idx][idx] += damping * damping
    inv = mat3_inverse(jj_t)
    if inv is None:
        return [0.0 for _ in range(7)]
    temp = [0.0, 0.0, 0.0]
    for r in range(3):
        temp[r] = inv[r][0] * error[0] + inv[r][1] * error[1] + inv[r][2] * error[2]
    dq = [0.0 for _ in range(7)]
    for j in range(7):
        dq[j] = jacobian[0][j] * temp[0] + jacobian[1][j] * temp[1] + jacobian[2][j] * temp[2]
    return dq


def solve_ik_position(
    target_position,
    initial_q,
    max_iterations=80,
    tolerance=0.015,
    damping=0.08,
    step_scale=0.35,
    max_joint_step=0.06,
):
    """Solve position-based IK. Returns solution_q, info."""
    q = clamp_joints(initial_q[:])
    best_q = q[:]
    best_error_norm = 999999.0
    for iteration in range(max_iterations):
        current_position = panda_fk_position(q)
        error = vector_sub(target_position, current_position)
        error_norm = vector_norm(error)
        if error_norm < best_error_norm:
            best_error_norm = error_norm
            best_q = q[:]
        if error_norm <= tolerance:
            return q, {
                "success": True,
                "iterations": iteration + 1,
                "error_norm": error_norm,
                "current_position": current_position,
            }
        jacobian = compute_position_jacobian(q)
        dq = damped_least_squares_step(jacobian, error, damping=damping)
        for j in range(7):
            scaled_step = dq[j] * step_scale
            scaled_step = clamp(scaled_step, -max_joint_step, max_joint_step)
            q[j] += scaled_step
        q = clamp_joints(q)
    final_position = panda_fk_position(best_q)
    return best_q, {
        "success": False,
        "iterations": max_iterations,
        "error_norm": best_error_norm,
        "current_position": final_position,
    }
