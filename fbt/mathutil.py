"""Small linear-algebra helpers.

Coordinate system used everywhere in the app is Unity's: X = right, Y = up,
Z = forward (left-handed). Rotation matrices store the local axes of an object
as COLUMNS: R[:, 0] = local right, R[:, 1] = local up, R[:, 2] = local forward.
Numerically the matrices behave exactly like Unity's (Quaternion <-> Matrix
formulas are identical), so the quaternions / euler angles produced here can be
fed straight into VRChat and the Nexovara Pose driver.
"""
from __future__ import annotations

import math
from typing import Optional, Tuple

import numpy as np

EPS = 1e-9
X_AXIS = np.array([1.0, 0.0, 0.0])
Y_AXIS = np.array([0.0, 1.0, 0.0])
Z_AXIS = np.array([0.0, 0.0, 1.0])


def normalize(v) -> Optional[np.ndarray]:
    """Return unit vector or None if the vector is degenerate / not finite.
    (Plain-float math: ~5x faster than np.linalg.norm for 3-vectors.)"""
    v = np.asarray(v, dtype=np.float64)
    if v.shape == (3,):
        x, y, z = float(v[0]), float(v[1]), float(v[2])
        n = math.sqrt(x * x + y * y + z * z)
    else:
        n = float(np.linalg.norm(v))
    if not math.isfinite(n) or n < EPS:
        return None
    return v / n


def cross3(a, b) -> np.ndarray:
    """Cross product of two 3-vectors (np.cross is ~20x slower for tiny inputs)."""
    a0, a1, a2 = float(a[0]), float(a[1]), float(a[2])
    b0, b1, b2 = float(b[0]), float(b[1]), float(b[2])
    return np.array([a1 * b2 - a2 * b1, a2 * b0 - a0 * b2, a0 * b1 - a1 * b0])


def _perp(primary: np.ndarray, hint, fallbacks) -> Optional[np.ndarray]:
    """Component of `hint` perpendicular to unit `primary` (normalized)."""
    for h in (hint, *fallbacks):
        if h is None:
            continue
        h = np.asarray(h, dtype=np.float64)
        s = normalize(h - primary * float(np.dot(h, primary)))
        if s is not None:
            return s
    return None


def basis_from_right_up(right, up_hint) -> Optional[np.ndarray]:
    """X is exact (right), Y is up_hint orthogonalized, Z = X x Y."""
    x = normalize(right)
    if x is None:
        return None
    y = _perp(x, up_hint, (Y_AXIS, Z_AXIS))
    if y is None:
        return None
    z = cross3(x, y)
    return np.column_stack((x, y, z))


def basis_from_up_forward(up, forward_hint) -> Optional[np.ndarray]:
    """Y is exact (up), Z is forward_hint orthogonalized, X = Y x Z."""
    y = normalize(up)
    if y is None:
        return None
    z = _perp(y, forward_hint, (Z_AXIS, X_AXIS))
    if z is None:
        return None
    x = cross3(y, z)
    return np.column_stack((x, y, z))


def basis_from_forward_up(forward, up_hint) -> Optional[np.ndarray]:
    """Z is exact (forward), Y is up_hint orthogonalized, X = Y x Z."""
    z = normalize(forward)
    if z is None:
        return None
    y = _perp(z, up_hint, (Y_AXIS, X_AXIS))
    if y is None:
        return None
    x = cross3(y, z)
    return np.column_stack((x, y, z))


def axis_angle(axis, angle_rad: float) -> np.ndarray:
    a = normalize(axis)
    if a is None:
        return np.eye(3)
    x, y, z = a
    c, s = math.cos(angle_rad), math.sin(angle_rad)
    C = 1.0 - c
    return np.array([
        [c + x * x * C, x * y * C - z * s, x * z * C + y * s],
        [y * x * C + z * s, c + y * y * C, y * z * C - x * s],
        [z * x * C - y * s, z * y * C + x * s, c + z * z * C],
    ])


def rotation_between(a, b) -> np.ndarray:
    """Smallest rotation that maps direction a onto direction b."""
    a = normalize(a)
    b = normalize(b)
    if a is None or b is None:
        return np.eye(3)
    d = float(np.clip(np.dot(a, b), -1.0, 1.0))
    if d > 1.0 - 1e-9:
        return np.eye(3)
    if d < -1.0 + 1e-9:
        # 180 degrees: prefer rotating around the world up axis (keeps "up" up)
        axis = _perp(a, Y_AXIS, (X_AXIS, Z_AXIS))
        return axis_angle(axis, math.pi)
    axis = cross3(a, b)
    return axis_angle(axis, math.acos(d))


def rot_x(deg: float) -> np.ndarray:
    r = math.radians(deg)
    c, s = math.cos(r), math.sin(r)
    return np.array([[1, 0, 0], [0, c, -s], [0, s, c]], dtype=np.float64)


def rot_y(deg: float) -> np.ndarray:
    """Rotation about the up axis. rot_y(a) @ (0,0,1) == (sin a, 0, cos a)."""
    r = math.radians(deg)
    c, s = math.cos(r), math.sin(r)
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]], dtype=np.float64)


def rot_z(deg: float) -> np.ndarray:
    r = math.radians(deg)
    c, s = math.cos(r), math.sin(r)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]], dtype=np.float64)


def yaw_of(forward) -> float:
    """Yaw (degrees) of a direction projected on the floor plane."""
    f = np.asarray(forward, dtype=np.float64)
    return math.degrees(math.atan2(f[0], f[2]))


def mat_to_quat(R: np.ndarray) -> Tuple[float, float, float, float]:
    """Rotation matrix -> quaternion (x, y, z, w)."""
    m = R
    tr = m[0, 0] + m[1, 1] + m[2, 2]
    if tr > 0.0:
        s = math.sqrt(tr + 1.0) * 2.0
        w = 0.25 * s
        x = (m[2, 1] - m[1, 2]) / s
        y = (m[0, 2] - m[2, 0]) / s
        z = (m[1, 0] - m[0, 1]) / s
    elif m[0, 0] > m[1, 1] and m[0, 0] > m[2, 2]:
        s = math.sqrt(1.0 + m[0, 0] - m[1, 1] - m[2, 2]) * 2.0
        w = (m[2, 1] - m[1, 2]) / s
        x = 0.25 * s
        y = (m[0, 1] + m[1, 0]) / s
        z = (m[0, 2] + m[2, 0]) / s
    elif m[1, 1] > m[2, 2]:
        s = math.sqrt(1.0 + m[1, 1] - m[0, 0] - m[2, 2]) * 2.0
        w = (m[0, 2] - m[2, 0]) / s
        x = (m[0, 1] + m[1, 0]) / s
        y = 0.25 * s
        z = (m[1, 2] + m[2, 1]) / s
    else:
        s = math.sqrt(1.0 + m[2, 2] - m[0, 0] - m[1, 1]) * 2.0
        w = (m[1, 0] - m[0, 1]) / s
        x = (m[0, 2] + m[2, 0]) / s
        y = (m[1, 2] + m[2, 1]) / s
        z = 0.25 * s
    n = math.sqrt(x * x + y * y + z * z + w * w) or 1.0
    return x / n, y / n, z / n, w / n


def quat_to_mat(q) -> np.ndarray:
    x, y, z, w = q
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ])


def unity_euler_to_mat(ex: float, ey: float, ez: float) -> np.ndarray:
    """Unity's Quaternion.Euler(x, y, z) as a matrix (Z, then X, then Y)."""
    return rot_y(ey) @ rot_x(ex) @ rot_z(ez)


def mat_to_unity_euler(R: np.ndarray) -> Tuple[float, float, float]:
    """Matrix -> Unity eulerAngles (degrees, each in [0, 360))."""
    sx = float(np.clip(-R[1, 2], -1.0, 1.0))
    ax = math.asin(sx)
    if math.cos(ax) > 1e-6:
        ay = math.atan2(R[0, 2], R[2, 2])
        az = math.atan2(R[1, 0], R[1, 1])
    else:  # gimbal lock
        az = 0.0
        ay = math.atan2(-R[2, 0], R[0, 0])
    return tuple(math.degrees(a) % 360.0 for a in (ax, ay, az))  # type: ignore[return-value]


def is_finite(*arrays) -> bool:
    return all(np.all(np.isfinite(a)) for a in arrays)


# ---------------------------------------------------------------- quaternions
def quat_normalize(q) -> np.ndarray:
    q = np.asarray(q, dtype=np.float64)
    # math.sqrt on a 4-dot is ~4x faster than np.linalg.norm for tiny vectors
    # (called tens of thousands of times per second by the 90 Hz output)
    n2 = float(q[0] * q[0] + q[1] * q[1] + q[2] * q[2] + q[3] * q[3])
    if not n2 > EPS * EPS or not math.isfinite(n2):
        return np.array([0.0, 0.0, 0.0, 1.0])
    return q / math.sqrt(n2)


def quat_angle(a, b) -> float:
    """Angle (radians) between two orientations (x, y, z, w)."""
    d = abs(float(np.dot(quat_normalize(a), quat_normalize(b))))
    return 2.0 * math.acos(min(1.0, d))


def quat_slerp(a, b, t: float) -> np.ndarray:
    """Spherical interpolation, shortest path; t may slightly exceed [0, 1]
    (used for short extrapolation)."""
    a = quat_normalize(a)
    b = quat_normalize(b)
    d = float(np.dot(a, b))
    if d < 0.0:          # take the short way round
        b, d = -b, -d
    if d > 0.9995:       # nearly identical -> lerp is accurate and stable
        return quat_normalize(a + (b - a) * t)
    th = math.acos(min(1.0, d))
    s = math.sin(th)
    return quat_normalize((math.sin((1 - t) * th) * a + math.sin(t * th) * b) / s)


class QuatOneEuro:
    """One Euro filter for orientations: strong smoothing when still,
    little lag when turning fast (adaptive slerp factor)."""

    def __init__(self, min_cutoff: float = 2.0, beta: float = 0.5, d_cutoff: float = 1.0):
        self.min_cutoff, self.beta, self.d_cutoff = min_cutoff, beta, d_cutoff
        self.reset()

    def reset(self) -> None:
        self._q = None
        self._t = None
        self._speed = 0.0

    @staticmethod
    def _alpha(dt: float, cutoff: float) -> float:
        tau = 1.0 / (2.0 * math.pi * cutoff)
        return 1.0 / (1.0 + tau / dt)

    def __call__(self, R: np.ndarray, t: float) -> np.ndarray:
        q = np.array(mat_to_quat(R))
        if self._q is None or self._t is None:
            self._q, self._t = q, t
            return R
        dt = t - self._t
        if dt <= 1e-6:
            return quat_to_mat(self._q)
        if dt > 1.0:
            self.reset()
            return self(R, t)
        speed = quat_angle(self._q, q) / dt
        a_d = self._alpha(dt, self.d_cutoff)
        self._speed = a_d * speed + (1 - a_d) * self._speed
        a = self._alpha(dt, self.min_cutoff + self.beta * self._speed)
        self._q = quat_slerp(self._q, q, a)
        self._t = t
        return quat_to_mat(self._q)
