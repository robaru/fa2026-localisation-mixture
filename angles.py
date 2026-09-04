"""
Angle wrapping helpers shared by ``metrics`` and ``bayesian_metric``.

Conventions
-----------
Horizontal-polar coordinates: the *lateral* angle lies in [-π/2, π/2]
(negative = left) and the *polar* angle lies in [-π/2, 3π/2), where 0 is
the front horizontal plane, π/2 is above the head and π is behind.
"""
import numpy as np


def wrap_to_pi(rad):
    """Wrap angles to [-π, π)."""
    return (rad + np.pi) % (2 * np.pi) - np.pi


def wrap_polar_angle(angle_rad):
    """Wrap polar angles to the range [-π/2, 3π/2) ≡ [-90°, 270°)."""
    return (angle_rad + np.pi / 2) % (2 * np.pi) - np.pi / 2
