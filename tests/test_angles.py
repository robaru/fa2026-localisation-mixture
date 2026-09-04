import numpy as np
import pytest

from angles import wrap_to_pi, wrap_polar_angle
from bayesian_metric import polar_mirror


def test_wrap_to_pi_boundaries():
    assert wrap_to_pi(np.pi) == pytest.approx(-np.pi)
    assert wrap_to_pi(3 * np.pi) == pytest.approx(-np.pi)
    assert wrap_to_pi(0.0) == 0.0


def test_wrap_to_pi_range_and_equivalence():
    x = np.linspace(-20, 20, 1001)
    y = wrap_to_pi(x)
    assert np.all(y >= -np.pi) and np.all(y < np.pi)
    np.testing.assert_allclose(np.cos(y), np.cos(x), atol=1e-12)
    np.testing.assert_allclose(np.sin(y), np.sin(x), atol=1e-12)


def test_wrap_polar_angle_values():
    assert wrap_polar_angle(np.deg2rad(275)) == pytest.approx(np.deg2rad(-85))
    assert wrap_polar_angle(np.deg2rad(-100)) == pytest.approx(np.deg2rad(260))
    assert wrap_polar_angle(np.deg2rad(45)) == pytest.approx(np.deg2rad(45))


def test_wrap_polar_angle_range():
    x = np.linspace(-20, 20, 1001)
    y = wrap_polar_angle(x)
    assert np.all(y >= -np.pi / 2) and np.all(y < 3 * np.pi / 2)
    np.testing.assert_allclose(np.cos(y), np.cos(x), atol=1e-12)


def test_polar_mirror():
    assert polar_mirror(0.0) == pytest.approx(np.pi)
    assert polar_mirror(np.pi / 2) == pytest.approx(np.pi / 2)
    x = np.deg2rad([-60.0, 0.0, 45.0, 90.0, 135.0, 200.0])
    np.testing.assert_allclose(polar_mirror(polar_mirror(x)), x, atol=1e-12)
