import numpy as np
import pytest
from scipy.special import i0

from bayesian_metric import (
    sigma_to_kappa, kappa_to_sigma, vm_product, polar_mirror,
    eval_mixture, sample_from_prior_mixture,
)


@pytest.mark.parametrize('sigma', [2.0, 10.0, 30.0, 60.0, 120.0])
def test_sigma_kappa_roundtrip(sigma):
    assert kappa_to_sigma(sigma_to_kappa(sigma)) == pytest.approx(sigma, rel=1e-5)


def test_sigma_to_kappa_monotonic():
    sigmas = np.array([2.0, 5.0, 10.0, 20.0, 40.0, 80.0, 160.0])
    kappas = np.array([sigma_to_kappa(s) for s in sigmas])
    assert np.all(np.diff(kappas) < 0)


def test_vm_product_collinear():
    R, mu = vm_product(0.3, 2.0, 0.3, 5.0)
    assert R == pytest.approx(7.0)
    assert mu == pytest.approx(0.3)


def test_vm_product_antipodal():
    R, mu = vm_product(0.0, 5.0, np.pi, 2.0)
    assert R == pytest.approx(3.0)
    assert mu == pytest.approx(0.0, abs=1e-12)


def test_vm_product_broadcasts_over_mu1():
    mu1 = np.deg2rad([0.0, 45.0, 90.0])
    R, mu = vm_product(mu1, 3.0, 0.0, 1.0)
    assert R.shape == mu.shape == (3,)
    assert R[0] == pytest.approx(4.0)


def test_eval_mixture_integrates_to_one():
    phi = np.linspace(-np.pi / 2, 3 * np.pi / 2, 4001)
    pdf = eval_mixture(phi, np.deg2rad(45.0), 0.8,
                       sigma_to_kappa(25.0), sigma_to_kappa(30.0))
    assert np.trapezoid(pdf, phi) == pytest.approx(1.0, abs=1e-4)


def test_eval_mixture_flat_prior_reduces_to_two_components():
    phi = np.linspace(-np.pi / 2, 3 * np.pi / 2, 721)
    phi_t, w, kappa = np.deg2rad(60.0), 0.75, sigma_to_kappa(20.0)

    def vm_pdf(x, mu, k):
        return np.exp(k * np.cos(x - mu)) / (2 * np.pi * i0(k))

    two = w * vm_pdf(phi, phi_t, kappa) + (1 - w) * vm_pdf(phi, np.pi - phi_t, kappa)
    np.testing.assert_allclose(eval_mixture(phi, phi_t, w, kappa, 1e-9), two,
                               rtol=1e-5, atol=1e-8)


def test_mixture_weights_flat_prior_limit():
    # With kappa_prior -> 0 the four component weights approach
    # [w/2, w/2, (1-w)/2, (1-w)/2].
    w, kappa, eps = 0.8, sigma_to_kappa(25.0), 1e-6
    pt = np.array([0.0])
    R_A, _ = vm_product(pt, kappa, 0.0, eps)
    R_B, _ = vm_product(pt, kappa, np.pi, eps)
    I0_A, I0_B = i0(R_A), i0(R_B)
    denom = I0_A + I0_B
    alpha = np.array([w * I0_A / denom, w * I0_B / denom,
                      (1 - w) * I0_B / denom, (1 - w) * I0_A / denom]).ravel()
    np.testing.assert_allclose(alpha, [w / 2, w / 2, (1 - w) / 2, (1 - w) / 2],
                               atol=1e-4)


def test_sample_from_prior_mixture_shape_range_and_seed():
    pt = np.deg2rad(np.tile(np.arange(-30.0, 211.0, 30.0), 10))
    kappa, kappa_prior = sigma_to_kappa(20.0), sigma_to_kappa(40.0)
    a = sample_from_prior_mixture(pt, 0.8, kappa, kappa_prior,
                                  np.random.default_rng(7))
    b = sample_from_prior_mixture(pt, 0.8, kappa, kappa_prior,
                                  np.random.default_rng(7))
    assert a.shape == pt.shape
    assert np.all(a >= -np.pi / 2) and np.all(a < 3 * np.pi / 2)
    np.testing.assert_array_equal(a, b)


def test_sample_from_prior_mixture_tracks_targets_when_precise():
    pt = np.deg2rad(np.tile(np.arange(-30.0, 211.0, 30.0), 20))
    pr = sample_from_prior_mixture(pt, 1.0, sigma_to_kappa(3.0), 1e-9,
                                   np.random.default_rng(0))
    err = np.abs(np.angle(np.exp(1j * (pr - pt))))
    assert np.rad2deg(err).max() < 20.0
    assert polar_mirror(pt).shape == pt.shape
