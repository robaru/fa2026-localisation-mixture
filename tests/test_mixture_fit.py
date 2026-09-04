import numpy as np
import pandas as pd
import pytest

from bayesian_metric import (
    fit_mixture_querr, fit_mixture_with_prior, sample_from_prior_mixture,
    sigma_to_kappa,
)

POLAR_GRID = np.arange(-30.0, 211.0, 30.0)
W, SIGMA, SIGMA_PRIOR = 0.85, 20.0, 40.0


def make_df(w, sigma, sigma_prior, n=200, seed=1234):
    rng = np.random.default_rng(seed)
    pol_t = rng.choice(POLAR_GRID, size=n)
    kappa_prior = sigma_to_kappa(sigma_prior) if sigma_prior is not None else 1e-9
    pol_r = sample_from_prior_mixture(np.deg2rad(pol_t), w, sigma_to_kappa(sigma),
                                      kappa_prior, rng)
    return pd.DataFrame({
        'lat_target': 0.0,
        'pol_target': pol_t,
        'lat_response': 0.0,
        'pol_response': np.rad2deg(pol_r),
    })


@pytest.fixture(scope='module')
def df_flat_prior():
    return make_df(W, SIGMA, None)


@pytest.fixture(scope='module')
def df_with_prior():
    return make_df(W, SIGMA, SIGMA_PRIOR)


def test_two_param_fit_recovers(df_flat_prior):
    fit = fit_mixture_querr(df_flat_prior)
    assert fit['n_trials'] == 200
    assert abs(fit['w_hat'] - W) < 0.1
    assert abs(fit['sigma_hat'] - SIGMA) < 8.0
    assert fit['confusion_pct'] == pytest.approx(100 * (1 - fit['w_hat']))
    assert np.isfinite(fit['nll'])


def test_three_param_fit_recovers(df_with_prior):
    ext = fit_mixture_with_prior(df_with_prior)
    assert ext['n_trials'] == 200
    assert abs(ext['w_hat'] - W) < 0.1
    assert abs(ext['sigma_hat'] - SIGMA) < 8.0
    assert abs(ext['sigma_prior_hat'] - SIGMA_PRIOR) < 15.0

    orig = fit_mixture_querr(df_with_prior)
    assert ext['nll'] <= orig['nll'] + 1.0


def test_too_few_trials_returns_nan(df_flat_prior):
    fit = fit_mixture_querr(df_flat_prior.iloc[:2])
    assert fit['n_trials'] == 2 and np.isnan(fit['w_hat'])
    ext = fit_mixture_with_prior(df_flat_prior.iloc[:2])
    assert ext['n_trials'] == 2 and np.isnan(ext['sigma_prior_hat'])


def test_lateral_cutoff_selects_central_trials(df_flat_prior):
    df = df_flat_prior.copy()
    df.loc[df.index[:50], 'lat_target'] = 45.0
    assert fit_mixture_querr(df)['n_trials'] == 150
    assert fit_mixture_querr(df, lat_cutoff_deg=60.0)['n_trials'] == 200
