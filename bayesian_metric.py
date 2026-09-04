"""
Bayesian mixture model metric for sound localisation.

Two-parameter model (use_prior=False):

    p(φ | φ_t) = w · VM(φ_t, κ) + (1 − w) · VM(π − φ_t, κ)

Three-parameter model with horizontal prior (use_prior=True):

    p(φ | φ_t) ∝ [w · VM(φ_t, κ) + (1−w) · VM(π−φ_t, κ)]
                × [0.5 · VM(0, κ_prior) + 0.5 · VM(π, κ_prior)]

Parameters estimated via pyBADS MLE:
    - w_hat           : correct-hemifield weight  (≈ 1 − querr/100)
    - kappa_hat       : von Mises concentration
    - sigma_hat       : polar precision in degrees (derived from kappa_hat)
    - kappa_prior_hat : prior concentration  (3-param model only)
    - sigma_prior_hat : prior width in degrees (3-param model only)
"""
import numpy as np
from scipy.optimize import brentq
from scipy.special import i0, i1, ive, expit
from scipy.special import logsumexp
from pybads import BADS

# Re-exported so that `from bayesian_metric import wrap_polar_angle` works.
from angles import wrap_to_pi, wrap_polar_angle  # noqa: F401

__all__ = [
    'wrap_to_pi', 'wrap_polar_angle', 'polar_mirror',
    'sigma_to_kappa', 'kappa_to_sigma', 'vm_product',
    'eval_mixture', 'sample_from_prior_mixture',
    'fit_mixture_querr', 'fit_mixture_with_prior',
]


# ---------------------------------------------------------------------------
# σ ↔ κ transforms

def _bessel_ratio(kappa):
    """Compute i1(kappa)/i0(kappa) with asymptotic expansion for large kappa."""
    if kappa < 1e-10:
        return 0.0
    if kappa > 500:
        return 1.0 - 1.0 / (2.0 * kappa) - 1.0 / (8.0 * kappa**2)
    return float(i1(kappa) / i0(kappa))


def sigma_to_kappa(sigma):
    """
    Convert circular standard deviation (degrees) to von Mises concentration κ.

    Solves i1(κ)/i0(κ) = exp(−σ²/2) via Brent's method.

    Parameters
    ----------
    sigma : float
        Circular standard deviation in degrees.

    Returns
    -------
    float
        Von Mises concentration parameter kappa.
    """
    if sigma >= 180.0:
        return 1e-6
    if sigma < 1e-3:
        return 1e6

    sigma_rad = np.deg2rad(sigma)
    R_target = np.exp(-sigma_rad**2 / 2)

    if R_target > 0.99:
        kappa_init = 1.0 / (2.0 * (1.0 - R_target))
        if abs(_bessel_ratio(kappa_init) - R_target) < 1e-10:
            return kappa_init

    try:
        kappa = brentq(lambda k: _bessel_ratio(k) - R_target, 1e-6, 1e4)
    except ValueError:
        kappa = 1.0 / (2.0 * (1.0 - R_target))

    return kappa


def kappa_to_sigma(kappa):
    """
    Convert von Mises concentration κ to circular standard deviation (degrees).

    Uses σ = √(−2 · log(i1(κ)/i0(κ))).

    Parameters
    ----------
    kappa : float
        Von Mises concentration parameter.

    Returns
    -------
    float
        Circular standard deviation in degrees.
    """
    if kappa < 1e-6:
        return 180.0
    R = _bessel_ratio(kappa)
    return np.rad2deg(np.sqrt(-2 * np.log(R)))


# ---------------------------------------------------------------------------
# Angle helpers

def polar_mirror(polar_rad):
    """Mirror polar angle front ↔ back: π − φ (wrapped to [−π/2, 3π/2))."""
    return wrap_polar_angle(np.pi - polar_rad)


# ---------------------------------------------------------------------------
# Von Mises product helper

def vm_product(mu1, kappa1, mu2, kappa2):
    """
    Resultant (R, mu*) of VM(phi; mu1, kappa1) * VM(phi; mu2, kappa2).

    Uses the von Mises addition theorem:
        R · exp(i·mu*) = kappa1 · exp(i·mu1) + kappa2 · exp(i·mu2)

    mu1 may be an array; other arguments may be scalars.

    Returns
    -------
    R   : resultant concentration
    mu* : resultant mean angle
    """
    x = kappa1 * np.cos(mu1) + kappa2 * np.cos(mu2)
    y = kappa1 * np.sin(mu1) + kappa2 * np.sin(mu2)
    return np.sqrt(x**2 + y**2), np.arctan2(y, x)


# ---------------------------------------------------------------------------
# Density evaluation and sampling (three-parameter model)

def eval_mixture(phi, phi_t, w, kappa, kappa_prior):
    """
    Evaluate the normalised 4-component mixture density at ``phi``.

    Parameters
    ----------
    phi : array
        Polar angles (radians) at which to evaluate the density.
    phi_t : float
        Polar target angle (radians).
    w, kappa, kappa_prior : float
        Model parameters. ``kappa_prior`` close to 0 recovers the
        two-parameter model.

    Returns
    -------
    array with the same shape as ``phi``.
    """
    pm = np.pi - phi_t  # mirror target

    R_A, mu_A = vm_product(phi_t, kappa, 0.0,   kappa_prior)
    R_B, mu_B = vm_product(phi_t, kappa, np.pi, kappa_prior)
    R_C, mu_C = vm_product(pm,    kappa, 0.0,   kappa_prior)
    R_D, mu_D = vm_product(pm,    kappa, np.pi, kappa_prior)

    I0_A = i0(float(R_A))
    I0_B = i0(float(R_B))
    denom = I0_A + I0_B

    alpha_A = w       * I0_A / denom
    alpha_B = w       * I0_B / denom
    alpha_C = (1 - w) * I0_B / denom
    alpha_D = (1 - w) * I0_A / denom

    def vm_pdf(phi, mu, R):
        return np.exp(R * np.cos(phi - mu)) / (2 * np.pi * i0(float(R)))

    return (alpha_A * vm_pdf(phi, mu_A, R_A)
          + alpha_B * vm_pdf(phi, mu_B, R_B)
          + alpha_C * vm_pdf(phi, mu_C, R_C)
          + alpha_D * vm_pdf(phi, mu_D, R_D))


def sample_from_prior_mixture(pol_targets_rad, w, kappa, kappa_prior, rng):
    """
    Sample one polar response per trial from the normalised prior-mixture model.

    Analytic: draws component index from Categorical(alpha), then samples
    from the selected VM(mu_i, R_i) via rng.vonmises().

    Parameters
    ----------
    pol_targets_rad : array (n_trials,)
    w, kappa, kappa_prior : scalars
        ``kappa_prior`` close to 0 samples from the two-parameter model.
    rng : numpy Generator

    Returns
    -------
    array (n_trials,) of polar responses in radians, wrapped to [-pi/2, 3pi/2)
    """
    pt = pol_targets_rad
    pm = wrap_polar_angle(np.pi - pt)

    R_A, mu_A = vm_product(pt, kappa, 0.0,    kappa_prior)  # correct × front prior
    R_B, mu_B = vm_product(pt, kappa, np.pi,  kappa_prior)  # correct × back prior
    R_C, mu_C = vm_product(pm, kappa, 0.0,    kappa_prior)  # mirror  × front prior
    R_D, mu_D = vm_product(pm, kappa, np.pi,  kappa_prior)  # mirror  × back prior

    # Mixture weights  (R_C = R_B, R_D = R_A by symmetry)
    I0_A = i0(R_A)  # shape (n,)
    I0_B = i0(R_B)
    denom = I0_A + I0_B

    alpha = np.stack([
        w       * I0_A / denom,   # alpha_A
        w       * I0_B / denom,   # alpha_B
        (1 - w) * I0_B / denom,   # alpha_C
        (1 - w) * I0_A / denom,   # alpha_D
    ], axis=1)  # shape (n, 4)

    R_stack  = np.stack([R_A,  R_B,  R_C,  R_D],  axis=1)  # (n, 4)
    mu_stack = np.stack([mu_A, mu_B, mu_C, mu_D], axis=1)  # (n, 4)

    n = len(pt)
    # Per-trial draw (kept as a loop so that seeded results match the paper).
    comp = np.array([rng.choice(4, p=alpha[i]) for i in range(n)])

    R_sel  = R_stack[np.arange(n),  comp]
    mu_sel = mu_stack[np.arange(n), comp]

    return wrap_polar_angle(rng.vonmises(mu_sel, R_sel))


# ---------------------------------------------------------------------------
# Mixture model fit

def fit_mixture_querr(df, lat_cutoff_deg=30.0):
    """
    Fit a von Mises mixture model to polar responses for central targets.

    Optimises over [logit(w), log(κ)] using pyBADS. The log-normalising
    constant uses ive() to remain numerically stable for large κ.

    Parameters
    ----------
    df : DataFrame
        Must contain columns: lat_target, pol_target, pol_response (degrees).
    lat_cutoff_deg : float
        Lateral cutoff for "central" targets (default 30°).

    Returns
    -------
    dict with keys:
        w_hat       : correct-hemifield weight
        kappa_hat   : von Mises concentration
        sigma_hat   : polar precision (degrees)
        confusion_pct : (1 − w_hat) × 100
        nll         : negative log-likelihood at optimum
        n_trials    : number of central trials used
    """
    lat_targ = np.deg2rad(df['lat_target'].values)
    pol_targ = np.deg2rad(df['pol_target'].values)
    pol_resp = np.deg2rad(df['pol_response'].values)

    central_mask = np.abs(lat_targ) <= np.deg2rad(lat_cutoff_deg)

    nan_result = dict(w_hat=np.nan, kappa_hat=np.nan, sigma_hat=np.nan,
                      confusion_pct=np.nan, nll=np.nan,
                      n_trials=int(central_mask.sum()))
    if central_mask.sum() < 3:
        return nan_result

    pr = wrap_polar_angle(pol_resp[central_mask])
    pt = wrap_polar_angle(pol_targ[central_mask])
    pm = polar_mirror(pt)

    def mixture_nll(x):
        w     = float(expit(x[0]))
        kappa = float(np.exp(x[1]))
        # log I0(κ) = log ive(0, κ) + κ  — stable for all κ
        log_C   = -np.log(2 * np.pi) - np.log(ive(0, kappa)) - kappa
        ll_corr = log_C + kappa * np.cos(pr - pt)
        ll_mirr = log_C + kappa * np.cos(pr - pm)
        log_w   = np.log(w)
        log_1mw = np.log(1 - w)
        max_ll  = np.maximum(log_w + ll_corr, log_1mw + ll_mirr)
        log_mix = max_ll + np.log(
            np.exp(log_w   + ll_corr - max_ll) +
            np.exp(log_1mw + ll_mirr - max_ll)
        )
        return -np.sum(log_mix)

    x0  = np.array([0.0,  np.log(sigma_to_kappa(30.0))])
    lb  = np.array([-10.0, np.log(sigma_to_kappa(80.0))])
    ub  = np.array([ 10.0, np.log(sigma_to_kappa(2.0))])
    plb = np.array([-2.0,  np.log(sigma_to_kappa(60.0))])
    pub = np.array([ 2.0,  np.log(sigma_to_kappa(5.0))])

    bads   = BADS(mixture_nll, x0, lb, ub, plb, pub, options={'display': 'off'})
    result = bads.optimize()

    x_opt     = result['x']
    w_hat     = float(expit(x_opt[0]))
    kappa_hat = float(np.exp(x_opt[1]))

    return dict(
        w_hat         = w_hat,
        kappa_hat     = kappa_hat,
        sigma_hat     = kappa_to_sigma(kappa_hat),
        confusion_pct = (1 - w_hat) * 100,
        nll           = float(result['fval']),
        n_trials      = int(central_mask.sum()),
    )


def fit_mixture_with_prior(df, lat_cutoff_deg=30.0):
    """
    Fit the extended 3-parameter von Mises mixture model via pyBADS.

    The model is a four-component mixture derived from:

        p(φ | φ_t) ∝ [w · VM(φ_t, κ) + (1−w) · VM(π−φ_t, κ)]
                    × [0.5 · VM(0, κ_prior) + 0.5 · VM(π, κ_prior)]

    The product is evaluated analytically via the von Mises addition theorem
    (vm_product), yielding components A–D with the symmetry R_C=R_B, R_D=R_A,
    so only two Bessel evaluations are needed per trial.

    Parameters
    ----------
    df : DataFrame
        Must contain columns: lat_target, pol_target, pol_response (degrees).
    lat_cutoff_deg : float
        Lateral cutoff for "central" targets (default 30°).

    Returns
    -------
    dict with keys:
        w_hat           : correct-hemifield weight
        kappa_hat       : von Mises concentration
        sigma_hat       : polar precision (degrees)
        kappa_prior_hat : prior concentration
        sigma_prior_hat : prior width (degrees); large = weak prior
        confusion_pct   : (1 − w_hat) × 100
        nll             : negative log-likelihood at optimum
        n_trials        : number of central trials used
    """
    lat_targ = np.deg2rad(df['lat_target'].values)
    pol_targ = np.deg2rad(df['pol_target'].values)
    pol_resp = np.deg2rad(df['pol_response'].values)

    central = np.abs(lat_targ) <= np.deg2rad(lat_cutoff_deg)
    n_central = int(central.sum())

    nan_result = dict(w_hat=np.nan, kappa_hat=np.nan, sigma_hat=np.nan,
                      kappa_prior_hat=np.nan, sigma_prior_hat=np.nan,
                      confusion_pct=np.nan, nll=np.nan, n_trials=n_central)
    if n_central < 3:
        return nan_result

    pr = wrap_polar_angle(pol_resp[central])
    pt = wrap_polar_angle(pol_targ[central])
    pm = wrap_polar_angle(np.pi - pt)

    def mixture_nll(x):
        w           = float(expit(x[0]))
        kappa       = float(np.exp(x[1]))
        kappa_prior = float(np.exp(x[2]))

        R_A, mu_A = vm_product(pt, kappa, 0.0,   kappa_prior)
        R_B, mu_B = vm_product(pt, kappa, np.pi, kappa_prior)
        R_C, mu_C = vm_product(pm, kappa, 0.0,   kappa_prior)
        R_D, mu_D = vm_product(pm, kappa, np.pi, kappa_prior)

        # R_C == R_B and R_D == R_A by symmetry; two Bessel evaluations suffice
        log_I0_A = np.log(ive(0, R_A)) + R_A
        log_I0_B = np.log(ive(0, R_B)) + R_B
        log_denom = np.logaddexp(log_I0_A, log_I0_B)

        log_w   = np.log(w)
        log_1mw = np.log(1.0 - w)

        log_alpha_A = log_w   + log_I0_A - log_denom
        log_alpha_B = log_w   + log_I0_B - log_denom
        log_alpha_C = log_1mw + log_I0_B - log_denom   # R_C == R_B
        log_alpha_D = log_1mw + log_I0_A - log_denom   # R_D == R_A

        def log_vm(phi, mu, R):
            return -np.log(2 * np.pi) - np.log(ive(0, R)) - R + R * np.cos(phi - mu)

        terms = np.stack([
            log_alpha_A + log_vm(pr, mu_A, R_A),
            log_alpha_B + log_vm(pr, mu_B, R_B),
            log_alpha_C + log_vm(pr, mu_C, R_C),
            log_alpha_D + log_vm(pr, mu_D, R_D),
        ], axis=0)

        return -np.sum(logsumexp(terms, axis=0))

    x0  = np.array([0.0, np.log(sigma_to_kappa(30.0)), np.log(sigma_to_kappa(30.0))])
    lb  = np.array([-10.0, np.log(sigma_to_kappa(150.0)), np.log(sigma_to_kappa(100.0))])
    ub  = np.array([ 10.0, np.log(sigma_to_kappa(2.0)),   np.log(sigma_to_kappa(2.0))])
    plb = np.array([-2.0,  np.log(sigma_to_kappa(60.0)),  np.log(sigma_to_kappa(60.0))])
    pub = np.array([ 2.0,  np.log(sigma_to_kappa(5.0)),   np.log(sigma_to_kappa(10.0))])

    bads   = BADS(mixture_nll, x0, lb, ub, plb, pub, options={'display': 'off'})
    result = bads.optimize()

    x_opt  = result['x']
    w_hat  = float(expit(x_opt[0]))
    k_hat  = float(np.exp(x_opt[1]))
    kp_hat = float(np.exp(x_opt[2]))

    return dict(
        w_hat           = w_hat,
        kappa_hat       = k_hat,
        sigma_hat       = kappa_to_sigma(k_hat),
        kappa_prior_hat = kp_hat,
        sigma_prior_hat = kappa_to_sigma(kp_hat),
        confusion_pct   = (1 - w_hat) * 100,
        nll             = float(result['fval']),
        n_trials        = n_central,
    )
