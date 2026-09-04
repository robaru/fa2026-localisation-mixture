"""
Localization-error metrics and the metric registry.

This module is borrowed from the ``bayesian_listener`` package
(https://github.com/robaru/bayesian_listener, file
``bayesian_listener/metrics.py``) and trimmed to the metrics used in this
repository: the classical median-plane metrics the paper compares against
(quadrant error rate, local polar RMS error, polar bias, polar gain) and the
von Mises mixture model. The full set of metrics is available upstream.
"""
import functools
import inspect
import warnings

import numpy as np
import pandas as pd
import pyfar as pf

# Re-exported so that `from metrics import wrap_to_pi` keeps working.
from angles import wrap_to_pi, wrap_polar_angle  # noqa: F401
from bayesian_metric import fit_mixture_querr, fit_mixture_with_prior


def localization_error(targets, estimations, metric,
                       auxiliary_output=False, **kwargs):
    """
    Compute the localization error between two sets of coordinates
    using the specified metric.

    Parameters
    ----------
    targets : pyfar.Coordinates
        The target (reference) coordinates.
    estimations : pyfar.Coordinates
        The estimated coordinates to compare against.
    metric : str or callable
        The metric to use for error computation.
        -   If a string, it should be a registered metric name.
            You can view available metrics using describe_metrics()
            and get specific details with describe_metrics(name).
        -   If a callable, it must be a function that takes
            two pyfar.Coordinates arguments (targets, estimations)
            as the first two positional arguments, followed by any
            additional keyword arguments passed via **kwargs.
            In this case, the user is responsible that the correct
            coordinate system, units, and extra arguments are used.
            The callable should return either a single float (error value)
            or a tuple (error_value, auxiliary_data).
    auxiliary_output : bool, optional
        This is irrelevant if `metric` is a callable,
        since the user should handle this in their custom function.
        If True, also returns the auxiliary output (dict)
        from the metric function, if available.
        Default is False.
    **kwargs : dict, optional
        Additional keyword arguments passed to the metric function.
        -   If `metric` is a registered string, kwargs are validated
            against the function signature. Unknown kwargs produce a
            UserWarning and are ignored; valid ones are forwarded.
            You can check the expected extra kwargs for a registered metric
            using describe_metrics(name).
        -   If `metric` is a callable, kwargs are forwarded as-is
            with no validation. The user is responsible for correctness.

    Returns
    -------
    float or tuple :
        The computed localization error.
        If `auxiliary_output` is True, the output will be a tuple:
        (error_value, auxiliary_data_dict).
        If the metric function does not provide auxiliary data,
        auxiliary_data_dict will be an empty dictionary.

    Examples
    --------
    Registered metric with extra kwarg:

    >>> error = localization_error(targets, estimations,
    ...                            'accP_cutoff',
    ...                            cutoff=np.deg2rad(30))

    Registered metric with auxiliary output:

    >>> error, aux = localization_error(targets, estimations,
    ...                                 'querrMiddlebrooks',
    ...                                 auxiliary_output=True)
    >>> print(error)
    9.375
    >>> print(aux)
    {'confusion_count': 48, 'response_count': 512}

    Custom callable with extra kwarg:

    >>> def my_metric(targets, estimations, threshold=0.5):
    ...     ...
    >>> error = localization_error(targets, estimations,
    ...                            my_metric,
    ...                            threshold=0.1)
    """
    # Accept only Coordinates instances
    if not isinstance(targets, pf.Coordinates) or \
       not isinstance(estimations, pf.Coordinates):
        raise TypeError(
            "Both targets and estimations must be " \
            "pyfar.Coordinates instances.")

    if targets.cshape != estimations.cshape:
        raise ValueError(
            f"Shape mismatch: {targets.cshape} vs {estimations.cshape}")

    # Case 1: metric is a custom function
    if callable(metric):
        return metric(targets, estimations, **kwargs)

    # Case 2: metric is a string, but not registered in METRIC_FUNCTIONS
    if metric not in METRIC_FUNCTIONS:
        raise ValueError(
            f"Unknown metric: {metric}. Available metrics are: "
            f"{list(METRIC_FUNCTIONS.keys())}")

    # Case 3: metric is a string and registered in METRIC_FUNCTIONS
    if kwargs: # Validate extra kwargs against the function's signature
        sig = inspect.signature(METRIC_FUNCTIONS[metric])
        # Skip the first two positional params (true, est)
        extra_params = set(list(sig.parameters.keys())[2:])
        invalid = set(kwargs.keys()) - extra_params
        if invalid:
            warnings.warn(
                f"localization_error: unknown kwargs {invalid} "
                f"for metric '{metric}' will be ignored. "
                f"Valid extra parameters are: {extra_params or 'none'}.",
                UserWarning,
                stacklevel=2,
            )
            kwargs = {k: v for k, v in kwargs.items() if k in extra_params}

    expected_coord_convention = \
        get_metric_metadata(metric)['coord_convention']
    expected_unit = get_metric_metadata(metric)['input_unit']

    # Expected conventions and units are internally generated
    # by the registration system, there is no need to check them here.
    # The conventions are in ['cartesian', 'spherical', 'horizontal-polar']
    # The units are in ['radians', 'degrees', 'meters']
    # For the same reason, we assume units are coherent within the conventions.

    # Convert coordinates to the expected convention
    if expected_coord_convention == 'cartesian':
        converted_tar = targets.cartesian
        converted_est = estimations.cartesian
    elif expected_coord_convention == 'spherical':
        converted_tar = targets.spherical_elevation
        converted_est = estimations.spherical_elevation
    else:  # expected_coord_convention == 'horizontal-polar'
        converted_tar = targets.spherical_side
        converted_est = estimations.spherical_side

    # Convert units if necessary
    # Coordinates class uses radians and meters internally,
    # so we only need a conversion if expected_unit is 'degrees'
    if expected_unit == 'degrees':
        # Only convert the angular components (rad, rad, m) → (deg, deg, m)
        converted_tar[:, :2] = np.rad2deg(converted_tar[:, :2])
        converted_est[:, :2] = np.rad2deg(converted_est[:, :2])

    value, aux_out = \
        METRIC_FUNCTIONS[metric](converted_tar, converted_est, **kwargs)

    return (value, aux_out) if auxiliary_output else value


# Shared dictionary to hold metric functions and their metadata
METRIC_FUNCTIONS = {}

def register_metric(name,
                    coord_convention,
                    input_unit,
                    output_unit=None,
                    description=None,
                    **extra_metadata,
                    ):
    """
    Decorator to register a metric function with metadata.

    Parameters
    ----------
    name : str
        Name of the metric.
    coord_convention : str
        Coordinate convention used (e.g., 'horizontal-polar').
    input_unit : str
        Unit of the input data (e.g., 'radians').
    output_unit : str, optional
        Unit of the output data (e.g., 'radians', 'percentage').
    description : str, optional
        Description of the metric.
    **extra_metadata : dict
        Additional metadata to store.

    Returns
    -------
    decorator : function
        Decorator that registers the metric function.
    """
    def decorator(func):
        """
        Decorator that registers the metric function with metadata.
        """
        @functools.wraps(func)
        def wrapped(*args, **kwargs):
            """
            Wrapper to ensure uniform output format.
            """
            result = func(*args, **kwargs)
            if isinstance(result, tuple):
                value, auxiliary_output = result
            else:
                value = result
                auxiliary_output = {}
            # Every function is uniformly formatted to return a tuple
            return value, auxiliary_output
        wrapped._metadata = {
            'name': name,
            'coord_convention': coord_convention,
            'input_unit': input_unit,
            'output_unit': output_unit,
            'description': description,
            **extra_metadata,
        }
        METRIC_FUNCTIONS[name] = wrapped
        return wrapped
    return decorator


def get_metric_metadata(name):
    """
    Retrieve metadata for a registered metric.

    Parameters
    ----------
    name : str
        Name of the metric.

    Returns
    -------
    metadata : dict
        Metadata dictionary for the metric.
    """
    func = METRIC_FUNCTIONS.get(name)
    if func is None:
        raise ValueError(f"Metric '{name}' not found.")
    # Return a copy to prevent external modification
    return func._metadata.copy()


def describe_metrics(name=None):
    """
    Print descriptions of registered metrics.

    Parameters
    ----------
    name : str, optional
        Name of the metric to describe. If None, lists all metrics.
    """
    if name:
        info = get_metric_metadata(name)
        print(f"Metric: {name}")
        for key, value in info.items():
            print(f"  {key}: {value}")
    else:
        print("Available metrics:")
        for name in METRIC_FUNCTIONS.keys():
            print(f"  {name}: {get_metric_metadata(name)['description']}")
        print(
            "Use describe_metrics(name) to get details for a specific metric.")




# -----------------------------------------------------------------------------
# Metric Functions
@register_metric(
    name="accP_cutoff",
    coord_convention="horizontal-polar",
    input_unit="radians",
    output_unit="radians",
    description=(
        "Elevation bias (mean signed error) within ±cutoff° lateral.\n\t"
        "Mean of the signed difference between response and target polar angles\n\t"
        "within ±cutoff° lateral. Cutoff defaults to 30° (π/6 radians).\n\t"
        "Positive values indicate upward bias, negative values indicate downward bias."
    ),
    ylabel="Elevation bias (rad)",
)
def accP_cutoff(true, est, cutoff=np.deg2rad(30)):
    """
    Compute elevation bias (mean signed error) within ±cutoff° lateral.
    More details in the decorator above.
    """
    lat_est = wrap_to_pi(est[..., 0])
    mask = np.abs(lat_est) <= cutoff
    if not np.any(mask):
        return np.nan

    pol_true = wrap_polar_angle(true[..., 1])
    pol_est = wrap_polar_angle(est[..., 1])

    diff = wrap_to_pi(pol_est - pol_true)[mask]
    return np.mean(diff)


@register_metric(
    name="rmsPmedianlocal",
    coord_convention="horizontal-polar",
    input_unit="radians",
    output_unit="radians",
    description=(
        "RMS polar error (local, central responses only).\n\t"
        "Root mean square of polar angle error,\n\t"
        "restricted to responses with:\n\t"
        "- lateral response within ±30° (±π/6 radians)\n\t"
        "- polar error less than 90° (π/2 radians).\n\t"
        "Based on definition in Middlebrooks (1999)."
    ),
    ylabel="Local central RMS polar error (rad)",
)
def rmsPmedianlocal(true, est):
    """
    Compute local RMS polar error within ±30° lateral and polar error < 90°.
    More details in the decorator above.
    """
    # lateral in [-π, π), then restrict to [-π/2, π/2]
    lat_est = wrap_to_pi(est[..., 0])
    if not np.all(np.abs(lat_est) <= np.pi/2):
        raise ValueError("Lateral angles must be in [-π/2, π/2].")

    pol_true = wrap_polar_angle(true[..., 1])  # polar in [-π/2, 3π/2)
    pol_est = wrap_polar_angle(est[..., 1])

    # 1. Select central responses: lateral response within ±30°
    central_mask = np.abs(lat_est) <= np.deg2rad(30)
    if not np.any(central_mask):
        raise ValueError(
            "No central responses found within ±30° lateral range.")

    # 2. Exclude responses with polar error greater than 90°
    polar_diff = wrap_to_pi(pol_est - pol_true)[central_mask]
    local_mask = np.abs(polar_diff) < np.deg2rad(90)
    if not np.any(local_mask):
        raise ValueError("No responses with polar error < 90° found.")

    local_polar_diff = polar_diff[local_mask]
    return np.sqrt(np.mean(local_polar_diff ** 2))



@register_metric(
    name="querrMiddlebrooks",
    coord_convention="horizontal-polar",
    input_unit="radians",
    output_unit="percentage",
    description=(
        "Quadrant error rate as defined in Middlebrooks (1999).\n\t"
        "Fraction of responses with polar error ≥ 90° (π/2 rad),\n\t"
        "restricted to responses with lateral angle in ±30° (±π/6 rad)."
    ),
    ylabel="Quadrant errors (%)",
    auxiliary_output={
        'confusion_count': 'Number of confusions (polar error ≥ 90°)',
        'response_count': \
            'Number of responses within the lateral range (|lat| ≤ 30°)',
    },
)
def querrMiddlebrooks(true, est):
    """
    Compute quadrant error rate as defined in Middlebrooks (1999).
    More details in the decorator above.
    """
    # lateral in [-π, π), then restrict to [-π/2, π/2]
    lat_est = wrap_to_pi(est[..., 0])
    if not np.all(np.abs(lat_est) <= np.pi/2):
        raise ValueError("Lateral angles must be in [-π/2, π/2].")

    pol_true = wrap_polar_angle(true[..., 1])  # polar in [-π/2, 3π/2)
    pol_est = wrap_polar_angle(est[..., 1])

    # 1. Filter central responses: lateral response within ±30°
    central_mask = np.abs(lat_est) <= np.deg2rad(30)
    if not np.any(central_mask):
        raise ValueError(
            "No central responses found within ±30° lateral range.")

    # 2. Compute polar error and count confusions (polar error ≥ 90°)
    polar_error = np.abs(wrap_to_pi(pol_est - pol_true))[central_mask]
    n_confusions = int(np.sum(polar_error >= np.deg2rad(90)))
    n_total = len(polar_error)

    qerr = 100 * n_confusions / n_total
    return qerr, {'confusion_count': n_confusions, 'response_count': n_total}


def polar_gain(pol_target_deg, pol_response_deg, lat_target_deg,
               lat_cutoff_deg=30.0, delta=40.0, Nmin=5, maxiter=100):
    """
    Polar gain (gainP) via the Selective Iterative Regression Procedure (SIRP).

    Computes the average of frontal and rear polar gain as defined in
    Macpherson & Middlebrooks (2000). The SIRP iteratively excludes outliers
    and reversals to obtain a robust linear regression slope.

    Parameters
    ----------
    pol_target_deg : array-like
        Polar target angles in degrees.
    pol_response_deg : array-like
        Polar response angles in degrees.
    lat_target_deg : array-like
        Lateral target angles in degrees. Used to filter central targets.
    lat_cutoff_deg : float
        Lateral cutoff for central targets (default 30°).
    delta : float
        Outlier tolerance in degrees (default 40°).
    Nmin : int
        Minimum number of inliers required for regression (default 5).
    maxiter : int
        Maximum number of SIRP iterations (default 100).

    Returns
    -------
    float
        gainP = (gainPfront + gainPrear) / 2, or NaN if insufficient data.
    """
    pol_t = np.asarray(pol_target_deg,   dtype=float)
    pol_r = np.asarray(pol_response_deg, dtype=float)
    lat_t = np.asarray(lat_target_deg,   dtype=float)

    # Filter to central targets
    central = np.abs(lat_t) <= lat_cutoff_deg
    pol_t = pol_t[central]
    pol_r = pol_r[central]

    gains = []
    for is_front in (True, False):
        # Pool = correct-hemifield responses only (paper section 6)
        if is_front:
            correct = (pol_t <= 90.0) & (pol_r <= 90.0)
        else:
            correct = (pol_t >= 90.0) & (pol_r >= 90.0)

        x = pol_t[correct]
        y = pol_r[correct]

        if len(x) < Nmin:
            gains.append(np.nan)
            continue

        # SIRP: start with all correct-hemifield responses, iterate until convergence.
        # new_inliers is evaluated on the full pool each step, so previously excluded
        # points are automatically available for re-selection (paper section 6).
        inliers = np.ones(len(x), dtype=bool)
        for _ in range(maxiter):
            A = np.column_stack([np.ones(inliers.sum()), x[inliers]])
            b, _, _, _ = np.linalg.lstsq(A, y[inliers], rcond=None)
            yhat = b[1] * x + b[0]
            dev = (y - yhat + 180.0) % 360.0 - 180.0
            new_inliers = np.abs(dev) < delta
            if np.array_equal(new_inliers, inliers):
                break
            inliers = new_inliers
            if inliers.sum() < Nmin:
                break

        if inliers.sum() < Nmin:
            gains.append(np.nan)
            continue

        A_final = np.column_stack([np.ones(inliers.sum()), x[inliers]])
        b_final, _, _, _ = np.linalg.lstsq(A_final, y[inliers], rcond=None)
        gains.append(b_final[1])

    if any(np.isnan(g) for g in gains):
        return np.nan
    return float(np.mean(gains))


@register_metric(
    name='gainP',
    coord_convention='horizontal-polar',
    input_unit='radians',
    output_unit='unitless',
    description=(
        "Polar gain via the Selective Iterative Regression Procedure (SIRP).\n\t"
        "Average of frontal and rear polar gain slopes as defined in\n\t"
        "Macpherson & Middlebrooks (2000). Fitted on central targets\n\t"
        "(|lat_target| ≤ 30°). Returns NaN if insufficient data."),
    ylabel="Polar gain",
)
def gainP(true, est, lat_cutoff_deg=30.0, delta=40.0, Nmin=5, maxiter=100):
    """
    Registered wrapper around polar_gain().

    Parameters
    ----------
    true : ndarray, shape (n, 3)
        Target directions: columns [lateral (rad), polar (rad), radius].
    est : ndarray, shape (n, 3)
        Response directions: columns [lateral (rad), polar (rad), radius].
    """
    return polar_gain(
        pol_target_deg=np.rad2deg(true[..., 1]),
        pol_response_deg=np.rad2deg(est[..., 1]),
        lat_target_deg=np.rad2deg(true[..., 0]),
        lat_cutoff_deg=lat_cutoff_deg,
        delta=delta,
        Nmin=Nmin,
        maxiter=maxiter,
    )


@register_metric(
    name='mixture_model',
    coord_convention='horizontal-polar',
    input_unit='radians',
    output_unit='mixed',
    description=(
        "Von Mises mixture model parameters fitted to polar responses.\n\t"
        "Fits central targets (|lat_target| ≤ 30°) via pyBADS MLE.\n\t"
        "Two-parameter model (use_prior=False):\n\t"
        "  p(φ|φ_t) = w·VM(φ_t,κ) + (1−w)·VM(π−φ_t,κ)\n\t"
        "Three-parameter model (use_prior=True) adds a horizontal spatial prior.\n\t"
        "Returns np.array([w_hat, sigma_hat, sigma_prior_hat]);\n\t"
        "sigma_prior_hat is NaN when use_prior=False."
    ),
    ylabel="[w_hat, sigma_hat (°), sigma_prior_hat (°)]",
    auxiliary_output={
        'nll':     'Negative log-likelihood at optimum',
        'bic':     'Bayesian Information Criterion',
        'n_trials': 'Number of central trials used for fitting',
    },
)
def mixture_model(true, est, use_prior=True, lat_cutoff_deg=30.0):
    """
    Fit the von Mises mixture model to polar responses for central targets.

    Parameters
    ----------
    true : ndarray, shape (n, 2)
        Target directions: columns [lateral (rad), polar (rad)].
    est : ndarray, shape (n, 2)
        Response directions: columns [lateral (rad), polar (rad)].
    use_prior : bool
        If True, fit the 3-parameter extended model with horizontal prior.
        If False, fit the 2-parameter model (no prior; sigma_prior_hat=NaN).
    lat_cutoff_deg : float
        Lateral cutoff for central targets (default 30°).

    Returns
    -------
    value : np.ndarray, shape (3,)
        [w_hat, sigma_hat (degrees), sigma_prior_hat (degrees)].
        sigma_prior_hat is NaN when use_prior=False.
    auxiliary_output : dict
        nll, bic, n_trials.
    """
    true = np.asarray(true)
    est  = np.asarray(est)

    df = pd.DataFrame({
        'lat_target':   np.rad2deg(true[:, 0]),
        'pol_target':   np.rad2deg(true[:, 1]),
        'lat_response': np.rad2deg(est[:, 0]),
        'pol_response': np.rad2deg(est[:, 1]),
    })

    if use_prior:
        fit = fit_mixture_with_prior(df, lat_cutoff_deg=lat_cutoff_deg)
        n_params = 3
    else:
        fit = fit_mixture_querr(df, lat_cutoff_deg=lat_cutoff_deg)
        fit['sigma_prior_hat'] = np.nan
        n_params = 2

    n = fit['n_trials']
    bic = n_params * np.log(n) + 2 * fit['nll'] if n > 0 else np.nan

    value = np.array([fit['w_hat'], fit['sigma_hat'], fit['sigma_prior_hat']])
    aux   = {'nll': fit['nll'], 'bic': bic, 'n_trials': n}
    return value, aux
