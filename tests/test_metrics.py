import numpy as np
import pyfar as pf
import pytest

from metrics import (
    METRIC_FUNCTIONS, localization_error, get_metric_metadata,
    describe_metrics, querrMiddlebrooks, rmsPmedianlocal, polar_gain,
    wrap_polar_angle,
)

POLAR_GRID = np.arange(-30.0, 211.0, 30.0)   # the 9-point AXD median-plane grid


def coords(lat_deg, pol_deg):
    lat = np.deg2rad(np.asarray(lat_deg, dtype=float))
    pol = np.deg2rad(np.asarray(pol_deg, dtype=float))
    return pf.Coordinates.from_spherical_side(lat, pol, np.ones_like(lat))


def hp(lat_deg, pol_deg):
    """(n, 2) horizontal-polar array in radians."""
    return np.column_stack([np.deg2rad(np.asarray(lat_deg, dtype=float)),
                            np.deg2rad(np.asarray(pol_deg, dtype=float))])


@pytest.fixture
def grid_targets():
    pol = np.tile(POLAR_GRID, 5)
    return coords(np.zeros_like(pol), pol)


# --- registry ---------------------------------------------------------------

def test_registry_metadata_complete():
    required = {'name', 'coord_convention', 'input_unit', 'output_unit',
                'description'}
    for name, func in METRIC_FUNCTIONS.items():
        md = get_metric_metadata(name)
        assert required <= md.keys()
        assert md['name'] == name
        assert md['coord_convention'] in ('cartesian', 'spherical',
                                          'horizontal-polar')
        assert func.__name__ == name


def test_describe_metrics(capsys):
    describe_metrics()
    describe_metrics('querrMiddlebrooks')
    out = capsys.readouterr().out
    assert 'querrMiddlebrooks' in out and 'coord_convention' in out


def test_get_metric_metadata_unknown():
    with pytest.raises(ValueError):
        get_metric_metadata('nope')


# --- localization_error -----------------------------------------------------

def test_localization_error_input_validation(grid_targets):
    with pytest.raises(TypeError):
        localization_error(np.zeros((3, 3)), np.zeros((3, 3)), 'rmsL')
    with pytest.raises(ValueError):
        localization_error(grid_targets, coords([0.0], [0.0]), 'rmsL')
    with pytest.raises(ValueError):
        localization_error(grid_targets, grid_targets, 'no_such_metric')


def test_localization_error_unknown_kwarg_warns(grid_targets):
    with pytest.warns(UserWarning):
        value = localization_error(grid_targets, grid_targets, 'rmsL', bogus=1)
    assert value == pytest.approx(0.0, abs=1e-9)


def test_localization_error_custom_callable(grid_targets):
    def my_metric(targets, estimations, scale=1.0):
        return scale * targets.cshape[0]
    assert localization_error(grid_targets, grid_targets, my_metric, scale=2.0) \
        == 2 * grid_targets.cshape[0]


@pytest.mark.parametrize('metric', ['rmsL', 'rmsPmedianlocal', 'angular_error',
                                    'rmsEle', 'sdPol', 'sdLat', 'accL_cutoff',
                                    'accP_cutoff'])
def test_perfect_response_gives_zero(grid_targets, metric):
    assert localization_error(grid_targets, grid_targets, metric) \
        == pytest.approx(0.0, abs=1e-9)


def test_perfect_response_no_quadrant_errors(grid_targets):
    value, aux = localization_error(grid_targets, grid_targets,
                                    'querrMiddlebrooks', auxiliary_output=True)
    assert value == 0.0
    assert aux == {'confusion_count': 0,
                   'response_count': grid_targets.cshape[0]}


def test_metric_without_aux_returns_empty_dict(grid_targets):
    value, aux = localization_error(grid_targets, grid_targets, 'rmsL',
                                    auxiliary_output=True)
    assert value == pytest.approx(0.0, abs=1e-9) and aux == {}


# --- classical metrics on hand-built data -----------------------------------

def test_querr_hand_built():
    pol_t = np.array([0, 30, 60, 90, 120, 150, 0, 30, 60, 90], dtype=float)
    pol_r = pol_t.copy()
    pol_r[:2] = 180.0 - pol_t[:2]           # two front/back confusions
    lat = np.zeros_like(pol_t)

    value, aux = querrMiddlebrooks(hp(lat, pol_t), hp(lat, pol_r))
    assert value == pytest.approx(20.0)
    assert aux['confusion_count'] == 2 and aux['response_count'] == 10

    value2, aux2 = localization_error(coords(lat, pol_t), coords(lat, pol_r),
                                      'querrMiddlebrooks', auxiliary_output=True)
    assert value2 == pytest.approx(20.0) and aux2 == aux


def test_rmsPmedianlocal_fixed_error():
    pol_t = np.array([0, 30, 60, 90, 120, 150], dtype=float)
    lat = np.zeros_like(pol_t)
    value, _ = rmsPmedianlocal(hp(lat, pol_t), hp(lat, pol_t + 10.0))
    assert value == pytest.approx(np.deg2rad(10.0))


def test_rmsPmedianlocal_excludes_lateral_and_confusions():
    pol_t = np.array([0, 30, 60, 90, 120, 150, 30, 60], dtype=float)
    pol_r = pol_t + 10.0
    pol_r[6] = 30.0 + 80.0     # lateral trial: excluded by |lat| > 30°
    pol_r[7] = 60.0 + 120.0    # confusion (> 90°): excluded from local RMS
    lat_r = np.array([0, 0, 0, 0, 0, 0, 45, 0], dtype=float)
    value, _ = rmsPmedianlocal(hp(np.zeros(8), pol_t), hp(lat_r, pol_r))
    assert value == pytest.approx(np.deg2rad(10.0))


def test_rmsPmedianlocal_and_querr_raise_on_invalid_input():
    pol = np.array([0.0, 30.0, 60.0])
    with pytest.raises(ValueError):
        rmsPmedianlocal(hp([0, 0, 0], pol), hp([0, 0, 100], pol))
    with pytest.raises(ValueError):          # no central responses
        rmsPmedianlocal(hp([0, 0, 0], pol), hp([45, 45, 45], pol))
    with pytest.raises(ValueError):
        querrMiddlebrooks(hp([0, 0, 0], pol), hp([45, 45, 45], pol))


def test_polar_gain():
    pol_t = np.tile(POLAR_GRID, 5)
    lat = np.zeros_like(pol_t)
    assert polar_gain(pol_t, pol_t, lat) == pytest.approx(1.0)
    assert polar_gain(pol_t, 0.5 * pol_t + 10.0, lat) == pytest.approx(0.5)
    assert np.isnan(polar_gain(pol_t[:3], pol_t[:3], lat[:3]))


def test_gainP_registered_matches_polar_gain(grid_targets):
    assert localization_error(grid_targets, grid_targets, 'gainP') \
        == pytest.approx(1.0)


def test_wrap_polar_angle_reexported():
    assert wrap_polar_angle(np.deg2rad(275.0)) == pytest.approx(np.deg2rad(-85.0))


# --- mixture model through the registry -------------------------------------

def test_mixture_model_via_localization_error():
    rng = np.random.default_rng(3)
    pol_t = rng.choice(POLAR_GRID, size=150)
    confused = rng.random(150) > 0.9
    mu = np.where(confused, 180.0 - pol_t, pol_t)
    pol_r = np.rad2deg(wrap_polar_angle(
        rng.vonmises(np.deg2rad(mu), 1 / np.deg2rad(20.0) ** 2)))
    lat = np.zeros_like(pol_t)

    value, aux = localization_error(coords(lat, pol_t), coords(lat, pol_r),
                                    'mixture_model', use_prior=False,
                                    auxiliary_output=True)
    assert value.shape == (3,)
    assert 0.0 < value[0] <= 1.0 and value[1] > 0 and np.isnan(value[2])
    assert aux['n_trials'] == 150 and np.isfinite(aux['bic'])
