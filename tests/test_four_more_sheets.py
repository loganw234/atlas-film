"""Lane K's four stocks: Plus-X, P3200, FP4+, HP5+.

Plus-X is the lane's calibration anchor - its ISO speed point
EMERGES on the unshifted axis. P3200 is the opposite and says so:
its sheet's artwork sits ~2.3 stops off its own EI (K2), so its
axis is anchored by prescription and the test is a wiring
assertion, labelled. The Ilfords' relative axes are anchored by
K8/K11's explicit offsets, with the translation-invariant ISO 6
contrast criterion as the real validation.
"""

import numpy as np
import pytest

from atlas_film import films


def speed_point(name):
    xx = np.linspace(-4.5, 1.0, 5501)
    d = films.characteristic(xx, name)
    floor = float(d.min())
    return float(np.interp(floor + 0.10, d, xx)), floor


def test_plusx_speed_emerges_unanchored():
    """The one stock this round whose absolute axis is the sheet's
    own: ISO 125 puts H_m at -2.194, and the traced table crosses
    floor+0.10 within the house 0.04 band."""
    x, _ = speed_point("plusx")
    assert abs(x - np.log10(0.8 / 125)) < 0.04


@pytest.mark.parametrize("name,iso", [
    ("p3200", 1000), ("fp4", 125), ("hp5", 400)])
def test_anchored_axes_land_where_prescribed(name, iso):
    """Anchored BY CONSTRUCTION (P3200's artwork defect K2; the
    Ilfords' relative axes K8/K11) - this asserts the wiring, not
    a discovery, and says so."""
    x, _ = speed_point(name)
    assert abs(x - np.log10(0.8 / iso)) < 0.02


@pytest.mark.parametrize("name,delta_tol", [
    ("fp4", 0.06), ("hp5", 0.06)])
def test_ilford_contrast_criterion_survives_translation(name,
                                                        delta_tol):
    """The real validation for a relative-axis sheet: ISO 6's
    contrast criterion (0.80 D over 1.30 log H from the speed
    point) is translation-invariant, and both Ilford curves sit
    within 0.05 D of it as traced."""
    x0, floor = speed_point(name)
    d = float(films.characteristic(np.array([x0 + 1.30]), name)[0])
    assert abs(d - (floor + 0.10 + 0.80)) < delta_tol


@pytest.mark.parametrize("name", ["plusx", "p3200", "fp4", "hp5"])
def test_the_tables_are_the_source(name):
    """Interp at the knots is the knots: the shipped table IS the
    traced geometry."""
    cx, cd = films.FILMS[name]["curve"]
    got = films.characteristic(np.asarray(cx), name)
    assert np.allclose(got, cd, atol=1e-12)


@pytest.mark.parametrize("name", ["plusx", "p3200", "fp4", "hp5"])
def test_the_new_sheets_are_grainy_pan_stocks(name):
    grey = np.full((48, 48, 3), 0.3)
    E = films.normal_exposure(grey, name)
    d = films.negative(grey, E, name, pitch_um=3.0, seed=6)
    assert float(d.std()) > 0.003
    assert np.array_equal(
        d, films.negative(grey, E, name, pitch_um=3.0, seed=6))
    red = np.zeros((4, 4, 3))
    red[..., 0] = 0.9
    dr = films.negative(red, E, name, grain=False)
    assert float(dr.max()) > films.FILMS[name]["curve"][1][0]


@pytest.mark.parametrize("name", ["plusx", "p3200", "fp4", "hp5"])
def test_the_new_sheets_refuse_what_no_lane_traced(name):
    with pytest.raises(ValueError, match="tracing lane"):
        films.characteristic(0.0, name, ci=0.7)
    with pytest.raises(ValueError, match="no reciprocity table"):
        films.reciprocity(name, 10.0)
