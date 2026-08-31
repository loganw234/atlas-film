"""The real colour stocks answer to their datasheets.

The FAKE-stock referees (test_colour_is_three_sheets) prove the
assembly machinery; these prove the SHIPPED CONSTANTS. The tables
embedded here are the traced source data - the characteristic
curves the constants were derived FROM, judged through the dye
matrices that came from a DIFFERENT figure - so reassembly is an
assembly check, not a fit grading its own homework.

Acceptance is sourced per stock class: the negatives within the
read-off tolerance of their plates (0.05 D), the print film within
one printer light (0.025 log H, the medium's own control quantum,
FILM-P-13) - it holds 0.12 lights.
"""

import numpy as np
import pytest

from atlas_film import colour

# KODAK VISION3 50D 5203, traced characteristic curves: log H
# against Status M (R, G, B)
TRACED_50D = [
    (-3.03, 0.19, 0.60, 0.87), (-2.2, 0.21, 0.62, 0.91),
    (-1.6, 0.42, 0.85, 1.13), (-0.9, 0.76, 1.25, 1.50),
    (-0.515, 0.96, 1.46, 1.70), (0.2, 1.28, 1.84, 2.07),
    (0.8, 1.56, 2.17, 2.40), (1.4, 1.77, 2.44, 2.66),
    (2.006, 1.91, 2.62, 2.82)]

# KODAK VISION3 500T 5219, same figure kind (ECN-2 densitometry)
TRACED_5219 = [
    (-4.0, 0.18, 0.58, 0.84), (-3.0, 0.28, 0.69, 0.93),
    (-2.5, 0.44, 0.88, 1.13), (-2.0, 0.67, 1.15, 1.40),
    (-1.5, 0.91, 1.42, 1.65), (-1.0, 1.14, 1.69, 1.91),
    (-0.5, 1.37, 1.97, 2.18), (0.0, 1.58, 2.22, 2.43),
    (0.5, 1.76, 2.43, 2.63), (1.0, 1.86, 2.56, 2.75)]

# KODAK VISION3 250D 5207 (lane E: raster/vector two-method
# agreement <= 0.003 D on the floors)
TRACED_250D = [
    (-3.700, 0.177, 0.577, 0.843), (-3.100, 0.208, 0.610, 0.867),
    (-2.500, 0.346, 0.766, 1.003), (-1.900, 0.606, 1.068, 1.319),
    (-1.300, 0.891, 1.402, 1.641), (-0.700, 1.177, 1.737, 1.963),
    (-0.100, 1.455, 2.066, 2.284), (0.500, 1.694, 2.358, 2.562),
    (1.100, 1.843, 2.553, 2.732)]

# KODAK VISION3 200T 5213 (lane E)
TRACED_200T = [
    (-3.684, 0.178, 0.580, 0.847), (-3.084, 0.194, 0.596, 0.864),
    (-2.484, 0.300, 0.711, 0.979), (-1.884, 0.570, 1.024, 1.277),
    (-1.284, 0.859, 1.363, 1.600), (-0.684, 1.145, 1.696, 1.920),
    (-0.084, 1.426, 2.029, 2.240), (0.516, 1.672, 2.328, 2.526),
    (1.116, 1.824, 2.527, 2.708)]

# KODAK VISION Premier 2383, traced as log H AT density (the curve
# is near-vertical: reading the figure the other way round would
# multiply the read-off error by gamma five), Status A per channel
TRACED_2383 = {
    0: [(0.20, 0.543), (0.50, 0.822), (1.00, 1.010), (1.50, 1.135),
        (2.00, 1.245), (2.50, 1.352), (3.00, 1.473), (3.50, 1.635),
        (4.00, 1.994)],
    1: [(0.20, 0.191), (0.50, 0.448), (1.00, 0.666), (1.50, 0.815),
        (2.00, 0.940), (2.50, 1.057), (3.00, 1.192), (3.50, 1.367),
        (4.00, 1.738)],
    2: [(0.20, -0.223), (0.50, 0.119), (1.00, 0.365), (1.50, 0.509),
        (2.00, 0.618), (2.50, 0.724), (3.00, 0.841), (3.50, 0.992),
        (4.00, 1.403)],
}


def assemble_at(name, x):
    """Drive the stock with a neutral log-exposure sweep and return
    the assembled per-channel densities."""
    grey = np.ones((len(x), 1, 3), np.float64)
    E = (10.0 ** np.asarray(x, np.float64))[:, None, None]
    st = colour._stock(colour.COLOUR_FILMS if name in
                       colour.COLOUR_FILMS else colour.PRINT_FILMS,
                       name, "?")
    dyes = colour._develop_layers(grey * E, 1.0, st, None, False,
                                  0, name)
    return colour._assemble(dyes, st)[:, 0, :]


@pytest.mark.parametrize("name,traced,tol", [
    ("50d", TRACED_50D, 0.05), ("5219", TRACED_5219, 0.05),
    ("250d", TRACED_250D, 0.03), ("200t", TRACED_200T, 0.035)])
def test_negative_reassembles_its_datasheet(name, traced, tol):
    x = np.array([p[0] for p in traced])
    want = np.array([p[1:] for p in traced])
    got = assemble_at(name, x)
    assert float(np.abs(got - want).max()) < tol


def test_print_film_holds_a_printer_light():
    """2383's assembly against every traced point, judged on the
    horizontal axis in units of the lab's own control quantum."""
    x = np.linspace(-1.3, 2.5, 1901)
    D = assemble_at("2383", x)
    worst = 0.0
    for c in range(3):
        for dd, hh in TRACED_2383[c]:
            h = abs(float(np.interp(dd, D[:, c], x)) - hh)
            worst = max(worst, h)
    assert worst < 0.025


def test_lad_lights_land_the_aims():
    """The printer calibration: solved lights print the LAD grey to
    the print film's published aims."""
    lights = colour.lad_lights("50d", "2383")
    t = 10.0 ** -np.asarray(colour.LAD_NEGATIVE, np.float64)
    d = colour.positive(t[None, None, :], "2383", lights=lights,
                        grain=False)[0, 0]
    assert np.abs(d - np.asarray(
        colour.PRINT_FILMS["2383"]["lad_aim"])).max() < 2e-3
    assert np.all(lights > 0) and np.all(np.isfinite(lights))


def test_the_chain_prints_a_grey_scene():
    """End to end: grey scene -> metered 50d negative -> LAD-lit
    2383 print. What the sources promise, no more: the print is a
    POSITIVE; the G channel, which the meter placed on its aim,
    follows its print aim; and the R/B deviations stay inside the
    negative's own patch-spacing deviation amplified by the print
    film's steepest slope (the dossier's 4.7) - the traced Vision3
    B channel sits ~0.2 D thinner over G than H-61A's nominal
    patch, and LAD's own paper says scenes need not match the
    patch. Neutrality by trim is the next test's job."""
    lum = np.full((3, 1, 3), 0.18)
    lum[0] *= 0.05 / 0.18
    lum[2] *= 1.0 / 0.18
    E = colour.normal_exposure(np.full((8, 8, 3), 1.0), "50d")
    neg = colour.negative(lum, E, "50d", grain=False)
    lights = colour.lad_lights("50d", "2383")
    pr = colour.positive(colour.transmit(neg), "2383",
                         lights=lights, grain=False)
    assert np.all(pr[0].mean() > pr[1].mean() > pr[2].mean())
    aims = np.asarray(colour.PRINT_FILMS["2383"]["lad_aim"])
    grey_dev = pr[1, 0] - aims
    assert abs(float(grey_dev[1])) < 0.10
    neg_dev = np.asarray(colour.LAD_NEGATIVE) - neg[1, 0]
    assert np.all(np.abs(grey_dev) < np.abs(neg_dev) * 4.7 + 0.10)
    # the deviations run the way the negative's spacing runs: the
    # thin-over-nominal channels print dense, not the reverse
    assert np.all(grey_dev[[0, 2]] > 0) == np.all(neg_dev[[0, 2]] > 0)


def test_grey_timing_prints_the_film_grey_neutral():
    """The trim: grey_lights times THIS stock's own metered grey to
    the print aims - the solve must land them."""
    E = colour.normal_exposure(np.full((8, 8, 3), 1.0), "50d")
    lights = colour.grey_lights("50d", "2383", E=E)
    neg = colour.negative(np.full((1, 1, 3), 0.18), E, "50d",
                          grain=False)
    pr = colour.positive(colour.transmit(neg), "2383",
                         lights=lights, grain=False)[0, 0]
    aims = np.asarray(colour.PRINT_FILMS["2383"]["lad_aim"])
    assert float(np.abs(pr - aims).max()) < 2e-3


def test_red_subject_prints_red():
    """A red patch through the whole chain must come out of the
    print transmitting red: lowest Status A density in R."""
    lum = np.zeros((1, 1, 3))
    lum[..., 0] = 0.4
    E = colour.normal_exposure(np.full((8, 8, 3), 1.0), "50d")
    neg = colour.negative(lum, E, "50d", grain=False)
    lights = colour.lad_lights("50d", "2383")
    pr = colour.positive(colour.transmit(neg), "2383",
                         lights=lights, grain=False)[0, 0]
    assert pr[0] < pr[1] and pr[0] < pr[2]


def test_grain_areas_answer_the_rms_curves():
    """The wiring referee for the granularity-equivalent areas: a
    50d green layer developed to mid density, read the way the
    sheet reads it (48 um aperture), must give back sigma-D near
    the published 5/1000. The derivation used the same thinning
    machinery, so this catches wiring (pitch squared, wrong layer,
    lost kappa), not tautology."""
    rng_pitch = 6.0
    ap = 8  # 48 um aperture at 6 um pitch
    aim = colour.COLOUR_FILMS["50d"]["base"][1] + 1.0
    grey = np.full((1, 1, 3), 1.0)
    lo, hi = 1e-6, 1e6
    for _ in range(60):
        mid = float(np.sqrt(lo * hi))
        d = float(colour.negative(grey, mid, "50d",
                                  grain=False)[0, 0, 1])
        lo, hi = (mid, hi) if d < aim else (lo, mid)
    E = float(np.sqrt(lo * hi))
    field = colour.negative(np.full((1024, 1024, 3), 1.0), E, "50d",
                            pitch_um=rng_pitch, grain=True,
                            seed=7)[..., 1]
    n = 1024 // ap
    patches = field[:n * ap, :n * ap].reshape(n, ap, n, ap)
    sigma = float(patches.mean(axis=(1, 3)).std()) * 1000.0
    assert 2.5 < sigma < 10.0


def test_print_film_grain_refuses_by_name():
    """2383 publishes no rms curve; asking for print grain must
    refuse with the gap named, not invent a noise constant."""
    t = np.full((2, 2, 3), 0.1)
    with pytest.raises(ValueError, match="granularity-equivalent"):
        colour.positive(t, "2383", grain=True, pitch_um=6.0)
