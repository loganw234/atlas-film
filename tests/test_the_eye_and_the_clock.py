"""Organ 4 (spectral response) and organ 3 (reciprocity).

The eye: each stock sees the aerial image through its own declared
band projection - the four Kodak stocks flat panchromatic, the 1890
plates blue-blind (early-plates D9/D13). The clock: each stock
compensates long and short exposures by its own sheet's table
(FILM-N13/N14) and refuses beyond it.

The plates bring their own referee: H&D's printed formulas (D2)
read gamma and inertia off the MODEL's curve, and hd22 must give
back the 1.176 H&D themselves printed - the fit's validator, not
its input.
"""

import numpy as np
import pytest

from atlas_film import films

# H&D's own printed tables (early-plates D1/D5): E in
# candle-metre-seconds ~ lux-seconds (D3, declared), D net of fog
MANCHESTER_1890 = [
    (0.625, .045), (1.25, .055), (2.5, .085), (5, .175), (10, .250),
    (20, .460), (40, .755), (80, 1.010), (160, 1.270), (320, 1.555),
    (640, 1.885), (1280, 2.088), (2560, 2.262), (5120, 2.352)]
HD22_1890 = [
    (1, .060), (2, .160), (4, .340), (8, .500), (16, .715),
    (32, .940), (64, 1.345), (128, 1.875), (256, 2.290),
    (512, 2.535), (1024, 2.985), (2048, 3.115), (4096, 3.280),
    (8192, 3.405)]


# ------------------------------------------------------ the eye

def test_a_grey_scene_meters_and_develops_as_before():
    """The flat pan projection with sum-one weights leaves a
    neutral scene EXACTLY where the scalar path puts it - organ 4
    changes what colour does, not what grey does."""
    rgb = np.full((8, 8, 3), 0.4)
    flat = np.full((8, 8), 0.4)
    assert films.normal_exposure(rgb, "trix") == \
        films.normal_exposure(flat, "trix")
    E = films.normal_exposure(flat, "trix")
    a = films.negative(rgb, E, "trix", pitch_um=25.0, seed=3)
    b = films.negative(flat, E, "trix", pitch_um=25.0, seed=3)
    assert np.array_equal(a, b)


def test_the_plate_is_blue_blind():
    """A pure red patch leaves an 1890 plate at its fog floor while
    the same patch exposes a panchromatic stock - Hardwich's vase
    of scarlet flowers, reproduced (D13)."""
    red = np.zeros((4, 4, 3))
    red[..., 0] = 0.8
    blue = np.zeros((4, 4, 3))
    blue[..., 2] = 0.8
    E = films.normal_exposure(np.full((4, 4, 3), 0.8), "manchester")
    d_red = films.negative(red, E, "manchester", grain=False)
    d_blue = films.negative(blue, E, "manchester", grain=False)
    assert float(d_red.max()) < 0.02      # fog=0: bare glass
    assert float(d_blue.min()) > 1.0      # the blue patch prints
    E_pan = films.normal_exposure(np.full((4, 4, 3), 0.8), "trix")
    d_pan = films.negative(red, E_pan, "trix", grain=False)
    assert float(d_pan.min()) > 1.0       # pan sees the scarlet


def test_every_stock_declares_its_eye():
    for name, st in films.FILMS.items():
        s = np.asarray(st["sens"], np.float64)
        assert s.shape == (3,) and abs(float(s.sum()) - 1.0) < 1e-12, \
            name


# ------------------------------------------- the plates' curves

@pytest.mark.parametrize("name,table,tol", [
    ("manchester", MANCHESTER_1890, 0.06),
    ("hd22", HD22_1890, 0.10)])
def test_the_plate_reproduces_hd_1890(name, table, tol):
    """The model's curve against H&D's printed table - tolerance is
    the 1890 densitometry's own (stated errors 2.4-5%, and H&D's
    straight-line theory deviates from their own table more)."""
    x = np.log10([p[0] for p in table])
    want = np.array([p[1] for p in table])
    got = films.characteristic(x, name)
    assert float(np.abs(got - want).max()) < tol


def test_hd_referee_reads_the_printed_gamma_back():
    """H&D's own formulas (D2) off the MODEL's curve. hd22's chord
    must give back the gamma H&D printed (1.176) - the validator -
    and its intercept must sit inside the 1890 data's own intercept
    scatter. Manchester lands in the dossier's computed band with
    S = 34/i near the plate's 5.6."""
    xx = np.linspace(-0.6, 4.2, 4001)

    def chord(name, e1, e2):
        lo, hi = np.log10(e1), np.log10(e2)
        d1, d2 = np.interp([lo, hi], xx,
                           films.characteristic(xx, name))
        g = (d2 - d1) / (hi - lo)
        logi = (d2 * lo - d1 * hi) / (d2 - d1)
        return g, logi

    g, logi = chord("hd22", 8, 256)
    assert abs(g - 1.176) < 0.03
    assert abs(logi - 0.579) < 0.12
    g, logi = chord("manchester", 20, 320)
    assert 0.87 < g < 0.93
    assert abs(logi - np.log10(6.12)) < 0.04
    assert 5.0 < 34.0 / 10 ** logi < 6.4       # H&D speed (D4)


def test_the_plate_grain_is_bracketed_but_counts():
    """The bracketed 0.196 um2 (D19's recorded silence, flagged at
    the constant) still behaves as a crystal field: grain present,
    seeded, and refusing pitches under the particle."""
    blue = np.full((64, 64, 3), 0.5)
    E = films.normal_exposure(blue, "manchester")
    d = films.negative(blue, E, "manchester", pitch_um=2.0, seed=1)
    assert float(d.std()) > 0.005
    assert np.array_equal(
        d, films.negative(blue, E, "manchester", pitch_um=2.0,
                          seed=1))
    with pytest.raises(ValueError, match="crystal|particle|floor"):
        films.negative(blue, E, "manchester", pitch_um=0.3, seed=1)


# ---------------------------------------------------- the clock

def test_reciprocity_reads_each_sheet():
    assert films.reciprocity("trix", 1.0) == 1.0
    assert films.reciprocity("trix", 100.0) == 3.0
    assert films.reciprocity("trix", 0.01) == 0.0
    assert abs(films.reciprocity("trix", 10 ** 0.5) - 1.5) < 1e-9
    assert films.reciprocity("tmax400", 1.0) == 0.0
    assert abs(films.reciprocity("tmax400", 100.0) - 1.5) < 1e-9
    assert films.reciprocity("5222", 0.5) == 0.0
    assert films.reciprocity("manchester", 40.0) == 0.0


def test_silence_is_not_zero():
    """5222's sheet stops at one second; TRI-X's at 100; asking
    beyond refuses by name rather than extrapolating."""
    for name, t in (("5222", 2.0), ("trix", 1000.0),
                    ("trix", 1e-6), ("tmax100", 500.0)):
        with pytest.raises(ValueError, match="silen"):
            films.reciprocity(name, t)


def test_the_meter_and_the_film_cancel():
    """The meter raises E by the sheet's compensation and the film
    discounts by the same factor: a 100 s TRI-X frame metered FOR
    100 s develops exactly like the instantaneous frame - which is
    the whole point of the sheet's table."""
    grey = np.full((6, 6, 3), 0.3)
    E0 = films.normal_exposure(grey, "trix")
    E100 = films.normal_exposure(grey, "trix", t=100.0)
    assert abs(E100 / E0 - 8.0) < 1e-9      # +3 stops
    d0 = films.negative(grey, E0, "trix", grain=False)
    d100 = films.negative(grey, E100, "trix", grain=False, t=100.0)
    assert np.allclose(d0, d100, atol=1e-9)


def test_uncompensated_long_exposure_underexposes():
    """The same E given for 100 s that was metered for an instant
    loses three stops on TRI-X, and on the straight line three
    stops cost 3 x 0.301 x gamma ~ 0.53 D - the drop must be that,
    not a vibe."""
    grey = np.full((6, 6, 3), 0.3)
    E0 = films.normal_exposure(grey, "trix")
    d = films.negative(grey, E0, "trix", grain=False, t=100.0)
    d0 = films.negative(grey, E0, "trix", grain=False)
    drop = float(d0.min()) - float(d.max())
    aim = 3 * 0.301 * films.FILMS["trix"]["gamma"]
    assert abs(drop - aim) < 0.08
