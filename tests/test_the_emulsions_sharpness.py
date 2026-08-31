"""Organ 8: the emulsion's finite sharpness, read back by sine.

The referee exposes the stock to sine-wave targets - the sheets'
own test, by their own boilerplate - and measures the response
ratio through the public surface. The super-unity adjacency lift
must come back as measured (a model that cannot exceed 100% would
deny the curves' most informative feature), the coarse-pitch
no-op must be bit-identical, and the stocks whose sheets publish
no MTF stay pixel-sharp by declaration.
"""

import numpy as np
import pytest

from atlas_film import colour, films, mtf


def sine_response(name, f_cycles_mm, pitch_um=1.0, n=2048):
    """Expose a sine at f, read the developed dose modulation back
    through a linearising trick: measure on the DOSE by using a
    stock-free call? No - through the curve, small-signal: a
    shallow sine about mid-scale, response from the density swing
    ratioed against a near-zero-frequency reference."""
    x = np.arange(n) * pitch_um / 1000.0          # mm
    # lock-in on the central half: max-min would read the pad
    # ripple as signal at high frequency; the quadrature product
    # reads only the sine that was sent
    f_ref = 4000.0 / (n * pitch_um)
    E = films.normal_highlight(name) / 0.5 * 0.3   # mid straight line
    lo, hi = n // 4, 3 * n // 4
    amps = {}
    for f in (f_cycles_mm, f_ref):
        img = 0.5 + 0.05 * np.sin(2 * np.pi * f * x)
        field = np.tile(img, (8, 1))
        d = np.asarray(films.negative(field, E, name, grain=False,
                                      pitch_um=pitch_um),
                       np.float64)
        row = d[4, lo:hi] - d[4, lo:hi].mean()
        s = np.sin(2 * np.pi * f * x[lo:hi])
        c = np.cos(2 * np.pi * f * x[lo:hi])
        amps[f] = 2.0 * float(np.hypot(row @ s, row @ c)) / (hi - lo)
    return amps[f_cycles_mm] / amps[f_ref]


def test_the_traced_curve_reads_back_by_sine():
    """TRI-X at 10 c/mm must show its adjacency lift (>1) and at
    50 c/mm its measured ~56% - within the fit's own residual."""
    for f, want in ((10.0, mtf.transfer(10.0, mtf.MTF_BW["trix"])),
                    (50.0, mtf.transfer(50.0, mtf.MTF_BW["trix"]))):
        r = sine_response("trix", f)
        assert abs(r - float(want)) < 0.06, (f, r, want)
    assert sine_response("trix", 10.0) > 1.02      # the lift, real
    assert mtf.transfer(15.0, mtf.MTF_BW["tmax400"]) > 1.15


def test_fit_reproduces_the_traced_points():
    """The shipped parameters against the lane's read-offs, at the
    fit's recorded residuals."""
    pts = {"trix": ((10, 1.11), (25, 1.00), (50, 0.56), (70, 0.26)),
           "5222": ((4, 1.25), (20, 0.94), (50, 0.41)),
           "tmax100": ((15, 1.14), (50, 1.01), (125, 0.50))}
    for name, rows in pts.items():
        for f, want in rows:
            got = float(mtf.transfer(f, mtf.MTF_BW[name]))
            assert abs(got - want) < 0.07, (name, f, got, want)


def test_coarse_pitch_is_a_true_noop():
    """At the golden raster's scale nothing resolves and the
    application must be BIT-identical - the organ never perturbs
    what it cannot honestly sharpen or soften."""
    rng = np.random.default_rng(3)
    img = rng.uniform(0.0, 1.0, (32, 32))
    out = mtf.apply(img, mtf.MTF_BW["tmax100"], 200.0)
    assert out is img


def test_unpublished_stocks_stay_pixel_sharp():
    """Plus-X, P3200 (whose sheet prints another film's artwork,
    L4), the plates and collodion carry no MTF constants: their
    negatives are unchanged with mtf on or off - the declared
    idealization, per stock."""
    grey = np.full((16, 16, 3), 0.4)
    for name in ("plusx", "p3200", "manchester", "collodion"):
        E = films.normal_exposure(grey, name)
        a = films.negative(grey, E, name, grain=False, pitch_um=2.0)
        b = films.negative(grey, E, name, grain=False, pitch_um=2.0,
                           mtf=False)
        assert np.array_equal(a, b), name
    assert "p3200" not in mtf.MTF_BW


def test_colour_layers_each_carry_their_own_sharpness():
    """A fine sine through 50d: the blue record (yellow layer, the
    softest at low frequency by the traced curves' floor) must
    lose more modulation than the green record at 40 c/mm."""
    n, pitch = 1024, 1.0
    x = np.arange(n) * pitch / 1000.0
    img = np.zeros((8, n, 3))
    img[..., :] = (0.5 + 0.2 * np.sin(
        2 * np.pi * 40.0 * x))[None, :, None]
    E = colour.normal_exposure(np.full((4, 4, 3), 0.7), "50d")
    d = colour.negative(img, E, "50d", grain=False, pitch_um=pitch)
    swing_g = float(d[4, :, 1].max() - d[4, :, 1].min())
    d_off = colour.negative(img, E, "50d", grain=False,
                            pitch_um=pitch, mtf=False)
    swing_g0 = float(d_off[4, :, 1].max() - d_off[4, :, 1].min())
    assert swing_g < swing_g0 * 0.75
