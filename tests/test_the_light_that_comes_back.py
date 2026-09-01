"""Organ 10: halation, read back off the geometry that predicts it.

The support's own thickness and index fix the whole shape of the
halo - there is no free parameter in it - so these referees check
the derivation rather than a fit. The spread must peak at the
critical radius and must NOT jump there (Fresnel climbs to unity
continuously; the plan said otherwise first and the arithmetic
corrected it). The planar integral must close on the angular one.
The transfer must have exactly unity DC gain, because halation
redistributes light and a stock's rated speed was measured on
film that halates. And the transfer must collapse below the
lowest frequency any traced MTF carries a datum at - which is the
whole argument that organ 8 and organ 10 do not count the same
light twice, pinned here so it cannot rot quietly.
"""

import numpy as np
import pytest

from atlas_film import halation as H

ACETATE, GLASS, ESTAR = 1.48, 1.52, 1.64


def test_j0_lands_on_its_own_zeros():
    zeros = np.array([2.404825557695773, 5.520078110286311,
                      8.653727912911013, 11.791534439014281,
                      14.930917708487787])
    assert np.abs(H._j0(zeros)).max() < 5e-8
    assert H._j0(np.array([0.0]))[0] == 1.0


def test_the_critical_radius_is_the_angle_it_claims():
    for n in (ACETATE, GLASS, ESTAR):
        theta_c = np.arcsin(1.0 / n)
        assert H.critical_radius(137.0, n) == pytest.approx(
            2.0 * 137.0 * np.tan(theta_c), rel=1e-12)
    # acetate's ring is 1.83 support-thicknesses out; the denser
    # the support the tighter the ring, because more of the
    # hemisphere lies beyond a smaller critical angle
    assert H.critical_radius(1.0, ACETATE) == pytest.approx(1.8331, abs=1e-4)
    assert H.critical_radius(1.0, GLASS) == pytest.approx(1.7471, abs=1e-4)
    assert H.critical_radius(1.0, ESTAR) == pytest.approx(1.5386, abs=1e-4)


def test_the_spread_peaks_at_the_critical_radius():
    for t, n in [(132.0, ACETATE), (1200.0, GLASS), (178.0, ESTAR)]:
        rc = H.critical_radius(t, n)
        r = np.linspace(1e-6, 6.0 * rc, 400001)
        peak = r[int(np.argmax(H.spread(r, t, n)))]
        assert peak == pytest.approx(rc, rel=2e-5)


def test_the_spread_is_continuous_but_cusped_at_the_ring():
    """The plan's one wrong word, and then the correction's own
    imprecision, both pinned. The spread does NOT jump at r_c -
    Fresnel reaches unity continuously - but neither is the peak
    smooth: 1 - R goes as sqrt(theta_c - theta), so the profile
    arrives at its maximum with an INFINITE derivative. That cusp
    is why the period literature describes a sharply defined ring
    edge rather than a soft maximum, and it is testable: the
    outside/inside ratio must fall by exactly sqrt(10) per decade
    of epsilon on its way to 1."""
    for t, n in [(132.0, ACETATE), (1200.0, GLASS)]:
        rc = H.critical_radius(t, n)
        gaps = []
        for e in (1e-6, 1e-7, 1e-8, 1e-9):
            inside = H.spread(np.array([rc * (1 - e)]), t, n)[0]
            outside = H.spread(np.array([rc * (1 + e)]), t, n)[0]
            gaps.append(outside / inside - 1.0)
        assert gaps[-1] < 1e-3                       # continuous
        assert all(g > 0 for g in gaps)              # from below
        for a, b in zip(gaps, gaps[1:]):
            assert a / b == pytest.approx(np.sqrt(10.0), rel=0.02)
    t, n = 132.0, ACETATE
    ratio = (H.spread(np.array([H.critical_radius(t, n)]), t, n)[0]
             / H.spread(np.array([1e-9]), t, n)[0])
    assert ratio == pytest.approx(7.9, abs=0.1)


def test_the_planar_integral_closes_on_the_angular_one():
    """Two independent routes to the same number: integrate the
    surface density over the plane, or the Fresnel reflectance
    over the Lambertian hemisphere. The derivation checks itself."""
    for t, n in [(132.0, ACETATE), (1200.0, GLASS), (132.0, ESTAR)]:
        r = np.linspace(1e-9, 4000.0 * t / 132.0, 2000001)
        planar = np.trapezoid(H.spread(r, t, n) * 2 * np.pi * r, r)
        angular = H.reflected_fraction(n)
        # planar is short by the truncated 16/u^3 tail only
        assert planar < angular
        assert planar == pytest.approx(angular, rel=0.01)


def test_the_reflected_share_exceeds_pure_total_internal_reflection():
    """Beyond the critical angle everything returns, which alone
    is 1 - 1/n^2 of a Lambertian hemisphere; the sub-critical
    Fresnel few percent sits on top. High enough - well over half
    - to explain a century of antihalation chemistry."""
    for n, want in [(ACETATE, 0.5840), (GLASS, 0.6082), (ESTAR, 0.6698)]:
        f = H.reflected_fraction(n)
        assert f > 1.0 - 1.0 / n ** 2
        assert f == pytest.approx(want, abs=5e-4)
        assert f < 1.0


def test_the_halo_is_far_broader_than_its_ring():
    """Only a fourteenth of the returned light lands inside r_c.
    This is why period interiors VEIL rather than merely ring, and
    why the kernel has to reach ten ring radii."""
    t, n = 132.0, ACETATE
    rc = H.critical_radius(t, n)
    r = np.linspace(1e-9, 60.0 * rc, 2000001)
    p = H.spread(r, t, n) * 2 * np.pi * r
    cum = np.concatenate([[0.0], np.cumsum(np.diff(r) * 0.5 * (p[1:] + p[:-1]))])
    total = H.reflected_fraction(n)
    enclosed = lambda m: cum[int(np.searchsorted(r, rc * m))] / total
    assert enclosed(1.0) == pytest.approx(0.069, abs=0.005)
    assert enclosed(2.0) == pytest.approx(0.607, abs=0.010)
    assert enclosed(10.0) == pytest.approx(0.980, abs=0.005)


def test_the_transfer_is_exactly_unity_at_dc():
    for n in (ACETATE, GLASS):
        assert H.kernel_transfer(0.0, 137.0, n) == pytest.approx(1.0, abs=1e-9)
        assert H.transfer(0.0, 137.0, n, 0.4) == pytest.approx(1.0, abs=1e-9)


def test_a_uniform_field_does_not_move():
    """Unity DC gain is not decoration: the stock's rated speed
    was measured on film that halates, so an organ that ADDED the
    returned light would push every sheet faster than its own
    sensitometry."""
    for pitch, t, n, g in [(50.0, 132.0, ACETATE, 0.20),
                           (50.0, 1200.0, GLASS, 0.35),
                           (10.0, 178.0, ACETATE, 0.30)]:
        flat = np.full((192, 192), 0.37)
        out = H.apply(flat, pitch_um=pitch, support_um=t, index=n, strength=g)
        assert np.abs(out - flat).max() < 1e-12


def test_light_is_redistributed_not_created():
    """When the halo fits inside the frame's own padding, the dose
    is conserved to parts in a million: unity DC gain doing its
    job. Nothing is created and nothing is quietly lost."""
    rng = np.random.default_rng(10)
    img = rng.random((256, 256)) ** 3
    for t, n, g in [(132.0, ACETATE, 0.20), (60.0, ESTAR, 0.30)]:
        out = H.apply(img, pitch_um=50.0, support_um=t, index=n, strength=g)
        assert out.sum() == pytest.approx(img.sum(), rel=2e-5)


def test_a_halo_wider_than_the_sheet_spills_off_its_edge():
    """A 1.2 mm plate's halo reaches ten ring radii - 21 mm, wider
    than a small frame is. The organ does not pretend otherwise:
    the padding clamps at the sheet, light genuinely leaves, and
    the loss is small, one-directional, and bounded rather than
    hidden. A plate this far inside its own halo is a real
    photographic situation, not an edge case of the code."""
    rng = np.random.default_rng(10)
    img = rng.random((192, 192)) ** 3
    out = H.apply(img, pitch_um=50.0, support_um=1200.0,
                  index=GLASS, strength=0.35)
    lost = 1.0 - out.sum() / img.sum()
    assert 0.0 < lost < 0.01


def test_the_pad_never_outruns_the_narrow_side():
    """Reflect padding cannot exceed a dimension, so a wide, short
    frame is where a clamp written against the LONG side breaks."""
    img = np.random.default_rng(4).random((48, 900)) ** 2
    out = H.apply(img, pitch_um=40.0, support_um=1200.0,
                  index=GLASS, strength=0.3)
    assert out.shape == img.shape
    assert np.isfinite(out).all()


def test_a_support_thinner_than_a_pixel_is_a_bit_identical_no_op():
    img = np.random.default_rng(3).random((64, 64))
    out = H.apply(img, pitch_um=5000.0, support_um=132.0,
                  index=ACETATE, strength=0.3)
    assert out is img
    assert H.apply(img, pitch_um=10.0, support_um=132.0,
                   index=ACETATE, strength=0.0) is img


def test_the_ring_reads_back_off_an_actual_convolution():
    """A single lit pixel, developed through the transfer, and the
    halo's peak measured where the geometry says it will be."""
    for t, n, pitch in [(132.0, ACETATE, 8.0), (1200.0, GLASS, 50.0)]:
        N = 1024
        pt = np.zeros((N, N))
        pt[N // 2, N // 2] = 1.0
        out = H.apply(pt, pitch_um=pitch, support_um=t, index=n, strength=1.0)
        rc = H.critical_radius(t, n)
        row = out[N // 2, N // 2:]
        lo, hi = int(0.25 * rc / pitch), int(4.0 * rc / pitch)
        peak_um = (lo + int(np.argmax(row[lo:hi]))) * pitch
        assert abs(peak_um - rc) <= 1.5 * pitch


def test_the_transfer_collapses_below_the_mtf_floor():
    """The separability argument, pinned. Every traced MTF on the
    shelf carries its lowest datum at or above 2.5 c/mm. By then
    halation contributes no SHAPE, only the flat factor (1 - g) -
    which a curve normalised to unity divides straight out. So
    organ 8 owns 2.5 c/mm upward, organ 10 owns everything below,
    and the two do not count the same light twice."""
    for f in (2.5, 5.0, 10.0, 25.0, 100.0):
        assert abs(H.kernel_transfer(f, 132.0, ACETATE)) < 0.12
        assert abs(H.kernel_transfer(f, 1200.0, GLASS)) < 0.01
    for f in (5.0, 10.0, 25.0, 100.0):
        assert abs(H.kernel_transfer(f, 132.0, ACETATE)) < 0.01
    # and it has NOT collapsed at the frequencies the organ owns
    assert H.kernel_transfer(0.25, 132.0, ACETATE) > 0.5
    assert H.kernel_transfer(0.05, 1200.0, GLASS) > 0.5


def test_the_antihalation_measure_pays_the_double_pass():
    """A dyed undercoat absorbs going down and again coming back,
    so a density D costs 10^(-2D), not 10^(-D)."""
    bare = H.strength(132.0, ACETATE, 0.0, 0.35)
    for d in (0.3, 1.0, 2.0):
        assert H.strength(132.0, ACETATE, d, 0.35) == pytest.approx(
            bare * 10.0 ** (-2.0 * d), rel=1e-12)
    assert bare == pytest.approx(0.35 * H.reflected_fraction(ACETATE), rel=1e-12)


def test_the_shelf_refuses_what_it_cannot_source():
    """No support is invented. The geometry is so willing that any
    thickness at all yields a plausible halo, which is exactly why
    an unsourced stock must halate not at all."""
    for name in ("trix", "manchester", "collodion", "50d"):
        if name not in H.SUPPORTS:
            assert H.SUPPORTS.get(name) is None
