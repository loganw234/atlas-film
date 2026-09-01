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

from atlas_film import films, halation as H

ACETATE = H.INDICES["acetate"]          # 1.4925, Kodak's own range
GLASS = H.INDICES["plate-glass-blue"]    # 1.5290, blue: the plates are blue-eyed
ESTAR = 1.64                            # in-plane PET; refused as a support


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
    assert H.critical_radius(1.0, ACETATE) == pytest.approx(1.8051, abs=1e-4)
    assert H.critical_radius(1.0, GLASS) == pytest.approx(1.7291, abs=1e-4)
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
    assert ratio == pytest.approx(7.8, abs=0.1)


def test_the_planar_integral_closes_on_the_angular_one():
    """Two independent routes to the same number: integrate the
    surface density over the plane, or the Fresnel reflectance
    over the Lambertian hemisphere. The derivation checks itself."""
    for t, n in [(132.0, ACETATE), (1200.0, GLASS)]:
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
    for n, want in [(ACETATE, 0.5464), (GLASS, 0.6014)]:
        f = H.reflected_fraction(n)
        # the pure-TIR share of a hemisphere, less what the
        # emulsion/support interface turns back before it crosses
        assert f > (1.0 - 1.0 / n ** 2) * (n / H.EMULSION_INDEX) ** 2
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
    assert enclosed(1.0) == pytest.approx(0.070, abs=0.005)
    assert enclosed(2.0) == pytest.approx(0.612, abs=0.010)
    assert enclosed(10.0) == pytest.approx(0.986, abs=0.005)


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
    for t, n, g in [(132.0, ACETATE, 0.20), (60.0, GLASS, 0.30)]:
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
    so a density D costs 10^(-2D), not 10^(-D). The only source
    that states this arithmetic is a BBC television report on the
    CRT analogue - the photographic literature describes the
    double pass and never algebraises it, and the citation says
    so rather than borrowing plausibility from a film datasheet."""
    bare = H.ceiling(ACETATE, 0.0)
    for d in (0.3, 1.0, 2.0):
        assert H.ceiling(ACETATE, d) == pytest.approx(
            bare * 10.0 ** (-2.0 * d), rel=1e-12)
    assert bare == pytest.approx(H.reflected_fraction(ACETATE), rel=1e-12)


def test_the_gain_is_a_fraction_of_a_sourced_ceiling():
    """The strength is the one quantity nobody published, so it is
    the operator's - but it is measured against a bound the record
    DOES fix, not against nothing. Outside [0, 1] it would return
    more light than the support reflects, and refuses."""
    for gain in (0.0, 0.25, 1.0):
        _, index, g = H.for_stock("manchester", gain)
        assert g == pytest.approx(gain * H.ceiling(index, 0.0), rel=1e-12)
    for bad in (-0.01, 1.5, 3.0):
        with pytest.raises(ValueError, match="fraction of the SOURCED CEILING"):
            H.for_stock("manchester", bad)


def test_the_plates_halate_and_everything_else_says_why_not():
    """The organ's whole point, and its whole honesty in one test.
    The 1890 plates are the only stocks whose antihalation state
    is POSITIVELY sourced - they had none, and the trade record
    dates the change to 1900-1912 - so they are the only ones that
    halate. Every other stock raises, and no two reasons are the
    same boilerplate."""
    assert set(H.SUPPORTS) == {"manchester", "hd22"}
    reasons = set()
    for name in sorted(films.FILMS):
        if name in H.SUPPORTS:
            continue
        with pytest.raises(ValueError, match="does not halate"):
            H.for_stock(name, 0.5)
        reasons.add(H.REFUSALS.get(name, ""))
    assert "" not in reasons, "a stock refuses without a stated reason"
    assert len(reasons) >= 6, "the refusals have collapsed into boilerplate"
    # and the one that refuses on PHYSICS rather than on silence
    assert "BY MECHANISM" in H.REFUSALS["collodion"]


def test_the_organ_is_off_until_asked():
    """Default OFF, and bit-identically so - not approximately
    unchanged. Every negative made before organ 10 existed still
    develops to the same digits, which is what lets 42 golden
    prints stay pinned through the landing."""
    rng = np.random.default_rng(7)
    img = rng.random((96, 96)) * 0.6 + 0.2
    for name in ("manchester", "hd22", "trix", "collodion"):
        E = films.normal_highlight(name) / 0.5 * 0.3
        base = films.negative(img, E, name, pitch_um=12.0, grain=False)
        none = films.negative(img, E, name, pitch_um=12.0, grain=False,
                              halation=None)
        assert np.array_equal(base, none), name
    # and zero gain is a no-op even where the stock DOES halate
    E = films.normal_highlight("manchester") / 0.5 * 0.3
    assert np.array_equal(
        films.negative(img, E, "manchester", pitch_um=12.0, grain=False),
        films.negative(img, E, "manchester", pitch_um=12.0, grain=False,
                       halation=0.0))


def test_a_plate_veils_more_than_it_rings():
    """Wall 1912: halation is "present in every photograph if the
    plate is not backed, destroying its quality and ruining its
    rendering of gradation" - not merely a ring around windows.
    At a pitch where a 1.3 mm plate's 2.2 mm halo is far wider
    than the frame, the visible effect must be exactly that: a
    global loss of tonal separation, with the dose conserved."""
    rng = np.random.default_rng(11)
    img = rng.random((160, 160)) ** 2 * 0.8 + 0.05
    E = films.normal_highlight("manchester") / 0.5 * 0.3
    plain = films.negative(img, E, "manchester", pitch_um=60.0, grain=False)
    veiled = films.negative(img, E, "manchester", pitch_um=60.0,
                            grain=False, halation=0.8)
    assert veiled.std() < plain.std()
    assert veiled.max() - veiled.min() < plain.max() - plain.min()
