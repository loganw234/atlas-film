"""Organ 5: development as a contrast dial, and the wet plate.

The CONTRAST tables are lane I's vector-exact traces of the
sheets' own CI figures; ci= displaces the curve along them and
refuses beyond them. Collodion lands with its traced 1998 curve
and refuses the dial by MECHANISM - development time is not a
contrast control there.
"""

import numpy as np
import pytest

from atlas_film import films

# the traced collodion points (lane I14), shifted to the declared
# absolute axis (+0.10: net D 0.1 at the ISO-0.1 speed point)
COLLODION_TRACE = list(zip(
    (0.80, 0.90, 1.00, 1.10, 1.20, 1.30, 1.40, 1.50, 1.60, 1.70,
     1.80, 1.90, 2.00, 2.10, 2.30, 2.50, 2.70, 2.90, 3.10),
    (0.07, 0.10, 0.17, 0.30, 0.44, 0.57, 0.70, 0.81, 0.91, 0.96,
     1.07, 1.13, 1.18, 1.22, 1.30, 1.38, 1.43, 1.49, 1.57)))


def test_the_tables_give_back_the_printed_rows():
    """Exact at the sheets' own rows: TRI-X's 6 min lands the
    confirmed 0.554; 5222's PRINTED gammas come back verbatim."""
    assert films.contrast_at("trix", 6.0) == 0.554
    assert films.contrast_at("5222", 6.5) == 0.66
    assert films.contrast_at("5222", 12.0) == 1.05
    assert abs(films.minutes_for("trix", 0.673) - 8.0) < 1e-9
    assert abs(films.minutes_for("tmax400", 0.606) - 8.0) < 1e-9


def test_normal_ci_is_the_identity():
    grey = np.full((6, 6, 3), 0.3)
    E = films.normal_exposure(grey, "trix")
    a = films.negative(grey, E, "trix", pitch_um=25.0, seed=2)
    b = films.negative(grey, E, "trix", pitch_um=25.0, seed=2,
                       ci=0.554)
    assert np.array_equal(a, b)


def test_pushing_steepens_by_the_ci_ratio():
    """Developed to the top of TRI-X's traced span, the straight
    line steepens by exactly ci/normal - the H&D scaling the
    modern tables inherit."""
    xx = np.array([-2.0, -1.0])
    d0 = films.characteristic(xx, "trix")
    d1 = films.characteristic(xx, "trix", ci=0.856)
    slope0 = float(d0[1] - d0[0])
    slope1 = float(d1[1] - d1[0])
    assert abs(slope1 / slope0 - 0.856 / 0.554) < 0.02


def test_the_span_is_the_law():
    with pytest.raises(ValueError, match="silent at"):
        films.characteristic(0.0, "trix", ci=1.0)
    with pytest.raises(ValueError, match="silent at"):
        films.characteristic(0.0, "tmax100", ci=0.45)
    with pytest.raises(ValueError, match="intensification"):
        films.characteristic(0.0, "collodion", ci=0.9)
    with pytest.raises(ValueError, match="capacity"):
        films.characteristic(0.0, "manchester", ci=1.2)


def test_collodion_reproduces_its_traced_curve():
    x = np.array([p[0] for p in COLLODION_TRACE])
    want = np.array([p[1] for p in COLLODION_TRACE])
    got = films.characteristic(x, "collodion")
    assert float(np.abs(got - want).max()) < 1e-12


def test_collodion_is_blue_blind_and_grainy():
    red = np.zeros((32, 32, 3))
    red[..., 0] = 0.8
    blue = np.full((32, 32, 3), 0.8)
    E = films.normal_exposure(blue, "collodion")
    d_red = films.negative(red, E, "collodion", grain=False)
    assert float(d_red.max()) < 0.08          # bare glass
    d = films.negative(blue, E, "collodion", pitch_um=2.0, seed=5)
    assert float(d.std()) > 0.003
    assert np.array_equal(
        d, films.negative(blue, E, "collodion", pitch_um=2.0,
                          seed=5))


def test_collodion_refuses_the_clock():
    with pytest.raises(ValueError, match="no reciprocity table"):
        films.reciprocity("collodion", 15.0)


# ---------------------------------------- the intensifier's bath

def test_the_bath_reproduces_both_printed_endpoints():
    """One factor, two printed numbers (I17): the 1:10 Dmax ratio
    2.6/1.57 must also carry the gradient 0.85 to the printed 1.37
    within the source's own consistency (3%)."""
    f = films.INTENSIFIERS["1:10"]
    assert abs(f * 1.57 - 2.6) < 1e-12
    assert abs(f * 0.85 - 1.37) < 0.05
    x = np.array([1.1, 1.3])
    d0 = films.characteristic(x, "collodion")
    d1 = films.negative(10.0 ** x, 1.0, "collodion", grain=False,
                        intensify="1:10")
    assert np.allclose(d1, d0 * f, atol=1e-6)


def test_intensification_amplifies_the_grain_it_touches():
    """The bath multiplies developed silver, so grain sigma scales
    by the same factor - I17's 'higher granularity' as an emergent
    prediction, not a dial."""
    blue = np.full((96, 96, 3), 0.5)
    E = films.normal_exposure(blue, "collodion")
    a = films.negative(blue, E, "collodion", pitch_um=2.0, seed=4)
    b = films.negative(blue, E, "collodion", pitch_um=2.0, seed=4,
                       intensify="1:10")
    f = films.INTENSIFIERS["1:10"]
    assert np.allclose(b, a * np.float32(f), rtol=1e-5)
    assert float(b.std()) > float(a.std()) * 1.5


def test_the_bath_refuses_what_it_cannot_hold():
    grey = np.full((2, 2, 3), 0.3)
    with pytest.raises(ValueError, match="shoulder"):
        films.negative(grey, 1.0, "collodion", grain=False,
                       intensify="1:5")
    with pytest.raises(ValueError, match="mercury"):
        films.negative(grey, 1.0, "manchester", grain=False,
                       intensify="1:10")
    with pytest.raises(ValueError, match="sourced recipe"):
        films.negative(grey, 1.0, "collodion", grain=False,
                       intensify="1:7")
