"""The camera stock's referee: the negative is a sheet, fog and all.

Organ 6 (PLAN.md): a stock at the film plane, exposed and developed
into a density field by the same emulsion physics the print grain
earned. These tests run the mechanism on an INJECTED stock with a
complete curve, so they are independent of which real films carry
sourced constants - the real stocks' curves are the sensitometry
lane's, and until it lands they refuse (tested here too, against a
stock stripped of its keys, so the test survives the constants
landing).

The claims proper to the NEGATIVE, beyond what the emulsion suite
already holds:

- fog counts crystals: an unexposed frame develops to the stock's
  fog with Saunders' own fluctuation about it - the rebate of a
  real negative is grainy, and so is this one's;
- the fog floor is a floor: exposure can only add density;
- transmit is exact: 10^-D, the printing light;
- a stored film sheet prints the identical negative.
"""
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from atlas_film import emulsion, films
from atlas_film.processes import KAPPA

TEST_STOCK = dict(fog=0.2, dmax=2.0, gamma=0.6, ht=-2.0, wt=0.2,
                  ws=0.35, grain_um2=1.0)
PITCH = 2.0


@pytest.fixture
def stocked(monkeypatch):
    monkeypatch.setitem(films.FILMS, "teststock", dict(TEST_STOCK))


def test_the_shelf_refuses_an_unknown_stock():
    with pytest.raises(ValueError, match="shelf"):
        films.negative(np.zeros((2, 2)), 1.0, "kodachrome",
                       grain=False)


def test_a_stock_without_its_curve_refuses_by_name(monkeypatch):
    """The real films ship grain areas now and curves only when the
    sensitometry lane lands; in between they refuse rather than
    carrying an invented curve under a real film's name. Tested on a
    stripped stock so it holds after the constants arrive."""
    monkeypatch.setitem(films.FILMS, "bare", dict(grain_um2=1.0))
    with pytest.raises(ValueError, match="sensitometry"):
        films.negative(np.zeros((2, 2)), 1.0, "bare", grain=False)


def test_grain_needs_the_films_pitch(stocked):
    with pytest.raises(ValueError, match="pitch_um"):
        films.negative(np.zeros((4, 4)), 1.0, "teststock")


def test_the_unexposed_frame_develops_to_fog(stocked):
    """The mean claim: zero light is not zero density. The smooth
    shoulder shaves ~2e-3 off the exact floor (the softplus cap's
    tail), which is the family's declared smoothness, not a leak."""
    d = films.negative(np.zeros((8, 8)), 1.0, "teststock", grain=False)
    assert d == pytest.approx(TEST_STOCK["fog"], abs=5e-3)


def test_the_rebate_is_grainy(stocked):
    """The counting claim: fog is developed CRYSTALS, so an unexposed
    frame fluctuates about fog by Saunders' law - sqrt(kappa*a*D/A)
    at D = fog - exactly as a real negative's rebate does."""
    d = films.negative(np.zeros((256, 256)), 1.0, "teststock",
                       pitch_um=PITCH, seed=3)
    want = float(np.sqrt(KAPPA * TEST_STOCK["grain_um2"]
                         * TEST_STOCK["fog"] / PITCH ** 2))
    assert float(d.mean()) == pytest.approx(TEST_STOCK["fog"], rel=0.02)
    assert float(np.std(d, ddof=1)) == pytest.approx(want, rel=0.05)


def test_exposure_only_adds_to_the_fog_floor(stocked):
    """On one sheet, light develops crystals fog left undeveloped -
    never the reverse: density is pixelwise monotone from the rebate
    up through the exposure series."""
    lums = (0.0, 0.1, 0.4, 1.5, 50.0)
    ds = [films.negative(np.full((96, 96), v, np.float32), 1.0,
                         "teststock", pitch_um=PITCH, seed=7)
          for v in lums]
    for lo, hi in zip(ds, ds[1:]):
        assert np.all(hi - lo > -1e-6), \
            "a crystal undeveloped as the light rose"


def test_transmit_is_the_printing_light(stocked):
    d = np.array([[0.0, 1.0], [2.0, 3.0]], np.float32)
    t = films.transmit(d)
    assert np.allclose(t, [[1.0, 0.1], [0.01, 0.001]], rtol=1e-5)


def test_a_stored_film_sheet_prints_the_same_negative(stocked):
    """The film-stock claim reaches the camera: coat once, expose
    against the stored crystals, bit-identical to the seeded path."""
    lum = np.linspace(0.0, 3.0, 96 * 96).reshape(96, 96) \
        .astype(np.float32)
    K, thr = emulsion.coat(lum.size, TEST_STOCK["dmax"],
                           TEST_STOCK["grain_um2"], PITCH, seed=13)
    a = films.negative(lum, 1.0, "teststock", pitch_um=PITCH, seed=13)
    b = films.negative(lum, 1.0, "teststock", pitch_um=PITCH,
                       sheet=(K, thr))
    assert np.array_equal(a, b)


def test_the_shipped_stocks_carry_their_sourced_grain():
    """The dossier's inversions, as constants: published rms
    granularity through a = A*sigma^2/(kappa*D) at the 48um / D 1.0
    convention. Derived here from the published integers themselves,
    so a drifted constant fails against its own source."""
    A = np.pi * 24.0 ** 2
    for name, rms in (("trix", 17), ("5222", 14), ("tmax100", 8),
                      ("tmax400", 10)):
        a = A * (rms / 1000.0) ** 2 / (KAPPA * 1.0)
        assert films.FILMS[name]["grain_um2"] == pytest.approx(
            a, abs=5e-4), name


# the datasheets' own characteristic curves, read off their figures
# at the named normal development (film-stocks dossier: N4 trix
# T-MAX dev 6 min, N9 5222 D-96 6.5 min with gamma 0.66 printed on
# the curve, N7 tmax100 T-MAX RS 8 min, N8 tmax400 D-76 8 min)
DATASHEET = {
    "trix": [(-3.0, 0.36), (-2.5, 0.50), (-2.0, 0.76), (-1.5, 1.04),
             (-1.0, 1.35), (-0.5, 1.65), (0.35, 2.09)],
    "5222": [(-2.0, 0.64), (-1.5, 0.96), (-1.0, 1.30), (-0.5, 1.60)],
    "tmax100": [(-2.5, 0.23), (-2.0, 0.35), (-1.5, 0.61),
                (-1.0, 0.87), (0.0, 1.37), (0.65, 1.75)],
    "tmax400": [(-2.5, 0.38), (-2.0, 0.61), (-1.5, 0.90),
                (-1.0, 1.24), (-0.5, 1.64), (0.0, 1.97),
                (0.35, 2.15)],
}


def test_the_shipped_curves_reproduce_their_datasheets():
    """Every stock's curve against its own sheet's read-off table,
    point by point: the largest residual of any fit is 0.028 D, and
    the read-off itself is validated at +/-0.04 by Double-X's
    printed gammas (dossier N9). A constant that drifts fails
    against the figure it came from."""
    for name, pts in DATASHEET.items():
        x = np.array([p[0] for p in pts])
        want = np.array([p[1] for p in pts])
        got = films.characteristic(x, name)
        assert np.abs(got - want).max() < 0.035, (
            name, np.abs(got - want).max())


def test_the_printed_aims_emerge_unfit():
    """Kodak's own numbers, none of them inputs to the fit: Tri-X's
    chord (the datasheet's own CI cross-check, N4) against the
    PRINTED aim 0.56 (N1); 5222's straight-line slope against the
    gamma 0.66 PRINTED beside its curve (N9)."""
    xx = np.linspace(-4.5, 1.0, 2000)
    tri = films.characteristic(xx, "trix")
    chord = (np.interp(-0.5, xx, tri) - np.interp(-2.5, xx, tri)) / 2.0
    assert chord == pytest.approx(0.56, abs=0.02)
    dx = films.characteristic(np.array([-1.5, -1.0]), "5222")
    assert (dx[1] - dx[0]) / 0.5 == pytest.approx(0.66, abs=0.03)


def test_the_speed_points_sit_where_iso_puts_them():
    """ISO 6's criterion (N11): the speed point is 0.10 above
    base+fog at H_m = 0.8/S. The datasheet axes are absolute, so
    each stock's curve must put that point within a third of a stop
    of its rated speed - unfit, all four."""
    xx = np.linspace(-4.5, 1.0, 4001)
    for name, rated in (("trix", 400), ("5222", 250),
                        ("tmax100", 100), ("tmax400", 400)):
        d = films.characteristic(xx, name)
        hm = float(np.interp(films.FILMS[name]["fog"] + 0.10, d, xx))
        want = float(np.log10(0.8 / rated))
        assert abs(hm - want) < 0.1, (name, hm, want)


def test_the_normal_highlight_is_on_the_curve():
    """H-740's latitude construction (N12): highlights land ~1.1 D
    above fog. Tri-X's must therefore sit near 0.13 lux-seconds -
    on the absolute axis, a real exposure."""
    h = films.normal_highlight("trix")
    assert 0.05 < h < 0.5
    d = films.characteristic(np.array([np.log10(h)]), "trix")
    assert d[0] == pytest.approx(films.FILMS["trix"]["fog"] + 1.10,
                                 abs=0.02)
