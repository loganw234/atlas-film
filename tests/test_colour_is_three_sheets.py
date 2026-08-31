"""The colour assembly's referee: three sheets, one mask, LAD.

Organ 7's mechanism, tested on an INJECTED stock so it stands while
the FILM-C and FILM-P lanes deliver the real constants - which the
empty shelves refuse by name until then (also tested). The real
stocks' own referee - reassembling the datasheets' published
neutral-sweep curves - lands with the constants.
"""
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from atlas_film import colour

FAKE = dict(
    layers=(
        dict(gamma=0.6, ht=-2.0, wt=0.2, ws=0.35, span=2.0,
             sens=(1.0, 0.05, 0.0), dye=(1.0, 0.15, 0.05),
             grain_um2=1.0),
        dict(gamma=0.6, ht=-2.0, wt=0.2, ws=0.35, span=2.2,
             sens=(0.05, 1.0, 0.05), dye=(0.2, 1.0, 0.1),
             grain_um2=1.2),
        dict(gamma=0.6, ht=-2.1, wt=0.2, ws=0.35, span=2.4,
             sens=(0.0, 0.05, 1.0), dye=(0.05, 0.3, 1.0),
             grain_um2=1.5),
    ),
    base=(0.20, 0.60, 0.90),          # an orange mask: B densest
)
FAKE_PRINT = dict(
    layers=(
        dict(gamma=2.4, ht=-1.2, wt=0.15, ws=0.4, span=3.6,
             sens=(1.0, 0.02, 0.0), dye=(1.0, 0.1, 0.03),
             grain_um2=0.5),
        dict(gamma=2.5, ht=-1.0, wt=0.15, ws=0.4, span=3.7,
             sens=(0.02, 1.0, 0.02), dye=(0.1, 1.0, 0.08),
             grain_um2=0.5),
        dict(gamma=2.6, ht=-0.8, wt=0.15, ws=0.4, span=3.8,
             sens=(0.0, 0.02, 1.0), dye=(0.03, 0.2, 1.0),
             grain_um2=0.5),
    ),
    base=(0.06, 0.07, 0.08),
    lad_aim=(1.09, 1.06, 1.03),
)


@pytest.fixture
def shelf(monkeypatch):
    monkeypatch.setitem(colour.COLOUR_FILMS, "fakecolour", FAKE)
    monkeypatch.setitem(colour.PRINT_FILMS, "fakeprint", FAKE_PRINT)
    monkeypatch.setattr(colour, "LAD_NEGATIVE", (0.80, 1.20, 1.60))


def test_the_shelves_refuse_what_they_do_not_hold(monkeypatch):
    """An unknown stock refuses by listing the shelf; an EMPTY shelf
    refuses by naming the lane that owes it (the state the module
    shipped in before the constants landed)."""
    rgb = np.zeros((2, 2, 3), np.float32)
    with pytest.raises(ValueError, match="50d"):
        colour.negative(rgb, 1.0, "portra", grain=False)
    with pytest.raises(ValueError, match="2383"):
        colour.positive(rgb, "ra4paper", grain=False)
    monkeypatch.setattr(colour, "COLOUR_FILMS", {})
    monkeypatch.setattr(colour, "LAD_NEGATIVE", None)
    with pytest.raises(ValueError, match="FILM-C"):
        colour.negative(rgb, 1.0, "5219", grain=False)
    with pytest.raises(ValueError, match="LAD"):
        colour.lad_lights("5219", "2383")


def test_the_unexposed_negative_is_the_mask(shelf):
    """Zero light develops nothing but the base floors - which for a
    colour negative ARE the orange mask, densest in blue: on a light
    table the rebate transmits orange, and here it must too."""
    d = colour.negative(np.zeros((4, 4, 3), np.float32), 1.0,
                        "fakecolour", grain=False)
    assert np.allclose(d[0, 0], FAKE["base"], atol=6e-3)
    t = colour.transmit(d)[0, 0]
    assert t[0] > t[1] > t[2], "the rebate does not transmit orange"


def test_a_red_subject_stains_the_channels_by_the_dye_matrix(shelf):
    """Colour through the assembly: red light exposes the
    red-sensitive layer, whose dye stains R most - and the coupling
    into G and B follows the dye triple, not zero. The negative of a
    red subject is dense in R above its mask and nearly quiet
    elsewhere."""
    rgb = np.zeros((4, 4, 3), np.float32)
    rgb[...] = (0.5, 0.0, 0.0)
    d = colour.negative(rgb, 1.0, "fakecolour", grain=False)
    rise = d[0, 0] - np.asarray(FAKE["base"])
    # R rises most; G rises genuinely (the green layer sees 5% of
    # red light AND cyan dye stains G 15% - coupling is the point,
    # not a defect); B least. The ordering and the R dominance are
    # the claims, not an invented ratio.
    assert rise[0] > rise[1] > rise[2] > 0
    assert rise[0] > 2 * rise[1]
    assert rise[0] > 8 * rise[2]


def test_each_layer_is_its_own_sheet(shelf):
    """Three crystal fields, three seed streams: the grain of the
    R-driven layer must not repeat in the G-driven one - a colour
    print's chromatic grain exists because the layers fluctuate
    independently. (This referees the CRYSTAL FIELDS on a fake
    stock with no interimage constants; the real stocks' MEAN
    responses deliberately talk through the organ-9 DIR coupling -
    grain streams stay independent even then, because the
    inhibitor couples development levels, not the per-crystal
    luck.)"""
    rgb = np.full((96, 96, 3), 0.05, np.float32)
    d = colour.negative(rgb, 1.0, "fakecolour", pitch_um=2.0, seed=4)
    base = np.asarray(FAKE["base"])
    fluct = [d[..., c] - base[c] for c in range(3)]
    rho = np.corrcoef(fluct[0].ravel(), fluct[1].ravel())[0, 1]
    assert abs(rho) < 0.35, (
        "layers share grain they must not share; the dye matrix "
        "couples means, the SHEETS stay independent")


def test_lad_solves_the_printer_to_the_aims(shelf):
    """The laboratories' own calibration: with the solved lights,
    the LAD grey prints to the aim densities in every channel -
    through the full coupled assembly, not per-channel algebra."""
    lights = colour.lad_lights("fakecolour", "fakeprint")
    t = 10.0 ** -np.asarray((0.80, 1.20, 1.60))
    d = colour.positive(t[None, None, :], "fakeprint", lights=lights,
                        grain=False)[0, 0]
    assert np.allclose(d, FAKE_PRINT["lad_aim"], atol=2e-3), d


def test_the_chain_inverts_to_a_positive(shelf):
    """Scene to negative to print: a bright neutral patch must come
    out LIGHT on the print (low density), a dim one dark - the
    positive ordered correctly through both inversions and the
    mask."""
    rgb = np.zeros((8, 8, 3), np.float32)
    rgb[:4] = 0.6
    rgb[4:] = 0.03
    neg = colour.negative(rgb, 1.0, "fakecolour", grain=False)
    lights = colour.lad_lights("fakecolour", "fakeprint")
    pos = colour.positive(colour.transmit(neg), "fakeprint",
                          lights=lights, grain=False)
    lum = pos.mean(axis=-1)
    assert float(lum[:4].mean()) < float(lum[4:].mean()), \
        "bright scene must print at LOW density"
