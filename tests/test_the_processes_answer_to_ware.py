"""The reconciliation organ's referee: the table answers to Ware.

The darkroom's optics-film-process dossier held the process constants
against Mike Ware's monographs and found two contradictions and one
inversion (its round-2 verdict table is the reference for every number
here):

- **platinum Dmax**: the table shipped 1.30; Ware measured ~1.45 three
  ways - "The value of the maximum density, Dmax, for most of the
  tests was ~1.45" (*Platinomicon* p. 235, X-Rite 312 reflectance),
  "approximately 1.40-1.45 using a typical Western paper" (p. 188),
  and Stan Klimek's "Dmax - 1.42" (p. 186). Contradicted-low.
- **platinum exposure scale**: Ware's measured platinum band is
  1.5-1.9 (Table 7.1 p. 158; his own densitometry "~1.9", p. 235)
  against palladium's 2.0-2.4. The shipped curve read 2.107 - outside
  the platinum band, inside palladium's.
- **salt vs albumen**: Reilly, reporting Hubl, has plain salted paper
  needing "the greatest density range in a negative" with albumen the
  shortest of the family; Ware independently sorts salted paper into
  his long-range class beside platinum-palladium (*Cyanomicon*
  pp. 213, 219). The shipped table read salt 1.568 under albumen's
  1.741 - inverted against both primaries.

WATCHED FAILING 2026-08-28, against the shipped constants, with the
sensitometer of `atlas_film/sensitometry.py` (which first reproduced
the dossier's independently derived column to +/-0.004 on every
full-absorption process): platinum Dmax read 1.300, platinum scale
2.107, salt 1.568 < albumen 1.741. The fit that turns these green is
JOINT - the dossier proved raising platinum's dmax alone worsens the
overshoot (2.107 -> 2.164) - and takes Ware's own densitometry pair
(Dmax ~1.45, scale ~1.9, one instrument on one sensitizer) as the
platinum target, with salt fitted to its classmate's scale because
class membership is the only numeric content the salt sources offer
(Ware's 2-2.4 is a negative range, and this curve maps scene to
print - the dossier's governing caveat).

What the fit does NOT chase, on the record: Ware's third measured
number, gamma ~0.96 (p. 235). One exponent governs this model's toe
and straight-line slope at once, so (dmax, toe) can hit two of Ware's
three measurements exactly and must let the third land where it
lands. The dossier's structural finding - in Ware's chemistry a
bigger toe means a longer scale, in this parameterisation a smaller
exponent does - is the same limitation seen from the other side.
Abstention elsewhere stays abstention: vandyke, albumen, salt and gum
Dmax remain unsourced and UNTOUCHED (the koraks 1.35 salt measurement
is a gold-toned print and non-archival; it licenses nothing).
"""
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from atlas_film import sensitometry as S               # noqa: E402
from atlas_film.processes import PROCESSES             # noqa: E402


def test_the_reader_reproduces_the_dossiers_independent_column():
    """Measure the instrument first. The dossier derived DlogH from the
    curve formula; this sensitometer measures prints. On the rows the
    fit does not touch, the two must agree - or every verdict below is
    a statement about the reader, not the table."""
    for name, derived in (("cyanotype", 1.553), ("vandyke", 1.75),
                          ("albumen", 1.741), ("silver", 2.078)):
        assert S.exposure_scale(name) == pytest.approx(derived, abs=0.01), name


def test_the_reader_sees_the_pigment_path_honestly():
    """Gum deposits a ground pigment, and a pigment absorbs less than
    fully in every channel - so a densitometer reads LESS than the
    table's dmax constant. The instrument must report the print, not
    the intent; the exact scaling is the absorb calibration's and is
    deliberately not pinned here."""
    assert S.dmax_reached("gum") < PROCESSES["gum"]["dmax"] - 0.1


def test_platinum_prints_wares_measured_dmax():
    """~1.45 at p. 235, 1.40-1.45 at p. 188, Klimek's 1.42 at p. 186.
    The shipped 1.30 read 0.15 below the bottom of that - the same
    size and direction of miss as salt's against its one measurement.
    Watched failing at 1.300."""
    assert S.dmax_reached("platinum") == pytest.approx(1.45, abs=5e-3)


def test_platinum_sits_in_wares_platinum_band_not_palladiums():
    """Table 7.1 (p. 158): platinum 1.5/1.5/1.8 across humidity,
    palladium 2.0/2.2/2.4; Ware's own densitometry stretches the
    platinum band to ~1.9 (p. 235). Watched failing at 2.107 -
    squarely palladium. The pin is the densitometry pair's ~1.9, the
    same measurement the Dmax above comes from."""
    scale = S.exposure_scale("platinum")
    assert 1.5 <= scale <= 1.95, f"palladium band again: {scale:.3f}"
    assert scale == pytest.approx(1.90, abs=0.03)


def test_salt_prints_longer_than_albumen():
    """Reilly ch. 7 (Hubl's hierarchy) and Ware's long-range class,
    two authors who never cite each other on the point. Watched
    failing inverted: salt 1.568 against albumen 1.741."""
    assert S.exposure_scale("salt") > S.exposure_scale("albumen")


def test_salt_shares_the_class_ware_put_it_in():
    """*Cyanomicon* pp. 213 and 219 sort salted paper and
    platinum-palladium into ONE negative-range class - the same
    negatives serve both. Class membership is the entire numeric
    content the sources give salt's scale, so the fit encodes it
    literally: salt prints its classmate's scale."""
    assert S.exposure_scale("salt") == pytest.approx(
        S.exposure_scale("platinum"), abs=0.05)


def test_gum_is_still_the_shortest_scale():
    """The one ordering the dossier CORROBORATED (gum bracketed with
    classic cyanotype and grade 2-3 silver-gelatin, *Cyanomicon*
    pp. 213, 218) must survive the fit untouched."""
    scales = {n: S.exposure_scale(n) for n in PROCESSES}
    assert min(scales, key=scales.get) == "gum", scales


def test_the_corroborated_rows_did_not_move():
    """The controls. Cyanotype is corroborated on both axes (measured
    Dmax family 1.16-1.55 with Ware's own at 1.54, Fig. 9.1 p. 276;
    scale inside the New/Simple bands) and silver's 2.10 is an exact
    Foma datasheet match. A joint fit that moved a corroborated row
    would be a restyle wearing a correction's name."""
    assert S.dmax_reached("cyanotype") == pytest.approx(1.45, abs=5e-3)
    assert S.exposure_scale("cyanotype") == pytest.approx(1.553, abs=0.01)
    assert S.dmax_reached("silver") == pytest.approx(2.10, abs=5e-3)
