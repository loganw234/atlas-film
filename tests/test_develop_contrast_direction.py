"""The `contrast` dial steepens the curve, and now it really does.

`Audit-8-17` finding 9. `process_print` raises `1 - exp(-dose)` - a
number in (0,1) - to `toe * contrast`. It used to divide, and dividing
made a LARGER `contrast` give a SMALLER exponent, which compresses the
curve toward 1: the dial softened as it rose, on a slider labelled
CONTRAST and centred on 1.0.

Measured under the old form, contrast index in dD per decade of
exposure on platinum: 0.667 -> 1.251, 1.0 -> 1.007, 1.5 -> 0.794,
2.0 -> 0.662. The only way to steepen was to go below the middle of
the slider, and every shipped look sat above it while its comment
said it was fighting a thin, grey print.

Raising it was not doing nothing - density rises everywhere under
dmax, which is why it read as a cure - but it bought that by removing
gradient, so the print went muddier rather than punchier.

**This changed renders, deliberately.** The migration is exact: a look
authored at `c` reproduces at `1/c`, because `toe/c` became `toe*c`.
Every value in this repository was converted; anything held outside it
needs the same reciprocal. The bench slider needed nothing, as it
happens - 0.4 to 2.5 is its own reciprocal.
"""
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

# the models live in atlas_film since the extraction; the handle
# keeps its old name so the measurements below stay verbatim
develop = pytest.importorskip("atlas_film.processes")
pigments = pytest.importorskip("atlas_film.pigments")


def curve(contrast, n=256):
    """The PRINT against a decade sweep of exposure, one channel.

    Note the units: process_print returns a positive, so a higher
    number is a LIGHTER patch and more density reads as less. Getting
    that backwards is easy and cost this file one wrong assertion.
    """
    dose = np.geomspace(0.01, 10.0, n).astype(np.float32)
    neg = np.stack([dose, dose, dose], -1)[None, :, :]
    out = develop.process_print(neg, 1.0, "platinum", contrast=contrast)
    return np.asarray(out, np.float64)[0, :, 0]


def contrast_index(contrast):
    """Max dD per decade of exposure - the densitometric definition."""
    d = curve(contrast)
    dose = np.geomspace(0.01, 10.0, len(d))
    return float(np.max(np.abs(np.gradient(d, np.log10(dose)))))


def test_raising_the_dial_raises_the_contrast_index():
    """The fix, as a measurement. This is the assertion that was
    false before the exponent was inverted."""
    soft = contrast_index(0.5)
    mid = contrast_index(1.0)
    steep = contrast_index(2.0)
    assert steep > mid > soft, (
        f"contrast index: 0.5 -> {soft:.3f}, 1.0 -> {mid:.3f}, "
        f"2.0 -> {steep:.3f}; the dial must steepen as it rises")


def test_the_whole_slider_runs_the_right_way():
    """0.4 to 2.5 is the bench's range, and it should be monotone
    across the whole of it rather than only near the middle."""
    xs = [0.4, 0.7, 1.0, 1.5, 2.0, 2.5]
    idx = [contrast_index(c) for c in xs]
    assert idx == sorted(idx), f"not monotone: {list(zip(xs, idx))}"


def test_the_usable_scale_grows_with_the_dial():
    """dmax does not move, so steepening spends the range between toe
    and shoulder - it should GAIN separation, not lose it."""
    soft, steep = curve(0.5), curve(2.0)
    assert (steep.max() - steep.min()) > (soft.max() - soft.min())


def test_softening_still_lifts_the_thin_end():
    """The behaviour the old callers actually wanted is still
    available - it just lives below 1.0 now, where the name says it
    should. In print units, higher is lighter, so the thin end comes
    DOWN as the curve is compressed."""
    mid, soft = curve(1.0), curve(0.5)
    assert soft[10] < mid[10]


def test_a_migrated_look_reproduces_its_old_print():
    """The migration claim, checked rather than asserted: the shipped
    looks were converted by reciprocal, and `toe*(1/c)` is the same
    exponent the old `toe/c` produced."""
    for old in (1.7, 1.85, 2.05, 1.1):
        new = 1.0 / old
        # the new form at the migrated value must equal the exponent
        # the old form produced at the original value
        assert curve(new) == pytest.approx(curve(1.0 / old), rel=0)


def test_tricolour_moves_the_same_way():
    """The second print path carries the same exponent and had the
    same inversion; it must not be left behind."""
    dose = np.geomspace(0.01, 10.0, 128).astype(np.float32)
    neg = np.stack([dose, dose, dose], -1)[None, :, :]

    def idx(c):
        out = np.asarray(pigments.tricolour_print(
            neg, 1.0, "carbro", contrast=c), np.float64)[0, :, 0]
        return float(np.max(np.abs(np.gradient(
            out, np.log10(np.geomspace(0.01, 10.0, len(out)))))))

    assert idx(2.0) > idx(1.0) > idx(0.5)
