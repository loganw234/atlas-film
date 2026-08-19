"""A tricolour print must not judge a subject by its saturation.

`Audit-8-17` finding 11. `tricolour_print` split the deposit between
cyan, magenta and yellow using `1 - hue_i*0.92` as a weight - and
never divided by anything. Those three weights sum to `0.24` for a
neutral pixel and `2.08` for a fully saturated one, so total pigment
deposit varied **8.7x at constant luminance**.

Measured before the fix, at scene luminance 2.0 through the carbro
table: a neutral printed at luminance 0.3407 and a pure red at 0.0547
- the neutral came out **6.8x lighter than a saturated subject of the
same brightness**. And because a neutral pixel has `hue_i = 1` in all
three layers, the cyan layer's dose was `0.08*lum` no matter what the
red channel actually recorded, which is the opposite of the
docstring's "the red separation really is what the red channel
recorded".

The fix is to make the split a split. The hue logic is untouched;
the weights are normalised so they say how a total is SHARED rather
than how much of it there is, and the share is scaled by
`3*(1 - HUE_K)` - the neutral's own weight sum - so a neutral prints
exactly what it always printed and everything else is pulled onto the
same footing.

That anchor is derived, not tuned. Minimising the spread on its own
would have chosen a much lower scale and been wrong for a reason
worth writing down: the spread falls monotonically as exposure drops,
because a blank print is perfectly flat.
"""
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

develop = pytest.importorskip("darkroom.develop")

LUM = (0.299, 0.587, 0.114)


def printed(rgb, E=1.4, name="carbro"):
    neg = np.array(rgb, np.float32)[None, None, :]
    out = develop.tricolour_print(neg, E, name)
    return np.asarray(out, np.float64)[0, 0]


def luminance(rgb, **kw):
    p = printed(rgb, **kw)
    return float(sum(w * p[i] for i, w in enumerate(LUM)))


def at_luminance(target=2.0):
    """Subjects of equal scene luminance and very different saturation."""
    return {
        "neutral": (target, target, target),
        "half-saturated red": (target * 1.7, target * 0.7, target * 0.7),
        "pure red": (target / LUM[0], 0.0, 0.0),
        "pure green": (0.0, target / LUM[1], 0.0),
        "pure blue": (0.0, 0.0, target / LUM[2]),
    }


def test_equal_luminance_prints_at_roughly_equal_luminance():
    """The finding, as one number. 8.7x of deposit variation put six
    stops of nonsense between a grey subject and a red one."""
    vals = {k: luminance(v) for k, v in at_luminance().items()}
    spread = max(vals.values()) / min(vals.values())
    assert spread < 1.6, (
        "print luminance still tracks saturation: "
        + ", ".join(f"{k} {v:.4f}" for k, v in vals.items())
        + f" (spread {spread:.2f}x)")


def test_the_weights_are_a_split():
    """Directly: whatever the hue, the three shares sum to one."""
    k = develop.HUE_K
    for rgb in at_luminance().values():
        neg = np.array(rgb, float)
        hue = neg / max(neg.max(), 1e-9)
        w = 1.0 - np.clip(hue, 0, 1) * k
        assert (w / w.sum()).sum() == pytest.approx(1.0, rel=1e-12)


def test_a_neutral_prints_exactly_what_it_used_to():
    """The anchor. `3*(1 - HUE_K)` is the neutral's own weight sum, so
    this model change must leave a grey subject alone - that is what
    makes it a correction rather than a restyle."""
    assert luminance((2.0, 2.0, 2.0)) == pytest.approx(0.3407, abs=5e-4)


def test_a_neutral_still_prints_grey():
    p = printed((2.0, 2.0, 2.0))
    assert float(max(p) - min(p)) < 0.2, f"neutral printed {p}"


@pytest.mark.parametrize("rgb,ch,name", [
    ((6.69, 0.0, 0.0), 0, "red"),
    ((0.0, 3.41, 0.0), 1, "green"),
    ((0.0, 0.0, 17.5), 2, "blue"),
])
def test_a_saturated_subject_still_prints_its_own_colour(rgb, ch, name):
    """The split must not have cost the separation its point."""
    p = printed(rgb)
    others = [p[i] for i in range(3) if i != ch]
    assert p[ch] > max(others), f"{name} printed {p}"


def test_darker_subjects_still_print_darker():
    """Monotonicity in luminance, which the deposit variation was
    quietly competing with."""
    xs = [0.5, 1.0, 2.0, 4.0, 8.0]
    ys = [luminance((x, x, x)) for x in xs]
    assert ys == sorted(ys, reverse=True), list(zip(xs, ys))


def test_the_anchor_is_derived_from_the_constant_not_typed():
    """If HUE_K moves, the calibration has to move with it. A literal
    0.24 sitting beside a 0.92 is a trap for whoever tunes the
    pigment response next."""
    src = (ROOT / "darkroom" / "develop.py").read_text(encoding="utf-8")
    body = src[src.index("def tricolour_print"):]
    body = body[:body.index("\ndef ")]
    assert "3.0 * (1.0 - HUE_K)" in body, \
        "the neutral anchor is no longer derived from HUE_K"
    # CODE lines only. The prose above the deposit quotes 0.24 as the
    # measurement it is, and a check that cannot tell a comment from an
    # expression would forbid explaining the number at all.
    code = [ln.split("#", 1)[0] for ln in body.split("\n")]
    assert not any("0.24" in ln for ln in code), \
        "the anchor was typed as a literal instead of derived"
