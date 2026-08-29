"""A misregistered plate slides off its edge; it must not wrap round.

Moved from the darkroom's develop units on the extraction: `_shift`
is the pigment model's slide, so its test lives beside the model.
Wrapping would print the right edge of one separation onto the left
of the sheet, which no physical misregistration can do.
"""
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from atlas_film import pigments


def test_shift_does_not_wrap():
    img = np.full((8, 12, 3), 0.5, np.float32)
    img[:, 0] = 0.9                        # bright left edge
    out = pigments._shift(img, 3.0, 0.0)
    # the bright edge moved right; nothing wrapped to the far side
    assert out[0, 3, 0] == pytest.approx(0.9)
    assert out[0, -1, 0] == pytest.approx(0.5)
    # beyond the slid plate's edge: fill (transmittance 1 = no pigment)
    assert out[0, 0, 0] == pytest.approx(1.0)
