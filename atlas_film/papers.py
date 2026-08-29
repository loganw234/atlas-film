"""The paper law: Beer-Lambert absorption on a reflective sheet.

Carved verbatim from the darkroom's develop monolith on
2026-08-28. The stock tints and the subtractive ink model, exactly
as the darkroom shipped them.
"""

import numpy as np

from atlas_film.processes import _hex


PAPERS = {
    "bright":  "#ffffff",
    "warm":    "#faf6ec",
    "cream":   "#f3ebd8",
    "rag":     "#efe9dd",
    "newsprint": "#e8e2d2",
}


def paper_print(neg, E, paper="#faf6ec", subtractive=True, ink=None):
    """Render the measure as ink on paper instead of light in darkness.

    The additive pipeline is exact for a luminous object photographed
    against black: brightness accumulates where sample points land. Paper
    is the opposite medium and wants the opposite law - Beer-Lambert
    absorption, where density subtracts from the light the sheet reflects:

        reflected = paper * exp(-density)

    That is the physics of dye on a page rather than an effect applied to
    look like one. Where filaments cross they grow darker, the way
    overlapping ink does in an etching, instead of burning toward white.

    With subtractive colour the plate's own hue becomes the ink's: light
    that was emitted red is absorbed as red ink, which takes out green and
    blue and leaves red. Without it the whole measure prints in one
    neutral ink, which is what a single-plate lithograph would give.
    """
    d = (0.299 * neg[..., 0] + 0.587 * neg[..., 1] + 0.114 * neg[..., 2])
    if subtractive:
        peak = np.maximum(neg.max(axis=-1, keepdims=True), 1e-9)
        hue = np.clip(neg / peak, 0.0, 1.0)          # what colour the light was
        absorb = d[..., None] * (1.0 - hue * 0.85)   # ink takes out the rest
    else:
        absorb = d[..., None] * np.ones(3, np.float32)
    if ink is not None:
        absorb = absorb * (1.0 - _hex(ink) * 0.85)
    t = np.exp(-absorb * E)                          # transmittance of the ink
    return t.astype(np.float32)
