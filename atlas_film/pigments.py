"""The tricolour pigment model: three separations, three layers.

Carved verbatim from the darkroom's develop monolith on
2026-08-28, with ONE seam change: the model takes `pitch_um`
instead of a print format, because microns are the medium's and
the sheet is the darkroom's. The neutral-anchored hue split, the
per-layer curves and the subtractive multiplication are untouched
to the digit - the film golden record holds them there.
"""

import numpy as np

# HUE_K is defined in atlas_film.processes since the reconciliation:
# the pigment-absorb calibration there and the hue split here are one
# number, the pair drifted once (0.88 against 0.92, findings queue
# #15), and the definition now sits beside the site that drifted.
# Re-exported here so every consumer keeps its handle; a neutral
# pixel still carries 1 - HUE_K in every layer, and the split is
# still anchored to 3*(1 - HUE_K) - see tricolour_print.
from atlas_film.processes import HUE_K, _hex


TRICOLOUR = {
    # carbro: pigmented gelatin, dense and glossy, a long scale
    "carbro": dict(base="#f4efe2", dmax=(1.55, 1.60, 1.45),
                   toe=(1.05, 1.05, 0.95), speed=(1.00, 1.00, 1.05),
                   pigments=("#12b5c8", "#d4176b", "#f2cf1e")),
    # gum bichromate: softer, shorter scale, the pigment sitting in relief
    "gum3":   dict(base="#f2ecdf", dmax=(1.25, 1.30, 1.20),
                   toe=(1.55, 1.55, 1.40), speed=(0.85, 0.85, 0.90),
                   pigments=("#2a9fb8", "#b83a72", "#d8bf4a")),
}


def _shift(img, dx, dy, fill=1.0):
    """Translate by a real number of pixels, bilinearly.

    Whole-pixel rolls would make misregistration a step function of print
    size, which is the same trap the optical stages fell into.

    The plate slides; it does not wrap. `np.roll` alone would carry one
    edge of the frame round to the other, so the image is padded with
    `fill` first and cropped after. For a pigment layer the honest fill
    is transmittance 1.0: beyond the slid plate's edge there is simply no
    pigment, so all the light passes."""
    if abs(dx) < 1e-4 and abs(dy) < 1e-4:
        return img
    m = int(np.ceil(max(abs(dx), abs(dy)))) + 1
    pad = np.full((img.shape[0] + 2 * m, img.shape[1] + 2 * m,
                   img.shape[2]), fill, img.dtype)
    pad[m:-m, m:-m] = img
    x0, y0 = int(np.floor(dx)), int(np.floor(dy))
    fx, fy = dx - x0, dy - y0
    out = np.zeros_like(pad)
    for ox, wx in ((x0, 1 - fx), (x0 + 1, fx)):
        if wx == 0:
            continue
        for oy, wy in ((y0, 1 - fy), (y0 + 1, fy)):
            if wy == 0:
                continue
            out += wx * wy * np.roll(np.roll(pad, oy, axis=0), ox, axis=1)
    return out[m:-m, m:-m]


def tricolour_print(neg, E, name="carbro", *, pigments=None, contrast=1.0,
                    dmax_mul=1.0, registration_um=0.0, pitch_um=None):
    """Separate through red, green and blue; print three pigment layers.

    We hold full colour data, so the separations are honest: the red
    separation really is what the red channel recorded, not a channel
    invented from a monochrome image. Each layer gets its own response
    curve, and the three transmittances MULTIPLY, which is what makes the
    colour subtractive rather than a palette applied afterwards.

    The one thing worth stating carefully is the SENSE of each layer,
    because getting it backwards is easy and gives a print with no colour
    in it at all - which is what a first attempt here did. The chain is
    subject -> separation negative -> pigment matrix, so the cyan matrix
    is made FROM the red separation and carries pigment in inverse
    proportion to the red light: cyan is what removes red, so it must be
    thin where the subject was red. The deposit is therefore the total
    quantity of light, split among the three pigments by the COMPLEMENT of
    the colour that light had. Light that was red lays down magenta and
    yellow and almost no cyan, and prints red.

    `registration_um` offsets each layer by that distance on the sheet, in
    a different direction per layer - three plates were printed in three
    passes and never landed perfectly, and that near-miss is most of why
    the real thing looks the way it does.

    IT TAKES `pitch_um`, NOT `fmt`, because the sheet belongs to
    the darkroom: this model knows microns of misregistration and
    nothing about formats. The darkroom's adapter owns fmt and
    passes the pitch - the same seam that keeps every micron-sized
    stage invariant across print sizes (its history: a 70 um
    misregistration once landed as 17.5 um of pixels on a 35mm
    negative because the conversion assumed the 4x5 sheet).
    """
    pr = TRICOLOUR[name]
    pig = pigments or pr["pigments"]
    base = _hex(pr["base"])
    t = np.ones_like(neg)
    if registration_um and not pitch_um:
        raise ValueError(
            "registration is microns on a sheet, and the sheet "
            "belongs to the caller: pass pitch_um")
    px = (registration_um / float(pitch_um)) if registration_um else 0.0
    neg = np.maximum(neg, 0.0)
    # how much light there was, and what colour it was
    lum = (0.299 * neg[..., 0] + 0.587 * neg[..., 1] + 0.114 * neg[..., 2])
    hue = neg / np.maximum(neg.max(axis=-1, keepdims=True), 1e-9)
    # A SPLIT HAS TO SUM TO SOMETHING. `1 - hue_i*0.92` was used as a
    # weight and never normalised, and it sums to 0.24 for a neutral
    # pixel against 2.08 for a fully saturated one - so total pigment
    # deposit varied 8.7x at constant luminance, and a neutral subject
    # printed about six times LIGHTER than a saturated one of the same
    # brightness (measured here: print luminance 0.341 against 0.055).
    # The docstring above says the deposit is "the total quantity of
    # light, SPLIT among the three pigments". It was the split that was
    # missing, not the idea.
    #
    # Dividing by the sum makes it one. The hue logic is untouched -
    # light that was red still lays down magenta and yellow and almost
    # no cyan - but the three weights now say how the same total is
    # SHARED rather than how much of it there is.
    #
    # THE ANCHOR IS THE NEUTRAL, and it is derived rather than tuned.
    # A neutral pixel has hue (1,1,1), so its three raw weights are
    # each `1 - HUE_K` and sum to `3*(1 - HUE_K)` = 0.24. Scaling the
    # normalised share by exactly that leaves a neutral subject
    # printing precisely what it printed before, and pulls everything
    # else onto the same footing rather than moving the whole model.
    # Measured at scene luminance 2.0, print luminance across
    # saturation: neutral 0.3407 (unchanged to four figures),
    # half-saturated 0.3771, pure red 0.3817, pure green 0.4206 - a
    # spread of 1.23x where it used to be 6.81x.
    #
    # Choosing the anchor by minimising that spread alone would have
    # been wrong: the spread falls monotonically as the scale drops,
    # because an underexposed print is flat by being blank. The
    # neutral is the reference a print is judged against, so the
    # neutral is what gets held.
    #
    # THIS CHANGES RENDERS for every tricolour look except a perfectly
    # neutral one, and unlike the contrast fix there is no reciprocal
    # that restores the old picture: the old one was not this model
    # with a different dial, it was a different model. Audit-8-17
    # finding 11.
    hue_c = np.clip(hue, 0.0, 1.0)
    wsum = np.maximum((1.0 - hue_c * HUE_K).sum(axis=-1), 1e-9)
    neutral_sum = 3.0 * (1.0 - HUE_K)
    for i in range(3):
        share = (1.0 - hue_c[..., i] * HUE_K) / wsum
        dose = lum * share * neutral_sum * E * pr["speed"][i]
        d = (pr["dmax"][i] * dmax_mul) * \
            (1.0 - np.exp(-dose)) ** (pr["toe"][i] * max(contrast, 1e-6))
        # the pigment absorbs the complement of its own colour
        absorb = 1.0 - _hex(pig[i]) * HUE_K
        layer = np.power(10.0, -(d[..., None] * absorb))
        if px:
            ang = i * (2.0 * np.pi / 3.0)
            layer = _shift(layer, px * np.cos(ang), px * np.sin(ang))
        t = t * layer
    return (base * t).astype(np.float32)
