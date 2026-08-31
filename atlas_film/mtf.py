"""The emulsion's sharpness: the traced MTF applied as physics.

Organ 8 (emulsion-sharpness dossier, lane L). Every Kodak sheet on
the shelf publishes a modulation-transfer curve - sine-wave MTFs
by the sheets' own boilerplate, so no square-wave correction - and
the lane traced twenty of them (vector where the art allowed,
calibration residuals down to 0.06% in data terms). Each traced
curve is fit by a three-part transfer the physics names:

    M(f) = c + (1 + a - c) * G(ss, f) - a * G(sb, f)

an UNSCATTERED share c (light that crosses the emulsion straight -
a delta), a DIFFUSED share through the turbidity sigma ss, and an
ADJACENCY LIFT a against the development-scale sigma sb - the
chemical edge effect that pushes every B&W curve above 100%
(TRI-X 112%, T-MAX 400 124%, Double-X 125%), captured faithfully
because a model that cannot exceed unity would deny the most
physically informative feature of the source. M(0) = 1 always.

Applied in FREQUENCY SPACE as the exact transfer function at the
negative's own pitch (reflect-padded; no kernel truncation), to
the linear exposure image before the curve - the measured MTF is
film-plus-process end-to-end, reused in the standard small-signal
sense, declared. At coarse pitches where every scale is sub-pixel
the application is an exact no-op.

WHAT THE RECORD FORCED, carried at the constants: TRI-X's curve
was measured on the discontinued TRI-X Pan 5063 (the 2007 edition
says so; the 2016 edition deleted the disclosure - L2). P3200's
sheet prints the T-MAX 100 artwork vertex-identical to 0.008 pt
(L4), so P3200 ships NO MTF rather than a lie wearing a figure.
The 500T sheet's R and B labels are swapped (L11, corrected here
by the brochure's colours, the three sibling sheets, and layer
physics). Plus-X, the Ilfords, the plates and collodion publish
no MTF: their emulsions stay pixel-sharp, the declared
idealization, named per stock in the shelf table. The overshoot
is developer-dependent (each figure names its process) and is not
transferable across developers - the sheets' own conditions ride
the comments.
"""

import numpy as np

# (c, a, ss_um, sb_um); fit rms/max against the traced points in
# the trailing comment. B&W: single curve per stock.
MTF_BW = {
    # F-4017, D-76 large tank; measured on TRI-X Pan 5063 (L2)
    "trix": (0.00, 0.16, 3.75, 55.8),      # rms 0.021 max 0.038
    # F-4016, D-76 small tank 20C (L3)
    "tmax100": (0.05, 0.16, 1.75, 43.8),   # rms 0.014 max 0.024
    # F-4043, D-76 small tank 20C - the strongest overshoot (L5)
    "tmax400": (0.50, 0.34, 5.00, 29.0),   # rms 0.021 max 0.041
    # H-1-5222, D-96 at control gamma - peak 125% at 4 c/mm (L7)
    "5222": (0.25, 0.26, 6.50, 68.5),      # rms 0.035 max 0.062
}

# colour: per-layer (cyan/magenta/yellow = R/G/B record) triples.
# VISION3 traced from the brochure vector twins, cross-checked
# against the sheet rasters to ~1-2% (L8-L11); 2383 from its
# raster, the set's lowest-resolution trace (L12), at the sheet's
# stated 35% modulation target - the only sheet that states one.
MTF_COLOUR = {
    "50d": ((0.10, 0.08, 5.75, 31.8),      # rms 0.021
            (0.15, 0.06, 5.00, 33.0),      # rms 0.023
            (0.30, 0.00, 7.25, 9.2)),      # rms 0.031
    "250d": ((0.20, 0.04, 6.75, 68.8),     # rms 0.017
             (0.35, 0.08, 5.50, 29.5),     # rms 0.010
             (0.35, 0.12, 4.75, 20.8)),    # rms 0.016
    "200t": ((0.20, 0.14, 6.25, 20.2),     # rms 0.017
             (0.35, 0.24, 5.50, 19.5),     # rms 0.009
             (0.30, 0.20, 4.25, 20.2)),    # rms 0.016
    # 500T: channel identities per the CORRECTED labelling (L11)
    "5219": ((0.25, 0.10, 7.50, 69.5),     # rms 0.023
             (0.25, 0.10, 4.75, 26.8),     # rms 0.010
             (0.40, 0.24, 5.75, 19.8)),    # rms 0.014
    "2383": ((0.65, 0.00, 11.25, 13.2),    # rms 0.041
             (0.80, 0.00, 16.00, 18.0),    # rms 0.032
             (0.35, 0.00, 9.25, 11.2)),    # rms 0.033
}


def transfer(f_cycles_per_mm, params):
    """The fitted MTF at spatial frequency f (cycles/mm)."""
    c, a, ss, sb = params
    f = np.asarray(f_cycles_per_mm, np.float64) / 1000.0
    g = lambda s: np.exp(-2.0 * np.pi**2 * s**2 * f**2)
    return c + (1.0 + a - c) * g(ss) - a * g(sb)


def apply(img, params, pitch_um):
    """Filter a 2-D linear-exposure image by the stock's MTF at
    the given pitch - exact, in frequency space, reflect-padded.
    A pitch too coarse to resolve any of the kernel's scales
    returns the image untouched (a true no-op, bit-identical)."""
    c, a, ss, sb = params
    pitch = float(pitch_um)
    if max(ss, sb) / pitch < 0.35:
        return img
    x = np.asarray(img, np.float64)
    pad = int(min(max(4.0 * sb / pitch, 4.0), max(x.shape)))
    xp = np.pad(x, pad, mode="reflect")
    fy = np.fft.fftfreq(xp.shape[0], d=pitch / 1000.0)
    fx = np.fft.rfftfreq(xp.shape[1], d=pitch / 1000.0)
    fr = np.sqrt(fy[:, None] ** 2 + fx[None, :] ** 2)
    out = np.fft.irfft2(np.fft.rfft2(xp) * transfer(fr, params),
                        s=xp.shape)
    return np.maximum(out[pad:pad + x.shape[0],
                          pad:pad + x.shape[1]], 0.0)
