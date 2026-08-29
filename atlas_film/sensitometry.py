"""The sensitometer: Ware's own quantities, read off the model's own prints.

The reconciliation organ's referee. The optics-film-process dossier
measured the process table against Ware's monographs by evaluating the
CURVE FORMULA under Ware's exposure-range convention; this module
measures the same quantities off `process_print`'s actual output, so
the call site is inside the measurement - a reader that re-derived
`dmax*(1-e^-h)^toe` on its own would agree with a wrong call site
forever, which is the lesson the optics side's exit-state fold taught.

The convention is Ware's, verbatim (*Platinomicon* p. 158, identical
in his workshop notes across an eight-year gap): "The Exposure Range
(DlogH) is from fog+0.04 to 0.9Dmax". This instrument works the way
his did - an X-Rite in reflectance mode over a step test (p. 234; the
*Cyanomicon*'s equivalent at pp. 274-275): density is read against
the unexposed sheet, in the strongest channel. The *Cyanomicon*
measures cyanotype in the red channel because that is "the waveband
where the absorption by Prussian blue has its maximum value" (p. 275)
- picking the maximum-absorption channel IS the published method, not
a convenience of ours.

One consequence worth stating: for a pigment-path process (gum, or
any explicit ink) the deposited colour absorbs less than fully in
every channel, so the density a densitometer reads is scaled below
the table's `dmax` - the instrument reports what a print measures,
not what a constant intends. Every non-pigment process has a
full-absorption channel and reads its constants exactly.
"""

import numpy as np

from atlas_film.processes import PROCESSES, process_print

# fog+0.04 to 0.9Dmax, Platinomicon p. 158. The model's fog is the
# unexposed sheet itself - the reader zeroes on it exactly as a
# reflection densitometer zeroes on the paper base - so the 0.04
# rides on measured density directly.
FOG_PLUS = 0.04
SHOULDER_FRAC = 0.9

# The sweep spans well below the 0.04 crossing of the slowest toe and
# far enough up the shoulder that 1-exp(-h) is 1.0 to float32.
_SWEEP = np.geomspace(1e-4, 50.0, 4001)
_SATURATE = 50.0


def _prints(name, doses, **kw):
    """The prints for a dose ramp, the sheet, and the saturated patch."""
    pr = PROCESSES[name]
    g = (np.asarray(doses, np.float64) / pr["speed"]).astype(np.float32)
    ramp = np.stack([g, g, g], -1)[None, :, :]
    out = np.asarray(process_print(ramp, 1.0, name, **kw), np.float64)[0]
    sheet = np.asarray(process_print(
        np.zeros((1, 1, 3), np.float32), 1.0, name, **kw),
        np.float64)[0, 0]
    sat = np.asarray(process_print(
        np.full((1, 1, 3), _SATURATE / pr["speed"], np.float32),
        1.0, name, **kw), np.float64)[0, 0]
    return out, sheet, sat


def _density(refl, sheet):
    return -np.log10(np.maximum(refl, 1e-12) / np.maximum(sheet, 1e-12))


def density_curve(name, doses, **kw):
    """Reflection density against dose, measured off the print.

    A grey-ramp negative goes through `process_print` itself; density
    is read in the process's maximum-absorption channel, zeroed on the
    unexposed sheet.
    """
    out, sheet, sat = _prints(name, doses, **kw)
    ch = int(np.argmax(_density(sat, sheet)))
    return _density(out[:, ch], sheet[ch])


def dmax_reached(name, **kw):
    """The measured Dmax: the saturated print against the sheet."""
    _, sheet, sat = _prints(name, [0.0], **kw)
    return float(np.max(_density(sat, sheet)))


def exposure_scale(name, **kw):
    """Ware's exposure range over the measured curve.

    log10 of the dose ratio between the fog+0.04 crossing and the
    0.9*Dmax crossing, Dmax itself measured rather than assumed.
    """
    d = density_curve(name, _SWEEP, **kw)
    logh = np.log10(_SWEEP)
    lo = FOG_PLUS
    hi = SHOULDER_FRAC * dmax_reached(name, **kw)
    return float(np.interp(hi, d, logh) - np.interp(lo, d, logh))


def contrast_index(name, **kw):
    """Max dD per decade of dose over the measured curve - the same
    construction the carried contrast-direction tests read, available
    beside the scale so a retune can watch both at once."""
    d = density_curve(name, _SWEEP, **kw)
    return float(np.max(np.gradient(d, np.log10(_SWEEP))))
