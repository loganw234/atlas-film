"""The granularity reader: RMS density fluctuation, aperture by aperture.

The grain organ's referee. Ware's *Argyronomicon* carries the whole
statistical frame (Appendix 21, pp. 205-207, the treatment he credits
to Dr. A. E. Saunders of Kodak Ltd.): image density is the Nutting
coverage of counted grains, D = kappa*N*a/A; the count in any aperture
is "randomly distributed about the mean" with sigma_N = sqrt(N); and
so the measured density fluctuates with

    sigma_D = sqrt(kappa * a * D / A)

invertible to a mean grain area a = A * sigma_D^2 / D. Selwyn's form
of the same fact is that sigma_D * sqrt(2A) is invariant in the
aperture area A.

This module is the INSTRUMENT half only: it develops nothing and
assumes nothing about how grain got into a print. It reads density
fields off actual prints (zeroed on the unexposed sheet, in the
process's strongest channel - the same densitometer convention as
`sensitometry`), block-averages them into apertures, and reports the
fluctuation. The organ that puts the counting statistics INTO the
prints is judged entirely through these readings, call site included.
"""

import numpy as np

from atlas_film.processes import PROCESSES, process_print


def uniform_print(name, dose, shape=(256, 256), **kw):
    """A uniform patch developed through process_print itself."""
    pr = PROCESSES[name]
    g = np.float32(dose / pr["speed"])
    neg = np.full((shape[0], shape[1], 3), g, np.float32)
    return process_print(neg, 1.0, name, **kw)


def density_field(name, img):
    """Per-pixel reflection density of a print, as a densitometer
    reads it: zeroed on the unexposed sheet, in the channel where the
    saturated print is densest."""
    pr = PROCESSES[name]
    sheet = np.asarray(process_print(
        np.zeros((1, 1, 3), np.float32), 1.0, name), np.float64)[0, 0]
    sat = np.asarray(process_print(
        np.full((1, 1, 3), 50.0 / pr["speed"], np.float32),
        1.0, name), np.float64)[0, 0]
    ch = int(np.argmax(-np.log10(np.maximum(sat, 1e-12)
                                 / np.maximum(sheet, 1e-12))))
    d = np.asarray(img, np.float64)[..., ch]
    return -np.log10(np.maximum(d, 1e-12) / np.maximum(sheet[ch], 1e-12))


def block_average(dfield, block):
    """Non-overlapping block*block aperture means - the scanning
    aperture of a microdensitometer, tiled instead of stepped."""
    if block == 1:
        return dfield
    h = (dfield.shape[0] // block) * block
    w = (dfield.shape[1] // block) * block
    d = dfield[:h, :w]
    return d.reshape(h // block, block, w // block, block).mean(axis=(1, 3))


def rms_granularity(dfield, block=1):
    """sigma_D at an aperture of block*block pixels."""
    tiles = block_average(dfield, block)
    return float(np.std(tiles, ddof=1))


def selwyn_coefficient(dfield, pitch_um, block=1):
    """Selwyn's invariant: sigma_D * sqrt(2A), A the aperture area.

    For counting-statistics grain this does not depend on the block
    size - which is the classical test that a noise field IS grain
    rather than something with a built-in scale.
    """
    area = (block * pitch_um) ** 2
    return rms_granularity(dfield, block) * float(np.sqrt(2.0 * area))
