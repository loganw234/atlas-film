"""The camera stocks: a sheet of film at the film plane.

Everything before this module printed on PAPER - the seven PROCESSES
are print media, and the render reached them as an ideal latent
image. A camera stock is the other sheet: it sits behind the lens,
takes the aerial image, and develops into a NEGATIVE - a density
field with the stock's own grain at the negative's own pitch - which
the paper then prints THROUGH.

The emulsion physics is the one the print grain already earned
(atlas_film.emulsion): a crystal field coated from a seed, storable
as film stock in the literal sense, thinned by the response curve
read as per-crystal probability, developed all-or-nothing on fixed
thresholds. What is new here is the stock's own accounting:

  FOG COUNTS CRYSTALS. A real negative's rebate is not clear - some
  crystals develop unexposed - so the developable fraction is
  p = p_fog + (1 - p_fog) * u, a crystal developing if exposed OR
  fogged, independently: thinning-consistent, and the shadows and
  rebate carry fog grain exactly as real film does.

  THE NEGATIVE IS DENSITY, not a print: `negative` returns D per
  cell, `transmit` turns it into the printing light 10^-D, and the
  double inversion (bright scene -> dense negative -> light print)
  belongs to the caller's chain, not to this module.

Constants ship sourced or refuse by name. The grain areas below are
already the dossier's - Saunders' inversion (a = A*sigma^2/(kappa*D),
W2) applied to each film's PUBLISHED rms granularity at the 48 um /
D 1.0 convention (B1/B4, A8) - and carry the honesty floor with
them: Tri-X's effective bundle is 1.24 um across, so a Tri-X
negative refuses pixels finer than that, exactly as the print
emulsions refuse theirs. The CURVE constants (dmax, toe, speed, fog)
are the sensitometry lane's to deliver; until a stock carries them,
`negative` refuses it with the reason, because a curve invented for
a named real film would be a lie wearing a datasheet's name.
"""

import numpy as np

from atlas_film import emulsion
from atlas_film.processes import KAPPA

# per stock:
#   grain_um2  effective developed-grain projective area, from the
#              film's own published diffuse rms granularity through
#              Saunders' inversion at the Kodak convention
#              (sigma = rms/1000 at net D 1.0, 48 um aperture,
#              A = pi*24^2 = 1809.56 um2):  a = A*sigma^2/(kappa*D)
#              - Tri-X 400: rms 17 (F-4017)  -> 1.204 um2 (d 1.24 um)
#              - Double-X 5222: rms 14       -> 0.817 um2 (d 1.02 um)
#              - T-MAX 100: rms 8 (F-4016)   -> 0.267 um2 (d 0.58 um)
#   dmax, toe, speed, fog - the characteristic curve, AWAITING the
#              sensitometry lane (FILM-N); a stock without them
#              refuses. toe will be fit jointly to the published
#              contrast aim, the reconciliation organ's discipline.
FILMS = {
    "trix":    dict(grain_um2=1.204),
    "5222":    dict(grain_um2=0.817),
    "tmax100": dict(grain_um2=0.267),
}

_CURVE_KEYS = ("dmax", "toe", "speed", "fog")


def _stock(name):
    if name not in FILMS:
        raise ValueError(
            f"no such camera stock {name!r}: the shelf holds "
            + ", ".join(sorted(FILMS)))
    st = FILMS[name]
    missing = [k for k in _CURVE_KEYS if k not in st]
    if missing:
        raise ValueError(
            f"camera stock {name!r} has no sourced "
            + "/".join(missing) +
            " yet - its curve is the sensitometry lane's to deliver, "
            "and a curve invented for a named real film would be a "
            "lie wearing a datasheet's name")
    return st


def negative(lum, E, name, *, pitch_um=None, grain=True, seed=0,
             sheet=None):
    """Expose a camera stock to the aerial image and develop it.

    `lum` is the scene luminance at the film plane, `E` the camera
    exposure; the return is the NEGATIVE's density field, float32,
    fog and all. With `grain` (the default - a camera stock without
    its grain is a contradiction) the deposit is the crystal count
    of the stock's own sheet at the negative's `pitch_um`; a stored
    sheet from `emulsion.coat` prints identically via `sheet=`.
    """
    st = _stock(name)
    dose = np.maximum(np.asarray(lum, np.float32) * E * st["speed"],
                      0.0)
    u = (1.0 - np.exp(-dose.astype(np.float64))) ** st["toe"]
    p_fog = st["fog"] / st["dmax"]
    p = p_fog + (1.0 - p_fog) * u
    if not grain:
        return (st["dmax"] * p).astype(np.float32)
    if not pitch_um:
        raise ValueError(
            "a negative's grain is a count of crystals on the film: "
            "pass pitch_um")
    area = float(pitch_um) ** 2
    if sheet is not None:
        K, thr = sheet
        if K.shape[0] != p.size:
            raise ValueError(
                f"this sheet holds {K.shape[0]} cells and the "
                f"exposure asks for {p.size}: a stock is cut for one "
                "geometry")
        n = emulsion.develop_on(K, thr, p.reshape(-1)).reshape(p.shape)
    else:
        n = emulsion.expose(p, float(st["dmax"]), st["grain_um2"],
                            float(pitch_um), seed, label=name)
    return ((KAPPA * st["grain_um2"] / area) * n).astype(np.float32)


def transmit(D):
    """The printing light: what the negative passes to the paper."""
    return np.power(10.0, -np.asarray(D, np.float64)).astype(np.float32)
