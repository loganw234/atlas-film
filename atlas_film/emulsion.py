"""The sheet as an object: coated before light, thinned by exposure.

The grain organ's first formulation drew each pixel's developed count
Poisson about the response curve - correct in law for any single
print (thinning a Poisson crystal field by any per-crystal
probability is again Poisson), but the sheet existed only as a
metaphor: nothing persisted between exposures, and a saturated area
drew itself a fresh ceiling every develop. This module lays the
emulsion down FIRST and lets exposure select from it, which is what
a sheet of film is:

1. COATING. Each pixel-cell holds K crystals, K ~ Poisson(lambda_K)
   with lambda_K = ceiling * A / (kappa * a) - the coating density
   Nutting demands for the process to reach its dmax at full
   development. Drawn from the SHEET stream (the seed), before and
   independent of any image. Same seed, same geometry: same sheet,
   whatever is later printed on it.

2. EXPOSURE. The response curve gives the fraction of crystals a
   dose renders developable: p = (1 - e^-dose)^toe, the curve read
   as the per-crystal probability it has always statistically been.

3. DEVELOPMENT. All-or-nothing and DETERMINISTIC: a crystal's
   sensitivity is a fixed property of the sheet (dossier B9/B12 -
   one latent speck develops the whole crystal), so each cell
   carries a fixed luck value v from the sheet stream, and the
   developed count is the binomial quantile n = F^-1_Bin(K,p)(v).
   No randomness enters at development; more light on the SAME
   sheet can only develop MORE crystals (n monotone in p at fixed
   v, K), a second saturating exposure prints the identical field
   (n = K: the sheet itself), and the marginal law at any single
   exposure is exactly Binomial(K, p) - whose Poisson-mixture is
   the same sigma_D = sqrt(kappa*a*D/A) the referee has pinned all
   along. The statistics did not move; the OBJECT appeared.

THE HONESTY FLOOR. A cell is an independent column of emulsion, and
that is only true of a pixel at least one crystal across: below
pitch = the particle diameter, real grains would span cells and
per-pixel independence would misrepresent them. `expose` REFUSES
pitches under the floor by name. Above it, any pitch is exact in
law - a coarse pixel is an honest aggregate of many crystals - and
the fine end near the floor is where the field resolves into
structure.

NUMERICAL METHODS, declared: the binomial quantile is evaluated
exactly by cumulative summation whenever the realised crystal count
allows (small-K cells - the resolved-grain regime this module
exists for), and by the normal quantile with Acklam's rational
approximation of Phi^-1 (|error| < 1.2e-9) when cells hold hundreds
of crystals or more - where a single grain is far below the step
the approximation could misplace. The crossover is a numerical
choice, not physics, and both paths use the same fixed luck v, so
the same-sheet coupling holds everywhere.
"""

import numpy as np

# the largest realised per-cell crystal count the exact cumulative
# path will walk; above it the normal quantile takes over
EXACT_KMAX = 192


def floor_um(grain_um2):
    """The particle's diameter - the least pitch an honest cell
    model can print."""
    return 2.0 * float(np.sqrt(grain_um2 / np.pi))


def _phi_inv(v):
    """Acklam's rational approximation of the standard normal
    quantile, |relative error| < 1.15e-9 - self-contained because
    this package is pure numpy by contract."""
    a = (-3.969683028665376e+01, 2.209460984245205e+02,
         -2.759285104469687e+02, 1.383577518672690e+02,
         -3.066479806614716e+01, 2.506628277459239e+00)
    b = (-5.447609879822406e+01, 1.615858368580409e+02,
         -1.556989798598866e+02, 6.680131188771972e+01,
         -1.328068155288572e+01)
    c = (-7.784894002430293e-03, -3.223964580411365e-01,
         -2.400758277161838e+00, -2.549732539343734e+00,
         4.374664141464968e+00, 2.938163982698783e+00)
    d = (7.784695709041462e-03, 3.224671290700398e-01,
         2.445134137142996e+00, 3.754408661907416e+00)
    v = np.clip(np.asarray(v, np.float64), 1e-300, 1.0 - 1e-16)
    out = np.empty_like(v)
    lo = v < 0.02425
    hi = v > 1.0 - 0.02425
    mid = ~(lo | hi)
    if np.any(mid):
        q = v[mid] - 0.5
        r = q * q
        out[mid] = (((((a[0] * r + a[1]) * r + a[2]) * r + a[3]) * r
                     + a[4]) * r + a[5]) * q / \
                   (((((b[0] * r + b[1]) * r + b[2]) * r + b[3]) * r
                     + b[4]) * r + 1.0)
    for m, sgn, u in ((lo, 1.0, v[lo] if np.any(lo) else None),
                      (hi, -1.0, (1.0 - v[hi]) if np.any(hi) else None)):
        if u is None or not np.any(m):
            continue
        q = np.sqrt(-2.0 * np.log(u))
        out[m] = sgn * (((((c[0] * q + c[1]) * q + c[2]) * q + c[3])
                         * q + c[4]) * q + c[5]) / \
            ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1.0)
    return out


def _binom_quantile_exact(K, p, v, kmax):
    """n = F^-1_Bin(K,p)(v) by walking the cumulative sum - exact,
    vectorised, one pmf-ratio step per candidate count."""
    Kf = K.astype(np.float64)
    q = np.clip(p.astype(np.float64), 0.0, 1.0 - 1e-12)
    ratio = q / (1.0 - q)
    pmf = (1.0 - q) ** Kf
    cdf = pmf.copy()
    n = (cdf < v).astype(np.int64)
    for k in range(kmax):
        pmf = pmf * np.maximum(Kf - k, 0.0) / (k + 1.0) * ratio
        cdf = cdf + pmf
        n += cdf < v
    return np.minimum(n, K)


def _binom_quantile_normal(K, p, v):
    Kf = K.astype(np.float64)
    q = np.clip(p.astype(np.float64), 0.0, 1.0)
    mu = Kf * q
    sig = np.sqrt(np.maximum(mu * (1.0 - q), 0.0))
    n = np.rint(mu + sig * _phi_inv(v)).astype(np.int64)
    return np.clip(n, 0, K)


def expose(p, ceiling, grain_um2, pitch_um, seed, label=""):
    """Coat a sheet, expose it to the developable-fraction field p,
    and return the developed crystal count per cell.

    `p` is the response curve's output in [0, 1]; `ceiling` is the
    process's full-development density (dmax after any dial), which
    with kappa and the particle area fixes the coating; the sheet is
    a pure function of (seed, shape, coating).
    """
    d_min = floor_um(grain_um2)
    if pitch_um < d_min:
        raise ValueError(
            f"a {pitch_um:g} um pixel cannot hold a {label or 'process'} "
            f"crystal of {d_min:.2f} um: the cell model is honest only "
            f"at pitch >= the particle diameter - render larger, or "
            f"accept the statistical account at a coarser pitch")
    from atlas_film.processes import KAPPA
    area = float(pitch_um) ** 2
    lam_k = float(ceiling) * area / (KAPPA * float(grain_um2))
    rng = np.random.default_rng(seed)
    K = rng.poisson(lam_k, size=p.shape)          # the coating
    v = rng.random(p.shape)                       # the cells' fixed luck
    kmax = int(K.max(initial=0))
    if kmax <= EXACT_KMAX:
        n = _binom_quantile_exact(K, p, v, kmax)
    else:
        n = _binom_quantile_normal(K, p, v)
    return n
