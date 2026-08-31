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

Constants ship sourced or refuse by name. The grain areas are the
dossier's - Saunders' inversion applied to each film's PUBLISHED rms
granularity - and carry the honesty floor with them: Tri-X's
effective bundle is 1.24 um across, so a Tri-X negative refuses
pixels finer than that, exactly as the print emulsions refuse
theirs. The curves are the sensitometry lane's (film-stocks
dossier, FILM-N): fit to each sheet's own characteristic-curve
table at the named normal development, on an axis the fits proved
ABSOLUTE - every stock's speed point lands within 0.04 log H of
where ISO 6 puts its rated speed, unfit - with Kodak's printed
contrast aims emerging as cross-checks rather than inputs. A stock
added without its curve refuses with the reason, because a curve
invented for a named real film would be a lie wearing a datasheet's
name.
"""

import numpy as np

from atlas_film import emulsion
from atlas_film.processes import KAPPA

# THE CURVE FAMILY IS SENSITOMETRY'S OWN, not the print table's. A
# print process saturates; a camera negative holds a STRAIGHT LINE
# for decades - Hurter & Driffield's 1890 law (film-grain dossier
# A1: "D = a + gamma*log t"), with a toe below and a shoulder above
# (H-740). The print family (1-e^-h)^toe cannot hold constant gamma
# over three decades, and the datasheets' own read-off tables prove
# the negatives do - so the stocks carry, in x = log10(H lux-s):
#
#     a(x) = gamma * wt * softplus((x - ht) / wt)
#     D(x) = fog + a - ws * softplus((a - (dmax-fog)) / ws)
#
# a smooth toe at ht widening over wt into an exact straight line of
# slope gamma, capped smoothly into dmax over ws. The crystals read
# it exactly as they read the print curves: p = D/dmax is the
# developable fraction, fog crystals included.
#
# per stock (film-stocks dossier, FILM-N; all fits over the sheet's
# own read-off characteristic-curve table at the NORMAL development
# the sheet names, fog and dmax read off the same figures, gamma
# printed or read off the straight line; rms of every fit <= 0.019 D
# with max residual 0.028 D):
#   ht, wt   fit; ht sits on the ABSOLUTE lux-second axis - the
#            model's speed point D = fog+0.10 lands within 0.04
#            log H of ISO 6's S = 0.8/Hm prediction from each
#            film's rated speed, unfit, for all four stocks
#   gamma    trix 0.59 (straight line read off N4; the chord then
#            reproduces Kodak's PRINTED aim CI 0.56 to 0.004);
#            5222 0.66 (PRINTED on the fitted curve itself, N9);
#            T-MAXes fit (their sheets print no numeric aim, N3)
#   fog      the curve floor, read off (N10: no sheet prints it as
#            a number); 5222's is Status M blue densitometry (N2) -
#            the one cross-convention constant, declared
#   dmax     the highest density the film's own datasheet plots the
#            emulsion reaching (longest-development curve top) -
#            NOT emulsion capability, which H-740 says the step
#            tablet never reaches (N12)
#   grain_um2  Saunders' inversion of the published rms granularity
#            (a = A*sigma^2/(kappa*D) at 48 um / D 1.0): Tri-X 17 ->
#            1.204 um2, Double-X 14 -> 0.817, T-MAX 100's 8 ->
#            0.267, T-MAX 400's 10 -> 0.417
# The sheets' reciprocity tables (FILM-N13/14: Tri-X 100s -> 1200s
# against T-MAX 400's 100s -> 300s, a 4x spread) ride the dossier as
# organ 3's calibration - reciprocity is per-stock, not shared.
# SPECTRAL RESPONSE (organ 4) is a DECLARED band projection, not a
# fit: the render's three channels carry no spectra to integrate
# against, so each stock's `sens` is the maximum-entropy projection
# consistent with its sheet's class, normalised to sum one so a
# neutral scene meters unchanged. Panchromatic (all four Kodak
# stocks, FILM-N16: ~250-650 nm with a cliff at 650-660; TRI-X
# peaks 380-400 with a 560-620 shoulder, 5222 peaks 420-440)
# projects flat; the 1890 plates (early-plates D9/D13: blue/violet/
# UV only - "as black as Indian ink" to the rest of the spectrum)
# project blue-only, their UV lobe beyond the render's reach,
# declared. An ortho class exists the moment a sourced ortho stock
# does.
_PAN = (1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0)
_BLUE = (0.0, 0.0, 1.0)

FILMS = {
    # F-4017: T-MAX Developer small tank 20C, 6 min - the condition
    # Kodak's own CI 0.56 aim names (FILM-N1/N4)
    "trix":    dict(fog=0.32, dmax=3.02, gamma=0.59, ht=-2.73,
                    wt=0.24, ws=0.35, grain_um2=1.204, sens=_PAN),
    # H-1-5222: D-96 21C, 6.5 min, gamma 0.66 printed on the curve;
    # EI 250 daylight; grey acetate base, Status M blue (FILM-N2/N9)
    "5222":    dict(fog=0.25, dmax=2.45, gamma=0.66, ht=-2.60,
                    wt=0.10, ws=0.35, grain_um2=0.817, sens=_PAN),
    # F-4016: T-MAX RS large tank 20C, 8 min - the ISO-anchored
    # curve (its speed point sits at -2.12 for ISO 100's -2.10);
    # the sheet's D-76 figure is defective in print (FILM-N7)
    "tmax100": dict(fog=0.21, dmax=3.13, gamma=0.525, ht=-2.25,
                    wt=0.16, ws=0.25, grain_um2=0.267, sens=_PAN),
    # F-4043: D-76 small tank 20C, 8 min (FILM-N8)
    "tmax400": dict(fog=0.24, dmax=3.07, gamma=0.74, ht=-2.39,
                    wt=0.36, ws=0.50, grain_um2=0.417, sens=_PAN),
    # THE 1890 PLATES (early-plates dossier, lane D). Both fit over
    # H&D's own printed 14-point tables, ferrous oxalate, on an
    # axis one candle-metre-second is DECLARED ~ one lux-second
    # (D3; the standard candle is blue-poor and these plates
    # blue-only, so period daylight speed ran faster than the
    # candle suggests - recorded, not corrected). Densities are net
    # of fog because H&D subtracted the fog strip; fog=0 honours
    # the source's own convention and its silence. The family's
    # `gamma` is its asymptotic slope - the H&D REFEREE in the
    # tests reads the chord and the intercept off the model's curve
    # by H&D's own printed formulas (D2), and hd22 must give back
    # the gamma H&D themselves printed. grain_um2 is BRACKETED,
    # wearing its flag (D19 is a recorded silence): 0.196 um2 is
    # the developed-silver particle class the silver print process
    # already carries, inside Vitale's modern gelatin range and
    # consistent with collodion's microfilm-low granularity - a
    # bracket, not a measurement.
    #
    # D1: the Manchester Slow, Experiment 21 - inertia 6.12 CMS,
    # H&D speed 5.6, chord gamma 0.89-0.91, Dmax 2.352 observed
    # still creeping at 5120 CMS. Fit rms 0.023, max 0.048 D.
    "manchester": dict(fog=0.0, dmax=2.352, gamma=1.020, ht=0.97,
                       wt=0.40, ws=0.20, grain_um2=0.196,
                       sens=_BLUE),
    # D5: Experiment 22, the faster unnamed plate H&D fit with
    # gamma 1.176 and log i 0.579 PRINTED - the validator. Dmax
    # 3.405 observed still creeping; the fit prefers a 3.5 ceiling.
    # Fit rms 0.051, max 0.084 D - 1890 densitometry with H&D's own
    # stated 2.4-5% errors, and their straight line deviates from
    # their own table more than this curve does.
    "hd22":    dict(fog=0.0, dmax=3.5, gamma=1.870, ht=1.18,
                    wt=0.55, ws=0.80, grain_um2=0.196, sens=_BLUE),
}

_CURVE_KEYS = ("fog", "dmax", "gamma", "ht", "wt", "ws")

# RECIPROCITY (organ 3): each sheet's own compensation table as
# (seconds, +stops) rows, interpolated in log t between rows and
# REFUSING beyond them - 5222's sheet is silent past one second,
# and silence is not zero (FILM-N13/N14). TRI-X pairs its +stops
# with development cuts precisely to hold contrast, so applying the
# exposure term against the normal-development curve reproduces the
# properly compensated negative; the uncompensated contrast rise
# the dossier warns about is quantified by no sheet and stays out.
# The plates hold reciprocity flat on H&D's own check (D3: quarter
# candle-metre for 40 s matched one for 10 s; their axis is I*t)
# across their tables' exposure span read as seconds at one
# candle-metre.
RECIPROCITY = {
    "trix":    ((1e-5, 1.0), (1e-4, 0.5), (1e-3, 0.0), (1e-2, 0.0),
                (1e-1, 0.0), (1.0, 1.0), (10.0, 2.0), (100.0, 3.0)),
    "5222":    ((1e-4, 0.0), (1.0, 0.0)),
    "tmax100": ((1e-4, 1 / 3), (1.0, 1 / 3), (10.0, 0.5),
                (100.0, 1.0)),
    "tmax400": ((1e-4, 0.0), (1.0, 0.0), (10.0, 1 / 3),
                (100.0, 1.5)),
    "manchester": ((0.625, 0.0), (5120.0, 0.0)),
    "hd22":    ((1.0, 0.0), (8192.0, 0.0)),
}


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
    if "sens" not in st:
        raise ValueError(
            f"camera stock {name!r} has no declared spectral "
            "projection (organ 4): name its class - a stock that "
            "sees everything equally by accident is a claim, not a "
            "default")
    return st


def characteristic(logH, name):
    """The stock's D-logH curve - H&D's straight line with a smooth
    toe and shoulder, on the absolute lux-second axis."""
    st = _stock(name)
    x = np.asarray(logH, np.float64)
    a = st["gamma"] * st["wt"] * np.logaddexp(0.0, (x - st["ht"])
                                              / st["wt"])
    span = st["dmax"] - st["fog"]
    return st["fog"] + a - st["ws"] * np.logaddexp(
        0.0, (a - span) / st["ws"])


def normal_highlight(name):
    """The exposure H (lux-seconds) a normally exposed negative's
    highlights receive: H-740's latitude construction (FILM-N12)
    puts them ~1.0 D above the speed point, itself 0.10 above fog -
    so this solves the stock's own curve at D = fog + 1.10. The
    darkroom meters the camera against it."""
    st = _stock(name)
    xx = np.linspace(-5.0, 4.5, 6001)
    return float(10.0 ** np.interp(st["fog"] + 1.10,
                                   characteristic(xx, name), xx))


def _project(img, st):
    """The stock's eye: a (..., 3) aerial image collapses through
    the film's own declared band weights; anything else is already
    projected light and passes untouched - the law tests' scalar
    fields keep their meaning."""
    a = np.asarray(img, np.float64)
    if a.ndim and a.shape[-1] == 3:
        return a @ np.asarray(st["sens"], np.float64)
    return a


def reciprocity(name, t):
    """The stock's clock: +stops of compensation its own sheet
    demands at exposure time t seconds (FILM-N13/N14; the plates
    flat by H&D's own check, D3). Interpolated in log t between the
    sheet's rows; beyond them the stock refuses, because a sheet's
    silence is not a zero."""
    _stock(name)
    if name not in RECIPROCITY:
        raise ValueError(
            f"no reciprocity table for {name!r}: its lane has not "
            "delivered one, and a shared curve would contradict the "
            "4x per-stock spread the dossier measured")
    rows = RECIPROCITY[name]
    t = float(t)
    if not rows[0][0] <= t <= rows[-1][0]:
        raise ValueError(
            f"{name}'s sheet tabulates reciprocity only for "
            f"{rows[0][0]:g}..{rows[-1][0]:g} s and is silent at "
            f"{t:g} s - silence is not zero")
    lt = np.log10([r[0] for r in rows])
    return float(np.interp(np.log10(t), lt, [r[1] for r in rows]))


def negative(img, E, name, *, pitch_um=None, grain=True, seed=0,
             sheet=None, t=None):
    """Expose a camera stock to the aerial image and develop it.

    `img` is the aerial image - a (..., 3) RGB field collapses
    through the STOCK's declared spectral projection (a plate is
    blue-blind here, organ 4), a scalar field is already-projected
    light. Times E it is the exposure at the film plane in the
    curve's own lux-seconds (the datasheet axes are absolute -
    their speed points sit where ISO 6 says the rated speeds put
    them - so a metered E carries real units here); the return is
    the NEGATIVE's density field, float32, fog and all. `t` is the
    exposure duration in seconds: given, the stock's own
    reciprocity table discounts the exposure it actually counts
    (organ 3; None stays reciprocity-free, declared). With `grain`
    (the default - a camera stock without its grain is a
    contradiction) the deposit is the crystal count of the stock's
    own sheet at the negative's `pitch_um`, the curve read as the
    developable fraction p = D/dmax, fog crystals included; a
    stored sheet from `emulsion.coat` prints identically via
    `sheet=`.
    """
    st = _stock(name)
    dose = np.maximum(_project(img, st) * E, 0.0)
    if t is not None:
        dose = dose * 2.0 ** -reciprocity(name, t)
    D = characteristic(np.log10(np.maximum(dose, 1e-30)), name)
    if not grain:
        return D.astype(np.float32)
    p = D / st["dmax"]
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


def normal_exposure(img, name, percentile=99.5, t=None):
    """The camera's meter for a film stage: the E that places the
    scene's bright decile - as THIS stock sees it, through its own
    spectral projection - at the stock's normal highlight. With t,
    the meter raises E by the sheet's own reciprocity compensation
    (that is literally what the tables instruct the photographer to
    do), so the developed negative lands where it should."""
    st = _stock(name)
    ref = float(np.percentile(_project(img, st), percentile))
    E = normal_highlight(name) / max(ref, 1e-12)
    if t is not None:
        E *= 2.0 ** reciprocity(name, t)
    return E


def transmit(D):
    """The printing light: what the negative passes to the paper."""
    return np.power(10.0, -np.asarray(D, np.float64)).astype(np.float32)
