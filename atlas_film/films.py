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
from atlas_film.mtf import MTF_BW as _MTF_BW
from atlas_film.mtf import apply as _mtf_apply
from atlas_film.halation import apply as _halation_apply
from atlas_film.halation import for_stock as _halation_for
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
    # F-4016: T-MAX RS large tank 20C - the ISO-anchored curve
    # (its speed point sits at -2.12 for ISO 100's -2.10); the
    # sheet's D-76 figure is defective in print (FILM-N7). The
    # normal time is 8 3/4 min, not the 8 an earlier comment
    # carried - lane I verified the table against a render and the
    # RS curve starts at 8.70 min (FILM-I6)
    "tmax100": dict(fog=0.21, dmax=3.13, gamma=0.525, ht=-2.25,
                    wt=0.16, ws=0.25, grain_um2=0.267, sens=_PAN),
    # F-4043: D-76 small tank 20C, 7 1/2 min (not 8 - FILM-I7's
    # render-verified correction; FILM-N8)
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
    # WET COLLODION (developer-hand dossier, lane I: the 1998 JIST
    # recreation, Skladnikiewitz/Hertel/Schmidt). The curve is the
    # traced 1.1%-iodide iron-developed characteristic (19 points
    # off the one separable curve, axis residuals 0.001 logE /
    # 0.009 D), carried as a SAMPLED TABLE because the bent
    # shoulder is the source's own geometry - "densities below 1
    # are formed only by surface silver" - and no smooth family
    # earns it. fog=0 on the paper's net-density axis. The
    # ABSOLUTE placement is bracketed and flagged twice over: the
    # authors' working index DIN -9 (~ISO 0.1 by the modern
    # conversion, the lane's own arithmetic, flagged I19) agrees
    # with Towler's 1864 field exposures within a stop; the
    # traced relative axis is shifted so net D 0.1 sits at the
    # ISO-style speed point H = 0.8/0.1 = 8 lux-seconds
    # (logH 0.90 at the traced logE_rel 0.80: offset +0.10).
    # grain_um2 is GRANULARITY-EQUIVALENT, not crystal-literal:
    # the measured granularity is microfilm-low (early-plates D18)
    # while the iodide image particle is 4-6 um (I16) - Dutton's
    # opaque-disc assumption fails for surface-stacked silver, so
    # the noise constant follows the measurement and the conflict
    # is recorded here. Blue-eyed: spectral max ~420 nm (I16,
    # corroborating D13). Contrast is NOT a development-time dial
    # for this stock (45-90 s "equally satisfactory", I18) - ci=
    # refuses by mechanism; intensification (the real knob, I17)
    # is the named future dial.
    "collodion": dict(
        fog=0.0, dmax=1.57, grain_um2=0.196, sens=_BLUE,
        curve=(
            (0.80, 0.90, 1.00, 1.10, 1.20, 1.30, 1.40, 1.50, 1.60,
             1.70, 1.80, 1.90, 2.00, 2.10, 2.30, 2.50, 2.70, 2.90,
             3.10),
            (0.07, 0.10, 0.17, 0.30, 0.44, 0.57, 0.70, 0.81, 0.91,
             0.96, 1.07, 1.13, 1.18, 1.22, 1.30, 1.38, 1.43, 1.49,
             1.57))),
    # FOUR MORE SHEETS (four-more-sheets dossier, lane K), all
    # carried as traced tables - the geometry is the source, and
    # tables clamp honestly where a sheet stops plotting.
    #
    # PLUS-X 125 (F-4018): D-76 small tank 20C at the tabulated
    # 5 1/2 min normal, time-interpolated between the plotted 5 and
    # 7 min curves (flagged derived, K4). The lane's calibration
    # anchor: its ISO 6 speed point EMERGES within 0.017 log H of
    # ISO 125 - the axis is absolute and unshifted. Grain: rms 10 -
    # measured in HC-110 (B) at 70F large tank (K5), a different
    # developer than the curve; the mismatch is the sheet's own and
    # cannot be closed inside it. Flat pan (F15).
    "plusx": dict(
        fog=0.0, dmax=1.718, grain_um2=0.417, sens=_PAN,
        curve=(
            (-3.10, -3.00, -2.80, -2.60, -2.40, -2.20, -2.00,
             -1.80, -1.60, -1.40, -1.20, -1.00, -0.80, -0.60,
             -0.40, -0.20, 0.00, 0.20),
            (0.255, 0.258, 0.264, 0.277, 0.306, 0.353, 0.420,
             0.502, 0.602, 0.708, 0.820, 0.940, 1.070, 1.206,
             1.347, 1.474, 1.598, 1.718))),
    # T-MAX P3200 (F-4001): T-MAX Developer small tank 20C at the
    # 9 1/2 min normal (interpolated 10/7 min, K1) - AND RE-ANCHORED
    # by -0.764 log H, because the lane caught the sheet's curve
    # artwork sitting ~2.3 stops off its own stated EI (K2, a
    # publisher defect: the Plus-X control from the same tracer
    # lands at +0.017, so the method is not the cause). The shape
    # is the sheet's; the axis is the sheet's own EI 1000 statement
    # applied as K2 prescribes. Speed here is anchored, NOT
    # emergent - the one stock on the shelf whose ISO check is a
    # wiring assertion rather than a discovery, recorded as such.
    # Grain: rms 18, D-76-based (K3) against a T-MAX-developer
    # curve - the sheet's own pairing, flagged.
    "p3200": dict(
        fog=0.0, dmax=2.338, grain_um2=1.350, sens=_PAN,
        curve=(
            (-3.864, -3.764, -3.564, -3.364, -3.164, -2.964,
             -2.764, -2.564, -2.364, -2.164, -1.964, -1.764,
             -1.564, -1.364, -1.164, -0.964, -0.764, -0.564),
            (0.289, 0.293, 0.305, 0.332, 0.371, 0.432, 0.541,
             0.682, 0.828, 0.971, 1.107, 1.239, 1.377, 1.529,
             1.705, 1.902, 2.117, 2.338))),
    # ILFORD FP4 PLUS: ILFOTEC HC (1+31) 8 min 20C - the sheet's
    # own tabulated normal for the rated EI 125 (K7, vector trace
    # cross-checked against the current Nov 2018 raster to 0.005
    # D). The sheet plots RELATIVE log exposure; the axis here is
    # anchored by K8's explicit offset (-3.4235) so the speed
    # point sits at ISO 125 BY CONSTRUCTION - an imposed
    # assumption, named, with the translation-invariant ISO 6
    # contrast criterion (+0.041 D) as the real validation. Grain:
    # rms 10 from the MOTION-PICTURE coating in D-96 (F1) - three
    # developers across three properties (curve ILFOTEC HC, speed
    # ID-11, grain D-96), the K8 mismatch carried openly.
    "fp4": dict(
        fog=0.0, dmax=1.862, grain_um2=0.417, sens=_PAN,
        curve=(
            (-3.024, -2.824, -2.624, -2.424, -2.224, -2.024,
             -1.824, -1.624, -1.424, -1.224, -1.024, -0.824,
             -0.624, -0.424, -0.224, -0.024, 0.176, 0.376, 0.576),
            (0.095, 0.095, 0.097, 0.117, 0.181, 0.271, 0.379,
             0.496, 0.630, 0.781, 0.936, 1.091, 1.259, 1.425,
             1.570, 1.693, 1.788, 1.843, 1.862))),
    # ILFORD HP5 PLUS: ILFOTEC HC (1+31) 6 1/2 min 20C, the rated
    # EI 400 normal (K10); relative axis anchored by K11's offset
    # (-3.7746), contrast criterion -0.040 D. The plotted curve is
    # STILL CLIMBING at its cut - the sheet plots no shoulder, and
    # the table's clamp is a clamp at the edge of the plot, not a
    # claim the emulsion shoulders there (K10's warning carried).
    # Grain: rms 16, MP coating in D-96 (F2), same three-developer
    # mismatch as FP4 - though the D-96 gamma aim 0.65-0.70
    # matches this curve's traced 0.64-0.65 gradient, so the
    # CONTRAST states are close even where the developers are not.
    "hp5": dict(
        fog=0.0, dmax=2.072, grain_um2=1.067, sens=_PAN,
        curve=(
            (-3.375, -3.175, -2.975, -2.775, -2.575, -2.375,
             -2.175, -1.975, -1.775, -1.575, -1.375, -1.175,
             -0.975, -0.775, -0.575, -0.375, -0.175, 0.025, 0.225),
            (0.177, 0.185, 0.210, 0.254, 0.320, 0.408, 0.521,
             0.654, 0.791, 0.922, 1.052, 1.181, 1.309, 1.437,
             1.564, 1.691, 1.817, 1.944, 2.072))),
}

_CURVE_KEYS = ("fog", "dmax", "gamma", "ht", "wt", "ws")

# THE DEVELOPER'S HAND (organ 5, developer-hand dossier). Each
# stock's contrast-vs-development-time table for the SAME developer
# and condition its characteristic curve was fit at, traced
# vector-exact from the sheets' own CI figures (lane I: curve
# identity by stroke signature, calibration residuals 0.0002-0.007
# in data units) - except 5222's, whose five gammas are PRINTED on
# the curves themselves and fit a linear time law to 0.007. The
# lane's load-bearing derivation rides here: every
# Kodak-recommended 20 C time lands on CI 0.553-0.571, so normal
# development IS CI ~ 0.56 and push/pull is a displacement along
# the stock's own curve. `normal` is the CI at the tabulated
# normal time; ci= requests scale the family's gamma by
# ci/normal, span fixed - the properly worked negative at a
# different contrast aim, bounded by the table's own traced span
# because extrapolating a development curve is inventing one.
# The measure is the stock's own: contrast index (diffuse visual)
# for the Kodak four, gamma (Status M blue) for 5222.
CONTRAST = {
    # FILM-I1: T-MAX Developer small tank 20C, the condition the
    # curve constants name; 6 min -> 0.554 confirms the printed
    # aim CI 0.56 to 0.006
    "trix": dict(normal=0.554, minutes=6.0,
                 developer="T-MAX small tank 20C",
                 curve=((5.42, 0.520), (6.0, 0.554), (6.5, 0.584),
                        (7.0, 0.614), (8.0, 0.673), (9.0, 0.737),
                        (10.0, 0.802), (10.83, 0.856))),
    # FILM-I8/I9: D-96 21C, gammas PRINTED on the sheet's curves;
    # the linear law gamma = 0.0693 t + 0.216 holds to 0.007
    "5222": dict(normal=0.66, minutes=6.5, developer="D-96 21C",
                 curve=((4.0, 0.50), (5.0, 0.56), (6.5, 0.66),
                        (9.0, 0.84), (12.0, 1.05))),
    # FILM-I6: T-MAX RS large tank 20C, normal 8 3/4 min
    "tmax100": dict(normal=0.561, minutes=8.75,
                    developer="T-MAX RS large tank 20C",
                    curve=((8.70, 0.560), (9.0, 0.569),
                           (10.0, 0.599), (12.0, 0.659),
                           (14.0, 0.724), (16.5, 0.820))),
    # FILM-I7: D-76 small tank 20C, normal 7 1/2 min
    "tmax400": dict(normal=0.568, minutes=7.5,
                    developer="D-76 small tank 20C",
                    curve=((5.29, 0.420), (6.0, 0.466),
                           (7.0, 0.532), (7.5, 0.568),
                           (8.0, 0.606), (9.0, 0.689),
                           (10.0, 0.778))),
}

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
    missing = ([] if "curve" in st else
               [k for k in _CURVE_KEYS if k not in st])
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


def _dev_factor(name, ci):
    """Validate a ci= request against the stock's own contrast
    table and return the gamma multiplier ci/normal."""
    _stock(name)
    if name not in CONTRAST:
        if name == "collodion":
            raise ValueError(
                "collodion contrast is not a development-time "
                "dial: 45-90 s develop 'equally satisfactory' "
                "(lane I18) - halide loading and intensification "
                "set its gamma")
        if name in ("manchester", "hd22"):
            raise ValueError(
                f"{name!r} has no contrast-vs-development table: "
                "the plates' H&D ratio law raises Dmax with "
                "development, and the crystal ceiling a scaled "
                "plate needs is not yet declared - the capacity "
                "question stands")
        raise ValueError(
            f"{name!r} has no traced contrast-vs-development "
            "table: its sheet's CI curves await a tracing lane, "
            "and extrapolating another stock's would contradict "
            "the per-stock spread the dossiers measured")
    c = CONTRAST[name]
    cis = [p[1] for p in c["curve"]]
    if not min(cis) <= ci <= max(cis):
        raise ValueError(
            f"{name}'s sheet traces contrast only over "
            f"{min(cis):g}..{max(cis):g} ({c['developer']}) and is "
            f"silent at {ci:g} - extrapolating a development curve "
            "is inventing one")
    return ci / c["normal"]


def minutes_for(name, ci):
    """The development time the stock's own table asks for a
    contrast aim - the darkroom notebook's column, inverted."""
    _dev_factor(name, ci)       # validates stock and span, by name
    c = CONTRAST[name]
    ts = [p[0] for p in c["curve"]]
    cs = [p[1] for p in c["curve"]]
    return float(np.interp(ci, cs, ts))


def contrast_at(name, minutes):
    """The contrast the stock's own table gives at a development
    time, interpolated between the sheet's traced rows."""
    c = CONTRAST.get(name)
    if c is None:
        raise ValueError(
            f"{name!r} has no contrast-vs-development table")
    ts = [p[0] for p in c["curve"]]
    if not ts[0] <= minutes <= ts[-1]:
        raise ValueError(
            f"{name}'s table spans {ts[0]:g}..{ts[-1]:g} min and "
            f"is silent at {minutes:g}")
    return float(np.interp(minutes, ts, [p[1] for p in c["curve"]]))


def characteristic(logH, name, ci=None):
    """The stock's D-logH curve - H&D's straight line with a smooth
    toe and shoulder, on the absolute lux-second axis; or, for the
    stocks whose traced geometry IS the source (collodion's bent
    surface-silver curve), the sampled table itself, clamped at its
    ends because beyond the traced range the source says nothing.
    With ci=, the curve is the one developed to that contrast: the
    family's gamma scaled along the stock's own traced span."""
    st = _stock(name)
    x = np.asarray(logH, np.float64)
    if "curve" in st:
        if ci is not None:
            _dev_factor(name, ci)           # raises with the reason
        cx, cd = st["curve"]
        return st["fog"] + np.interp(x, cx, cd, left=cd[0],
                                     right=cd[-1])
    g = st["gamma"] * (1.0 if ci is None else _dev_factor(name, ci))
    a = g * st["wt"] * np.logaddexp(0.0, (x - st["ht"]) / st["wt"])
    span = st["dmax"] - st["fog"]
    return st["fog"] + a - st["ws"] * np.logaddexp(
        0.0, (a - span) / st["ws"])


def normal_highlight(name, ci=None):
    """The exposure H (lux-seconds) a normally exposed negative's
    highlights receive: H-740's latitude construction (FILM-N12)
    puts them ~1.0 D above the speed point, itself 0.10 above fog -
    so this solves the stock's own curve at D = fog + 1.10. The
    darkroom meters the camera against it. Collodion's traced
    curve tops at 1.57 net, so its highlight aim is capped a
    tenth under its own ceiling rather than solved past the
    table's reach."""
    st = _stock(name)
    aim = min(st["fog"] + 1.10, st["dmax"] - 0.10)
    xx = np.linspace(-5.0, 4.5, 6001)
    return float(10.0 ** np.interp(aim,
                                   characteristic(xx, name, ci=ci),
                                   xx))


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


# THE BATCH LOTTERY AND THE POURING FIELD (era-look, early-plates
# D10). A period dry plate was not a reference specimen: the same
# boxed product moved actinograph speed 7 to 18 between purchases
# ("without the slightest notification of alteration"), and one
# supposedly uniform strip varied D 1.335 to 0.820 across its own
# width - one part passing three times the light of another.
# `batch=` is the box you bought, an integer: it draws a
# deterministic speed shift inside the sourced order-of-magnitude
# bracket of +/-1 stop, and lays the hand that coated the plate -
# a smooth low-frequency pouring field multiplying the developed
# density (grain with it, like the intensifier's bath) at the
# sourced strip's worst-case relative amplitude, 0.24 about the
# mean. THE METER DOES NOT KNOW THE BATCH: the photographer rated
# the plate at its nominal speed and the box betrayed them, which
# is the era experience this dial exists to reproduce. Plates
# only: the Kodak stocks' machine coating has no sourced variance,
# collodion's pour is not in the record, and both refuse by name.
_BATCH_STOPS = 1.0          # D10: order-of-magnitude, +/-1 stop
_POUR_AMPLITUDE = 0.24      # D10 p.198: D 0.820..1.335 about 1.08


def _batch_draw(name, batch, shape):
    """The box and the hand: (speed shift in stops, pouring field
    over `shape`), both deterministic in the batch number."""
    if name not in ("manchester", "hd22"):
        raise ValueError(
            f"{name!r} has no sourced batch variance: the era's "
            "lottery is the dry plates' (early-plates D10) - "
            "Kodak's machine coating and collodion's pour are not "
            "in the record, and a lottery invented for them would "
            "be noise wearing a source's name")
    rng = np.random.default_rng(
        np.random.SeedSequence([0x1890, int(batch)]))
    shift = float(rng.uniform(-_BATCH_STOPS, _BATCH_STOPS))
    yy, xx = np.meshgrid(np.linspace(0.0, 1.0, shape[0]),
                         np.linspace(0.0, 1.0, shape[1]),
                         indexing="ij")
    field = np.zeros(shape, np.float64)
    for _ in range(3):
        fx, fy = rng.uniform(0.5, 2.0, 2)
        ph = rng.uniform(0.0, 2.0 * np.pi, 2)
        field += np.cos(2 * np.pi * fx * xx + ph[0]) \
            * np.cos(2 * np.pi * fy * yy + ph[1])
    peak = float(np.abs(field).max())
    return shift, 1.0 + _POUR_AMPLITUDE * field / max(peak, 1e-12)


# THE INTENSIFIER'S BATH (organ 5b, developer-hand I17): collodion's
# real contrast dial. Physical intensification deposits silver ON
# the developed image, so the whole output field - grain included -
# multiplies; the crystals were already developed when the bath
# touched them, so the counting statistics are untouched and
# "intensification results in higher granularity" (I17) EMERGES
# rather than being dialled. The pictorial 1:10 recipe at 180 s
# ships as the sourced Dmax ratio 2.6/1.57 = 1.656, refereed
# against the second printed number: 0.85 x 1.656 = 1.41 vs the
# printed gradient 1.37, within 3%. The 1:5 line-work regime
# refuses by SHAPE - the source says it straightens the shoulder,
# and a pure scale cannot honestly represent a shape change.
INTENSIFIERS = {"1:10": 2.6 / 1.57}


def _intensify_factor(name, intensify):
    if intensify is None:
        return 1.0
    st = _stock(name)
    if "curve" not in st:
        raise ValueError(
            f"{name!r} has no intensification model: metol-silver "
            "is the wet plate's bath (the dry plates were "
            "mercury-intensified in period, but no lane has traced "
            "what mercury does to their curves)")
    if intensify == "1:5":
        raise ValueError(
            "the 1:5 line-work intensification straightens the "
            "curve's shoulder (developer-hand I17: 'a satisfactory, "
            "long linear relationship') - a shape change no scale "
            "factor honestly represents; it waits for a traced "
            "intensified curve")
    if intensify not in INTENSIFIERS:
        raise ValueError(
            f"intensify {intensify!r} is not a sourced recipe: "
            + ", ".join(sorted(INTENSIFIERS))
            + " (metol-silver, 180 s, JIST 1998)")
    return INTENSIFIERS[intensify]


def negative(img, E, name, *, pitch_um=None, grain=True, seed=0,
             sheet=None, t=None, ci=None, intensify=None,
             batch=None, mtf=True, halation=True):
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
    `sheet=`. `intensify` (organ 5b) is the wet plate's bath: the
    developed field - grain and all - multiplies by the sourced
    recipe's factor, applied LAST because the bath touches silver
    the developer already made. `halation` (organ 10, on by
    default) sends light across the stock's own support and back
    off its rear surface, redistributing it into the ring the
    support's thickness and index dictate; stocks whose sheets do
    not state a base do not halate at all.
    """
    st = _stock(name)
    f = _intensify_factor(name, intensify)
    dose = np.maximum(_project(img, st) * E, 0.0)
    if mtf and pitch_um and name in _MTF_BW and dose.ndim == 2:
        # THE EMULSION'S SHARPNESS (organ 8): the stock's traced
        # MTF applied to the exposure at the negative's own pitch.
        # Needs the pitch, like grain - a curve evaluation without
        # a sheet geometry has no spatial physics to apply.
        dose = _mtf_apply(dose, _MTF_BW[name], float(pitch_um))
    if halation and pitch_um and dose.ndim == 2:
        # THE LIGHT THAT COMES BACK (organ 10): across the support
        # and off its back surface. Order against organ 8 does not
        # matter - both are linear transfers and commute exactly -
        # and the halo's hundreds of microns dwarf the emulsion's
        # own few-micron turbidity either way. A stock whose sheet
        # is silent on its base returns None here and does not
        # halate; see the shelf table for which those are.
        spec = _halation_for(name)
        if spec is not None:
            support_um, index, g = spec
            dose = _halation_apply(dose, pitch_um=float(pitch_um),
                                   support_um=support_um,
                                   index=index, strength=g)
    pour = None
    if batch is not None:
        shift, pour = _batch_draw(name, batch, dose.shape)
        dose = dose * 2.0 ** shift
    if t is not None:
        dose = dose * 2.0 ** -reciprocity(name, t)
    D = characteristic(np.log10(np.maximum(dose, 1e-30)), name,
                       ci=ci)
    if pour is not None:
        f = f * pour
    if not grain:
        return (D * f).astype(np.float32)
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
    return (f * (KAPPA * st["grain_um2"] / area) * n).astype(
        np.float32)


def normal_exposure(img, name, percentile=99.5, t=None, ci=None):
    """The camera's meter for a film stage: the E that places the
    scene's bright decile - as THIS stock sees it, through its own
    spectral projection - at the stock's normal highlight. With t,
    the meter raises E by the sheet's own reciprocity compensation
    (that is literally what the tables instruct the photographer to
    do), so the developed negative lands where it should; with ci,
    it meters against the curve as developed to that contrast."""
    st = _stock(name)
    ref = float(np.percentile(_project(img, st), percentile))
    E = normal_highlight(name, ci=ci) / max(ref, 1e-12)
    if t is not None:
        E *= 2.0 ** reciprocity(name, t)
    return E


def transmit(D):
    """The printing light: what the negative passes to the paper."""
    return np.power(10.0, -np.asarray(D, np.float64)).astype(np.float32)
