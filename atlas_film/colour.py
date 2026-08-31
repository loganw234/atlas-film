"""Colour by the cinema chain: three sheets deep, printed by LAD.

A colour negative here is three of the camera-stock sheets stacked
(PLAN.md organ 7): blue-, green- and red-sensitive layers, each its
own crystal field with its own curve, seed and grain. What is new
is the ASSEMBLY:

  layer exposure   H_L = (sens_L . rgb) * E      the 3x3 matrix that
                   is this RGB system's declared fidelity ceiling
  layer dye        d_L = curve_L(log10 H_L)      the film family of
                   films.py, in dye-amount units (density at the
                   layer's primary channel)
  channel density  D_c = base_c + sum_L d_L * dye_L[c]

The integral orange mask is not modelled twice: the per-channel
`base` floors and the published curves already carry it, and the
dye triples come from a DIFFERENT figure (spectral dye density)
than the curves - which is what makes the assembly referee a real
check: driven with a neutral sweep, the assembled D_c must
reproduce the datasheet's own three published curves.

GRAIN IS DECLARED ONE RUNG DOWN. Dye clouds are not opaque discs -
Dutton's derivation (film-grain dossier A9) excludes "partially
transparent cells such as in a dye image" - so each layer's
`grain_um2` is a GRANULARITY-EQUIVALENT area matched to the sheet's
published rms curves through the same thinning machinery, a noise
constant and not a measured cloud. The B&W sheets' crystal-literal
claim does not transfer, and the rung is named here and at the
constants.

PRINTING IS LAD. The print film's three layers expose through the
negative's per-channel transmittance times per-channel printer
lights, and the lights are solved the way laboratories solve them:
a mid-scale LAD grey at the negative's published aim densities must
print to the print film's published aim densities. No taste, no
auto-meter - a calibration with sources on both ends.

Constants ship sourced or refuse by name: the tables below are
EMPTY until the FILM-C (negatives) and FILM-P (print film + LAD)
lanes land, and every entry point refuses an unshelved stock with
the lane that owes it.
"""

import numpy as np

from atlas_film import emulsion
from atlas_film.processes import KAPPA

# stock -> dict(
#   layers = (layer_r, layer_g, layer_b) each dict(
#       gamma, ht, wt, ws, span   the films.py curve family, in
#                                 dye-amount units (span replaces
#                                 dmax-fog; the layer starts at 0 -
#                                 the mask lives in `base`)
#       curve = (xs, ds)          OR a sampled table instead of the
#                                 family (see _layer_curve)
#       sens = (r, g, b)          the layer's share of each render
#                                 channel - identity DECLARED as
#                                 this RGB system's fidelity ceiling
#       dye = (r, g, b)           the developed dye's density per
#                                 unit dye-amount in each channel,
#                                 dye[primary] = 1, from the
#                                 spectral-dye-density figure
#       grain_um2                 granularity-equivalent, DECLARED
#   ),
#   base = (r, g, b)              per-channel D-min: base + mask
# )
#
# Camera negatives: layer families fit to the pointwise inversion
# dyes = Minv(D - base) of the traced characteristic curves, dye
# matrix from the spectral-dye-density plate (a DIFFERENT figure -
# the reassembly referee in the tests is non-circular). Assembly
# residual at the traced points: 50d worst 0.046 D, 5219 worst
# 0.036 D - within the read-off tolerance of the plates.
#
# grain_um2 is granularity-equivalent (dossier colour-films lane C:
# Dutton's crystal-literal derivation excludes dye images), matched
# to each sheet's published rms curve at mid density. The rung-
# consistency check the FILM-F lane later delivered: measured dye
# clouds in Kodak chromogenic PAPERS run 1.25-4 um across (Weaver &
# Long 2009, film-not-paper caveat named), and 5219's largest
# equivalent area, yellow 1.504 um^2, is a 1.38 um disc - inside
# the measured range. A noise constant, but a plausible one.
COLOUR_FILMS = {
    # KODAK VISION3 50D 5203 (FILM-C lane: curves, spectral dye
    # density, rms 5/5/6 at D-min+1.0, 48 um aperture)
    "50d": dict(
        base=(0.17, 0.58, 0.86),
        layers=(
            dict(gamma=0.55, ht=-1.98, wt=0.40, ws=0.60, span=1.78,
                 sens=(1.0, 0.0, 0.0), dye=(1.0, 0.1325, 0.1084),
                 grain_um2=0.088),
            dict(gamma=0.50, ht=-2.08, wt=0.28, ws=0.60, span=2.31,
                 sens=(0.0, 1.0, 0.0), dye=(0.0816, 1.0, 0.0204),
                 grain_um2=0.088),
            dict(gamma=0.50, ht=-2.00, wt=0.32, ws=0.30, span=2.02,
                 sens=(0.0, 0.0, 1.0), dye=(0.0303, 0.0707, 1.0),
                 grain_um2=0.287),
        )),
    # KODAK VISION3 500T 5219 (FILM-C lane; its sheet prints
    # "Densitometry: ECN-2" where 50D prints "Status M" - recorded,
    # both treated as the ECN-2 process's own control densitometry)
    "5219": dict(
        base=(0.18, 0.58, 0.84),
        layers=(
            dict(gamma=0.45, ht=-2.98, wt=0.36, ws=0.30, span=1.77,
                 sens=(1.0, 0.0, 0.0), dye=(1.0, 0.1412, 0.0588),
                 grain_um2=0.198),
            dict(gamma=0.45, ht=-2.94, wt=0.36, ws=0.30, span=1.77,
                 sens=(0.0, 1.0, 0.0), dye=(0.0808, 1.0, -0.0303),
                 grain_um2=0.204),
            dict(gamma=0.65, ht=-2.80, wt=0.44, ws=0.60, span=2.15,
                 sens=(0.0, 0.0, 1.0), dye=(0.0101, 0.1212, 1.0),
                 grain_um2=1.504),
        )),
}

# print stocks share the structure; their `lad_aim` is the published
# print-film aim densities (Status A) and the negative's aims ride
# LAD_NEGATIVE below.
#
# 2383's layers carry sampled TABLES, not families: at gamma five
# the family missed the sourced acceptance (one printer light =
# 0.025 log H, FILM-P-13) by a factor of six, and the traced
# geometry IS the source. Construction, declared: channel curves
# from the traced logH-at-D spine plus the low-density toe points,
# last chord continued to the published channel top (a flat clamp
# right after the last traced point is a jump that corrupts the
# inversion), pointwise inversion through the dye matrix, monotone
# enforcement (a layer cannot un-develop; the wobble lives above
# the traced range), knots placed adaptively to reproduce the
# inversion within 0.003 dye. Assembly at all 27 traced points:
# worst 0.003 log H = 0.12 printer lights.
PRINT_FILMS = {
    # KODAK VISION Premier Color Print Film 2383 (FILM-P lane)
    "2383": dict(
        base=(0.04, 0.05, 0.10),
        lad_aim=(1.09, 1.06, 1.03),
        layers=(
            dict(curve=(
                (-1.300, -0.002, 0.000, 0.250, 0.500, 0.542, 0.822,
                 1.010, 1.136, 1.244, 1.352, 1.472, 1.636, 1.812,
                 2.080, 2.500),
                (0.000, 0.000, 0.011, 0.030, 0.095, 0.111, 0.361,
                 0.823, 1.306, 1.783, 2.275, 2.764, 3.263, 3.499,
                 3.875, 3.875)),
                 span=3.875, sens=(1.0, 0.0, 0.0),
                 dye=(1.0, 0.1205, 0.0892)),
            dict(curve=(
                (-1.300, -0.502, -0.500, -0.224, 0.004, 0.190,
                 0.250, 0.364, 0.448, 0.542, 0.666, 0.816, 0.940,
                 1.056, 1.192, 1.368, 1.474, 1.634, 1.812, 2.500),
                (0.000, 0.000, 0.006, 0.033, 0.049, 0.117, 0.158,
                 0.287, 0.376, 0.569, 0.809, 1.263, 1.702, 2.146,
                 2.579, 2.979, 3.065, 3.224, 3.437, 3.437)),
                 span=3.437, sens=(0.0, 1.0, 0.0),
                 dye=(0.0456, 1.0, 0.0810)),
            dict(curve=(
                (-1.300, -1.002, -0.498, -0.224, 0.000, 0.118,
                 0.250, 0.366, 0.510, 0.618, 0.724, 0.840, 0.992,
                 1.352, 1.494, 2.500),
                (0.000, 0.000, 0.032, 0.097, 0.252, 0.389, 0.651,
                 0.874, 1.355, 1.826, 2.296, 2.750, 3.177, 3.397,
                 3.508, 3.508)),
                 span=3.508, sens=(0.0, 0.0, 1.0),
                 dye=(0.0149, 0.0558, 1.0)),
        )),
}

# the LAD patch's aim densities on the camera negative, Status M
# (FILM-P lane: Kodak's Laboratory Aim Density control method)
LAD_NEGATIVE = (0.80, 1.20, 1.60)


def _stock(table, name, lane):
    if name not in table:
        raise ValueError(
            f"no such colour stock {name!r} on the shelf"
            + (": " + ", ".join(sorted(table)) if table else
               f" - the {lane} lane has not landed its constants yet, "
               "and a colour film invented for a named real stock "
               "would be a lie wearing a datasheet's name"))
    return table[name]


def _layer_curve(layer, logH):
    """A layer's dye-amount response. Two representations, chosen by
    what honesty demands: the camera negatives carry the film FAMILY
    (fit within traced tolerance, extrapolating gracefully into deep
    shadow), but the print film carries its traced curve as a
    SAMPLED TABLE - on a gamma-five curve the family missed by six
    printer lights, and the traced geometry IS the source, so
    parametrising it added error for no physical gain. Tables clamp
    at their ends: beyond the traced range the datasheet says
    nothing, and neither do we."""
    x = np.asarray(logH, np.float64)
    if "curve" in layer:
        cx, cd = layer["curve"]
        return np.interp(x, cx, cd, left=cd[0], right=cd[-1])
    a = layer["gamma"] * layer["wt"] * np.logaddexp(
        0.0, (x - layer["ht"]) / layer["wt"])
    return a - layer["ws"] * np.logaddexp(
        0.0, (a - layer["span"]) / layer["ws"])


def _develop_layers(rgb, E, stock, pitch_um, grain, seed, label):
    """The three sheets, each exposed through its own sensitivity
    row and developed on its own crystal field."""
    rgb = np.maximum(np.asarray(rgb, np.float64), 0.0)
    dyes = []
    for i, layer in enumerate(stock["layers"]):
        h = (rgb @ np.asarray(layer["sens"], np.float64)) * E
        d = _layer_curve(layer, np.log10(np.maximum(h, 1e-30)))
        if grain:
            if not pitch_um:
                raise ValueError(
                    "a negative's grain is a count of crystals on "
                    "the film: pass pitch_um")
            if "grain_um2" not in layer:
                raise ValueError(
                    f"{label} layer {i} has no granularity-equivalent "
                    "area: its sheet publishes no rms curve, and "
                    "abstention is not invention (the print film's "
                    "grain is exactly this gap)")
            p = d / layer["span"]
            n = emulsion.expose(p, float(layer["span"]),
                                layer["grain_um2"], float(pitch_um),
                                seed + i, label=f"{label}[{i}]")
            d = (KAPPA * layer["grain_um2"] / float(pitch_um) ** 2) * n
        dyes.append(d)
    return dyes


def _assemble(dyes, stock):
    """base + the dye contributions, per channel."""
    out = np.empty(dyes[0].shape + (3,), np.float32)
    for c in range(3):
        acc = float(stock["base"][c])
        total = dyes[0] * stock["layers"][0]["dye"][c]
        for i in (1, 2):
            total = total + dyes[i] * stock["layers"][i]["dye"][c]
        out[..., c] = (acc + total).astype(np.float32)
    return out


def negative(rgb, E, name, *, pitch_um=None, grain=True, seed=0):
    """Expose a colour camera stock and develop it: the negative's
    per-channel Status M density field, mask and all."""
    st = _stock(COLOUR_FILMS, name, "FILM-C")
    return _assemble(
        _develop_layers(rgb, E, st, pitch_um, grain, seed, name), st)


def transmit(D):
    """The printing light, per channel: 10^-D."""
    return np.power(10.0, -np.asarray(D, np.float64)).astype(np.float32)


def lad_lights(neg_name, print_name):
    """Solve the printer: per-channel exposures such that the LAD
    grey at the negative's published aims prints to the print film's
    published aims - the laboratories' own calibration, with sources
    on both ends. Deterministic bisection per channel against the
    assembled print response.

    Note what this is NOT: the negative aims are the LAD control
    patch's nominal densities (H-61A), and the dossier's own
    warning applies - LAD sets the printer's centre, it is not a
    reference every scene must match. A particular stock's metered
    grey lands NEAR the patch, not on it; as traced, both Vision3
    stocks carry a B channel about 0.2 D thinner over G than the
    nominal patch spacing, so their greys print warm of neutral at
    LAD lights (recorded in the findings queue - the B traces
    deserve re-adjudication). Scene-to-scene neutrality is the
    trim's job: grey_lights below."""
    if LAD_NEGATIVE is None:
        raise ValueError(
            "the LAD negative aims have not landed (FILM-P lane): "
            "a printer calibrated by taste is not a calibration")
    t_lad = 10.0 ** -np.asarray(LAD_NEGATIVE, np.float64)
    return _solve_lights(t_lad, print_name)


def grey_lights(neg_name, print_name, E=None):
    """The grey-card timing: lights such that THIS stock's own
    metered 16% grey prints to the print film's neutral aims. The
    same solved mechanism as lad_lights pointed at the film instead
    of the control patch - real labs call the difference TRIM and
    apply it per scene (H-61B's printer-light quantum)."""
    if E is None:
        E = normal_exposure(np.ones((2, 2)), neg_name)
    grey = np.full((1, 1, 3), 0.18, np.float64)
    d = negative(grey, E, neg_name, grain=False)[0, 0]
    return _solve_lights(10.0 ** -np.asarray(d, np.float64),
                         print_name)


def _solve_lights(t_patch, print_name):
    pf = _stock(PRINT_FILMS, print_name, "FILM-P")
    t_lad = np.asarray(t_patch, np.float64)
    # the channels COUPLE - every layer sees every light through its
    # sensitivity row, every dye stains every channel - so the solve
    # is Gauss-Seidel over the three lights, each channel bisected
    # against the full assembled response with the others held. The
    # diagonal dominates (a layer's primary sensitivity and its
    # dye's primary stain), so this converges, and deterministically.
    lights = np.ones(3)
    for _ in range(16):
        prev = lights.copy()
        for c in range(3):
            aim = float(pf["lad_aim"][c])
            lo, hi = 1e-8, 1e8
            for _ in range(60):
                mid = float(np.sqrt(lo * hi))
                trial = lights.copy()
                trial[c] = mid
                d = float(positive(t_lad[None, None, :], print_name,
                                   lights=trial, grain=False)[0, 0, c])
                lo, hi = (mid, hi) if d < aim else (lo, mid)
            lights[c] = float(np.sqrt(lo * hi))
        if float(np.abs(lights - prev).max()) < 1e-10:
            break
    return lights


def positive(neg_T, name, *, lights=(1.0, 1.0, 1.0), pitch_um=None,
             grain=False, seed=0):
    """Print through the negative: the print film's three layers
    exposed by transmitted light times the printer's per-channel
    lights, assembled to the PRINT's density field."""
    pf = _stock(PRINT_FILMS, name, "FILM-P")
    light = np.asarray(neg_T, np.float64) * np.asarray(lights,
                                                       np.float64)
    return _assemble(
        _develop_layers(light, 1.0, pf, pitch_um, grain, seed, name),
        pf)


def normal_exposure(lum, name, percentile=90.0):
    """The colour camera's meter, anchored where the laboratories
    anchor: E such that the scene's highlight-referenced grey (18%
    of the bright decile) develops to the negative's green-channel
    LAD aim. The same discipline as lad_lights, pointed at the
    camera instead of the printer."""
    if LAD_NEGATIVE is None:
        raise ValueError("the LAD negative aims have not landed")
    _stock(COLOUR_FILMS, name, "FILM-C")
    ref = 0.18 * float(np.percentile(
        np.asarray(lum, np.float64).max(axis=-1)
        if np.ndim(lum) == 3 else np.asarray(lum, np.float64),
        percentile))
    aim = float(LAD_NEGATIVE[1])
    grey = np.full((1, 1, 3), ref, np.float64)
    lo, hi = 1e-8, 1e12
    for _ in range(80):
        mid = float(np.sqrt(lo * hi))
        d = float(negative(grey, mid, name, grain=False)[0, 0, 1])
        lo, hi = (mid, hi) if d < aim else (lo, mid)
    return float(np.sqrt(lo * hi))
