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
#       sens = (r, g, b)          the layer's share of each render
#                                 channel, from the spectral figure
#       dye = (r, g, b)           the developed dye's density per
#                                 unit dye-amount in each channel,
#                                 dye[primary] = 1, from the
#                                 spectral-dye-density figure
#       grain_um2                 granularity-equivalent, DECLARED
#   ),
#   base = (r, g, b)              per-channel D-min: base + mask
# )
COLOUR_FILMS = {}

# print stocks share the structure; their `lad_aim` is the published
# print-film aim densities and the negative's lad aims ride
# LAD_NEGATIVE below
PRINT_FILMS = {}

# the LAD patch's aim densities on the camera negative (Status M),
# from the FILM-P lane when it lands
LAD_NEGATIVE = None


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
    a = layer["gamma"] * layer["wt"] * np.logaddexp(
        0.0, (np.asarray(logH, np.float64) - layer["ht"]) / layer["wt"])
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
    assembled print response."""
    if LAD_NEGATIVE is None:
        raise ValueError(
            "the LAD negative aims have not landed (FILM-P lane): "
            "a printer calibrated by taste is not a calibration")
    pf = _stock(PRINT_FILMS, print_name, "FILM-P")
    t_lad = 10.0 ** -np.asarray(LAD_NEGATIVE, np.float64)
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
