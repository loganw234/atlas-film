"""The historic processes: dose to deposit to density, sourced.

Carved verbatim from the darkroom's develop monolith on 2026-08-28;
RECONCILED against the optics-film-process dossier's round-2 verdict
table the same day, as the joint fit the dossier demanded (raising
platinum's dmax alone worsened the scale overshoot 2.107 -> 2.164,
so dmax and toe moved together). The referee is
`atlas_film/sensitometry.py` - Ware's own exposure-range convention
read off this module's actual prints - and the fit's tests were
watched failing at the shipped values before a constant moved
(`tests/test_the_processes_answer_to_ware.py` carries the record).
What no source licenses stays put: vandyke, albumen, salt and gum
dmax are wanted, not invented, and every remaining absence is the
medium ledger's (`atlas-darkroom/docs/film-extraction.md`).
"""

import numpy as np


def _hex(c):
    """A colour a person typed, in either length CSS allows.

    Three-digit shorthand used to raise deep inside numpy: the slices
    (0,2,4) on "#123" give "12", "3" and "", and int("", 16) is a
    ValueError with nothing in the message about hex stops. The root
    CLI's own help for --bg-stops documents "#000,#123,#fff" as the
    example, so following the documentation was the way to hit it -
    and the same parser reads --ink, --split-lo/hi and every custom
    tone stop.
    """
    c = c.lstrip("#").strip()
    if len(c) == 3:
        c = "".join(ch * 2 for ch in c)
    if len(c) != 6:
        raise ValueError(
            f"{c!r} is not a colour: hex stops are three or six "
            f"digits, like #123 or #8a5c34")
    return np.array([int(c[i:i + 2], 16) / 255.0 for i in (0, 2, 4)], np.float32)


# HOW COMPLETELY A COLOUR SPARES ITS OWN BAND. One number with two
# duties: the tricolour split weighs each layer by `1 - hue*HUE_K`,
# and the pigment-absorb lines - process_print's below and the
# tricolour's - darken by `1 - colour*HUE_K`, both statements that a
# saturated colour is not QUITE transparent to itself. The name was
# coined "because two places need the same number and one of them is
# a calibration that has to follow if it moves" - and the calibration
# did not follow: this module's line drifted to a literal 0.88
# against 0.92 (the darkroom's findings queue #15; inventory OPT-G7
# records the site at 0.92, so the drift postdates the survey).
# Reconciled 2026-08-28: the constant lives here, up the import
# direction the package already has, both sites read the name, and
# pigments re-exports it. A comment saying "has to follow" is a
# hope; an import is a mechanism.
HUE_K = 0.92


# THE OPAQUE-DISC COEFFICIENT IS log10(e), NOT A NUMBER CHOSEN BY
# EYE. Nutting's relation reads density off a random scatter of
# opaque grains: N grains of mean projected area a in aperture A
# transmit T = exp(-N*a/A), so D = log10(e)*N*a/A. Ware's
# Argyronomicon states the same equation as D = kappa*N*a/A with
# kappa "for approximate purposes ... taken as unity" (p. 205,
# SS21.2); this is the exact value his approximation stands in for,
# and the grain statistics below use it in both directions - count
# from density, density from count - so the mean is preserved to the
# digit whatever kappa's value.
KAPPA = 0.4342944819032518


#
#   dmax      reflection density the process can reach - how black its black
#   toe       shadow compression; above 1 the deposit starts slowly
#   speed     sensitivity, how much dose it takes to get going
#   absorb    per-channel absorption of the deposited substance
#   base      the stock each process is traditionally coated on
#   grain_um2 mean projected area of ONE image particle, in square
#             microns - the number that makes density a count
#             (film-grain dossier, whose synthesis table is this
#             list's audit trail). Absent = no source has measured
#             this process's particle, and grain REFUSES rather than
#             inventing one. Derived from stated DIAMETERS as
#             pi*(d/2)^2, because the diameter is what the sources
#             state:
#             - salt, vandyke: photolytic (print-out) silver, 10-100
#               nm with Ware's own median r = 10 nm (Argyronomicon
#               pp. 206, 212; W3/W5, salt's <=30 nm bound W9),
#               d = 20 nm -> 3.14e-4. A salt print's grainlessness is
#               a PREDICTION here, not a setting.
#             - albumen: the one photolytic process actually MEASURED
#               - Reilly's TEM size-frequency histograms, mean 8.1 nm
#               on a nineteenth-century print (202 particles; 5.5 nm
#               on a laboratory one; dossier C1) -> 5.2e-5.
#               Measurement outranks the class median.
#             - platinum: single ellipsoidal Pt nanoparticles, 15-25
#               nm by SEM at 500,000x (Ravines, Platinomicon p. 130;
#               W7; the 2001 film-not-particles reading resolved at
#               C9), d = 20 nm -> 3.14e-4 - why a platinotype is
#               smooth is now physics, not reputation.
#             - cyanotype: Prussian blue measured IN cyanotypes at
#               20-100 nm on fibre surfaces, 20-50 nm within (Sentoku
#               2025, C11); d = 50 nm, the within-fibre ceiling both
#               ranges share -> 1.96e-3.
#             - silver (an enlarging paper, like its Foma dmax
#               source): developed-out B&W papers carry filamentary
#               bundles "ranging around 0.5 micrometers" (Weaver,
#               corroborated by Moon & Curran TEM and Ali 2012;
#               B14/C7/C6, W4's micron ceiling), d = 0.5 um -> 0.196.
#               Crosscheck through Saunders' inversion: Tri-X FILM's
#               published rms 17 (B1/B4) gives a = 1.2 um2 - film
#               coarser than paper, both micron-order, the 1935 law
#               and the 2016 datasheet agreeing.
#             - gum: the pigment's aggregate, not its 95 nm primary
#               (C13): the class spans 1-10 um (umber's stated range,
#               ivory black to ~7, lamp-black aggregates "microns";
#               C12), d = geometric mean 3.2 um -> 8.0. The ONE
#               process whose grain is visible at print pitch -
#               "grainy or granular" is the conservation literature's
#               own description - and the constant describes the
#               default pigment class, not whatever ink a caller
#               grinds.
PROCESSES = {
    # Prussian blue: absorbs hard through the red, barely touches blue
    "cyanotype": dict(dmax=1.45, toe=1.35, speed=1.00,
                      absorb=(1.00, 0.62, 0.14), base="#eef2ef",
                      grain_um2=1.96e-3),
    # silver in a ferric process - warm brown, so blue is taken out
    "vandyke":   dict(dmax=1.50, toe=1.15, speed=1.10,
                      absorb=(0.72, 0.92, 1.00), base="#f4ecdc",
                      grain_um2=3.14e-4),
    # platinum metal in the fibre - the reason platinotypes look
    # luminous. Reconciled 2026-08-28 to Ware's own densitometry pair
    # (Platinomicon: Dmax "~1.45" p. 235, "1.40-1.45" on Western paper
    # p. 188, Klimek's 1.42 p. 186; exposure scale ~1.9 p. 235, his
    # Table 7.1 band 1.5-1.8 at p. 158): dmax was 1.30, contradicted-
    # low, and toe 0.85 read scale 2.107 - inside his PALLADIUM band
    # (2.0-2.4). The pair moved JOINTLY because dmax 1.45 alone reads
    # 2.164; toe 1.01 solves the scale to 1.90 measured. "A famously
    # long straight scale" was this comment's old claim - true against
    # silver-gelatin enlarging papers, false against palladium, and
    # the third measured number (gamma ~0.96) is the one this
    # single-exponent curve cannot also hit; the dossier carries both.
    "platinum":  dict(dmax=1.45, toe=1.01, speed=0.90,
                      absorb=(0.95, 0.97, 1.00), base="#f1ece0",
                      grain_um2=3.14e-4),
    # silver chloride printed out: reddish through to mauve. Scale
    # reconciled 2026-08-28: Reilly (ch. 7, reporting Hubl) has plain
    # salted paper the longest-scaled of its family, and Ware sorts it
    # into his long-range class BESIDE platinum-palladium (Cyanomicon
    # pp. 213, 219) - the shipped toe 1.25 read 1.568, SHORTER than
    # albumen's 1.741, inverted against both primaries. toe 0.95
    # gives salt its classmate's measured scale (1.90); class
    # membership is the only numeric content the sources offer, since
    # Ware's 2-2.4 is a negative range and this curve maps scene to
    # print. dmax 1.20 stays: the one measurement in hand (koraks
    # 2023, 1.35) is a gold-toned print and non-archival - abstention
    # over invention.
    "salt":      dict(dmax=1.20, toe=0.95, speed=0.85,
                      absorb=(0.66, 0.94, 1.00), base="#f3e8d4",
                      grain_um2=3.14e-4),
    # albumen sits on the surface rather than in it, so it goes deeper
    "albumen":   dict(dmax=1.70, toe=1.20, speed=1.00,
                      absorb=(0.72, 1.00, 0.86), base="#f6eeda",
                      grain_um2=5.2e-5),
    # modern silver gelatin: neutral, and blacker than any of the above
    "silver":    dict(dmax=2.10, toe=1.00, speed=1.00,
                      absorb=(1.00, 1.00, 1.00), base="#fbfbfa",
                      grain_um2=0.196),
    # gum bichromate carries whatever pigment you grind into it
    "gum":       dict(dmax=1.35, toe=1.60, speed=0.80,
                      absorb=(1.00, 1.00, 1.00), base="#f2ecdf",
                      pigment="#243447", grain_um2=8.0),
}


def process_print(neg, E, name, *, pigment=None, contrast=1.0, dmax_mul=1.0,
                  grain=False, pitch_um=None, seed=0):
    """Expose, develop and read a print in one of the historic processes.

    `grain=True` makes the deposit a COUNT: each pixel's density
    becomes a Poisson realisation of the number of image particles a
    pixel of that physical size holds, through Nutting's relation both
    ways (Argyronomicon SS21.2/21.4 - Saunders' treatment: sigma_N =
    sqrt(N), so sigma_D = sqrt(kappa*a*D/A)). It needs `pitch_um`
    because a count lives on a sheet, and it needs the process to
    carry a SOURCED particle area `grain_um2` - a process whose
    particle no source has measured refuses rather than inventing
    one, the same abstention the absorption tables keep.

    Declared simplifications, from the dossier's own synthesis: the
    particle area is CONSTANT per process, while printing-out papers
    grow particle size along with count as exposure rises (Reilly's
    TEM, dossier C2) and real emulsions carry size distributions -
    corrections of order the size spread, on processes whose grain
    is invisible at print pitch anyway; and the field is white below
    the pixel, which is honest while pixels dwarf particles (the
    published spectrum rolls off as k^-3 far above any print pitch,
    dossier A14).
    """
    pr = PROCESSES[name]
    if grain:
        if not pitch_um:
            raise ValueError(
                "grain is a count of particles on a sheet, and the "
                "sheet belongs to the caller: pass pitch_um")
        if "grain_um2" not in pr:
            raise ValueError(
                f"no sourced particle area for {name!r}: grain would "
                "be an invented number, and abstention is not "
                "invention (the dossier records what is wanted)")
    dose = (0.299 * neg[..., 0] + 0.587 * neg[..., 1]
            + 0.114 * neg[..., 2]) * E * pr["speed"]
    # THERE IS NO SUCH THING AS NEGATIVE LIGHT, and letting one through
    # does not produce a dark pixel - it produces NaN, which casts to
    # black. `1 - exp(-dose)` goes negative below zero, and a negative
    # base raised to a FRACTIONAL power is undefined. Every process here
    # has a fractional toe except silver, whose toe is exactly 1.0, so a
    # single bad sample in the negative came out as six black pixels and
    # one correct one - found on the seven-process strip, six pixels in
    # 9.7 million, invisible in review and permanent in print.
    dose = np.maximum(dose, 0.0)
    # dose -> deposited substance. The exponential saturates on its own,
    # which is the shoulder; the toe exponent slows the start.
    # CONTRAST NOW MEANS CONTRAST. `u = 1 - exp(-dose)` is in (0,1),
    # so a SMALLER exponent compresses it toward 1: the old form
    # divided the toe by the dial, which made a LARGER `contrast`
    # SOFTEN the curve. Measured on platinum (dmax 1.30, toe 0.85),
    # contrast index in dD per decade under the old form: 0.667 ->
    # 1.251, 1.0 -> 1.007, 1.5 -> 0.794, 2.0 -> 0.662. The only way
    # to steepen was to go below 1.0, on a slider centred at 1.0.
    #
    # It did lift a thin print - density rises everywhere under dmax -
    # which is why it read as a cure; but it bought that by removing
    # gradient, so the print went muddier rather than punchier. Every
    # shipped look sat above 1.0 with a comment about fighting
    # flatness, and was therefore softer than the default it was
    # raised from.
    #
    # THIS CHANGES RENDERS, and the migration is exact: `toe/c` became
    # `toe*c`, so a look authored at c reproduces at 1/c. Every value
    # this repository ships has been converted. Anything held outside
    # it - a saved shot, a stored recipe - needs the same reciprocal.
    # Audit-8-17 finding 9.
    d = (pr["dmax"] * dmax_mul) * (1.0 - np.exp(-dose)) ** (pr["toe"] * contrast)
    if grain:
        # the deposit, realised as the count it physically is: mean
        # particle number per pixel from Nutting, a Poisson draw
        # about it, and back to density through the same relation -
        # so the mean is exact and the fluctuation is Saunders' law
        # with no dial anywhere. One population of particles: the
        # per-channel colouring below multiplies the SAME realised
        # deposit, which is why grain is monochrome in a single
        # process, as it is on a real print.
        area = float(pitch_um) ** 2
        lam = d.astype(np.float64) * area / (KAPPA * pr["grain_um2"])
        rng = np.random.default_rng(seed)
        d = (KAPPA * pr["grain_um2"] / area) * rng.poisson(lam)
    absorb = np.array(pr["absorb"], np.float32)
    if pigment or pr.get("pigment"):
        # a ground pigment absorbs the complement of its own colour
        absorb = 1.0 - _hex(pigment or pr["pigment"]) * HUE_K
    t = np.power(10.0, -(d[..., None] * absorb))       # optical density
    return (_hex(pr["base"]) * t).astype(np.float32)
