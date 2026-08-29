"""The historic processes: dose to deposit to density, sourced.

Carved verbatim from the darkroom's develop monolith on
2026-08-28 (the extraction is a text change with no licence to
touch a number). The constants carry the standing findings of the
darkroom's optics-film-process dossier ON PURPOSE: platinum's
dmax 1.30 is contradicted (~1.45 in Ware's own measurements, three
page-pinned statements), the salt/albumen scale ordering is
inverted against Reilly and Ware both, and the pigment absorb
0.88 below has drifted from its sibling HUE_K = 0.92 against the
constant's own comment contract. The reconciliation organ fixes
these as a JOINT fit against Ware's tables - the dossier proved a
one-number edit worsens the exposure-scale overshoot - and until
then the numbers ship exactly as the darkroom shipped them, with
this paragraph as the ledger entry.
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


#
#   dmax   reflection density the process can reach - how black its black
#   toe    shadow compression; above 1 the deposit starts slowly
#   speed  sensitivity, how much dose it takes to get going
#   absorb per-channel absorption of the deposited substance
#   base   the stock each process is traditionally coated on
PROCESSES = {
    # Prussian blue: absorbs hard through the red, barely touches blue
    "cyanotype": dict(dmax=1.45, toe=1.35, speed=1.00,
                      absorb=(1.00, 0.62, 0.14), base="#eef2ef"),
    # silver in a ferric process - warm brown, so blue is taken out
    "vandyke":   dict(dmax=1.50, toe=1.15, speed=1.10,
                      absorb=(0.72, 0.92, 1.00), base="#f4ecdc"),
    # platinum metal in the fibre: a famously long straight scale, low
    # toe, and a modest dmax - the reason platinotypes look luminous
    "platinum":  dict(dmax=1.30, toe=0.85, speed=0.90,
                      absorb=(0.95, 0.97, 1.00), base="#f1ece0"),
    # silver chloride printed out: reddish through to mauve, short dmax
    "salt":      dict(dmax=1.20, toe=1.25, speed=0.85,
                      absorb=(0.66, 0.94, 1.00), base="#f3e8d4"),
    # albumen sits on the surface rather than in it, so it goes deeper
    "albumen":   dict(dmax=1.70, toe=1.20, speed=1.00,
                      absorb=(0.72, 1.00, 0.86), base="#f6eeda"),
    # modern silver gelatin: neutral, and blacker than any of the above
    "silver":    dict(dmax=2.10, toe=1.00, speed=1.00,
                      absorb=(1.00, 1.00, 1.00), base="#fbfbfa"),
    # gum bichromate carries whatever pigment you grind into it
    "gum":       dict(dmax=1.35, toe=1.60, speed=0.80,
                      absorb=(1.00, 1.00, 1.00), base="#f2ecdf",
                      pigment="#243447"),
}


def process_print(neg, E, name, *, pigment=None, contrast=1.0, dmax_mul=1.0):
    """Expose, develop and read a print in one of the historic processes."""
    pr = PROCESSES[name]
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
    absorb = np.array(pr["absorb"], np.float32)
    if pigment or pr.get("pigment"):
        # a ground pigment absorbs the complement of its own colour
        absorb = 1.0 - _hex(pigment or pr["pigment"]) * 0.88
    t = np.power(10.0, -(d[..., None] * absorb))       # optical density
    return (_hex(pr["base"]) * t).astype(np.float32)
