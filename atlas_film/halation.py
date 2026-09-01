"""Halation: the light that crosses the support and comes back.

Organ 10 (halation dossier, lanes N/Q/R). Light that has already
exposed the emulsion scatters within it, crosses the support, and
meets the support/air boundary at the back. Beyond the critical
angle it is totally internally reflected; below it, Fresnel still
returns a few percent. What returns re-exposes the emulsion at a
radial displacement set by the support's OWN thickness and index -
which is why a 1.2 mm glass plate halates in millimetres and a
130 um film base in tenths of one, and why the effect is the
defining signature of period plate photography.

THE GEOMETRY IS DERIVED, NOT DIALLED. The emulsion scatters the
light Lambertianly, so the flux into angle theta1 is
sin(2 theta1) d theta1; it refracts into the support at theta2 by
Snell, returns at r = 2 t tan(theta2), and the surface density
follows in a few lines:

    p(r) = (n2/n1)^2 T(theta1) R(theta2) cos^4(theta2) / (4 pi t^2)

- the cos^4 law arriving from a direction it was not expected
from. R is the Fresnel reflectance at the support's back face, T
the transmission into the support at the front one, and the
(n2/n1)^2 is exactly the share of the emulsion's hemisphere that
reaches the support at all. So the spread is a weak disc (a few
percent, at normal incidence) climbing to a peak at

    r_c = 2 t tan(asin(1/n2))

and decaying outward. NOT a discontinuity: Fresnel climbs to unity
CONTINUOUSLY as theta approaches the critical angle. But not a
smooth maximum either - 1 - R goes as sqrt(theta_c - theta), so
the profile arrives at its peak (7.9x the centre value in
acetate) with an INFINITE derivative. A cusp, which is what a
sharply defined ring edge looks like written down. The halo is
much broader than r_c alone suggests - only 6.9% of the returned
light falls inside r_c, 60% inside 2 r_c, 98% inside 10 r_c -
which is why period interiors veil rather than merely ring.

WHICH WAY THE INDEX STEP GOES MATTERS, and it is easy to miss.
Gelatin is 1.541, and acetate (1.480-1.505) and plate glass
(1.523) are both THINNER than that, so light grazes freely in the
support and the halo trails away without limit. ESTAR (1.65
in-plane) is DENSER, so
refraction at the emulsion/support interface compresses every ray
inward and the halo stops dead at 2 t tan(asin(n1/n2)) - a hard
outer edge no acetate stock has. The first version of this organ
assumed a support-Lambertian source and got ESTAR wrong by 7%
while being exact for the other two; see `reach_radius`.

There is no free shape parameter anywhere above: thickness and
index are datasheet facts and the profile follows. And it is not
this repo's invention - Cornu published it in 1890, the same year
as the plates on this shelf, as rho = 2e tan R with 2 rho = 3.578e
at n = 3/2; Law re-derived the same cos^4 kernel for television
in 1939; Dailliez and Hebert publish it again as a coated-print
PSF in 2022 and 2023. Three independent arrivals, and this
module's own derivation makes a fourth.

THE STRENGTH, by contrast, IS NOT SOURCED AND IS NOT INVENTED.
What share of the exposing light leaves the emulsion downward as
wide-angle scatter is the one quantity nobody has published: no
measured radial density profile across a halation ring in real
film, no halation-to-direct exposure ratio, no measured returning
fraction - across archive.org, the Kodak/Agfa/Fuji/3M patent
families, Kodak's H-1 and H-24 series, SMPTE, JOSA, IS&T, MDPI,
PMC, HathiTrust and Gallica (N28). So `halation=` is a FRACTION
of the sourced ceiling and belongs to the operator, who is told
so. The one photographic paper that would close it - Vendrovsky
& Pakoushko, J. Photogr. Sci. 12(2) 71-75, 1964, whose equations
assume "complete light diffusion in photographic layers" and
whose "experimental trials show the equations are valid" - is
known here only by its abstract, and is this organ's standing
acquisition target.

Stocks whose sheets are silent on their support refuse by name,
each with its own reason, in REFUSALS.

WHY THIS DOES NOT DOUBLE-COUNT ORGAN 8. A published MTF is
measured on real film including whatever halation it has, so two
spatial organs invite the question. The geometry answers it: K(f),
the transform of the unit-normalised spread, collapses fast - for
a 132 um acetate base it is +0.06 by 1 c/mm, -0.11 at 2, and
within 0.001 of zero from 5 c/mm up; for a 1.2 mm plate it is done
by 0.5 c/mm. Every traced MTF on the shelf carries its lowest
datum at or above 2.5 c/mm. So across the entire measured range
halation contributes no SHAPE, only the flat factor (1 - g),
and a curve normalised to unity divides exactly that factor out.
The organs are separable in frequency and the normalisation is the
seam. The uncomfortable corollary, stated rather than buried: the
published MTF therefore carries NO information about g either, so
the strength cannot be bounded from the curves.

APPLIED as a transfer with unity DC gain,

    H(f) = (1 - g) + g * K(f)

so halation REDISTRIBUTES light and never creates it. That matters
for more than tidiness: the stock's rated speed was measured on
film that halates, so an organ that ADDED g would silently push
every sheet on the shelf faster than its own sensitometry.
"""

import numpy as np

# Gelatin, the emulsion the light scatters IN. Kodak's own words,
# in a Kodak antihalation patent contemporaneous with nitrate
# stock: "Practically all ordinary gelatin photographic emulsions
# have refractive indices equal to that of the gelatin alone
# (1.541)" (Nadeau, US 2,481,770, Eastman Kodak, 1949; N19). The
# same patent adds that the dispersed silver salts do not shift
# it, so one index serves the whole pack. No wavelength or
# temperature is stated anywhere in it - D-line at room
# temperature is the near-certain intent for 1949 Kodak optical
# work, but the document does not say so, and that silence rides
# here rather than being quietly resolved.
EMULSION_INDEX = 1.541

# Support indices. Sourced or absent; a support whose index the
# record does not settle does not get a guess.
INDICES = {
    # Kodak, same patent (N19): "the index of refraction is between
    # 1.480 and 1.505" for the cellulose esters, "regardless of
    # which ester is chosen or the kind and quantity of
    # plasticizer". A RANGE, not a value - the midpoint rides here
    # and the bracket rides in this comment, which is the whole
    # reason the number is written to three places and not four.
    "acetate": 1.4925,
    # Soda-lime, measured: Kamptner 2024 tabulates 1.52225 (air
    # face) and 1.52503 (tin face) at 588.25 nm; Rubin 1985's
    # published fit evaluates to 1.5233 at 589.3 nm; Schott's own
    # B 270 soda-lime datasheet prints n_D 1.5229. Three routes
    # agreeing inside 0.003. BUT the 1890 plates are BLUE-EYED
    # (organ 4), and Cornu's fourth result is that the halo's
    # inner edge is bordered blue because n rises to the blue and
    # the ring tightens with it - so a blue-recording plate gets
    # the blue index, not the sodium one. Rubin's fit at 486 nm.
    "plate-glass-blue": 1.5290,
    "plate-glass": 1.5233,
}

# ESTAR/PET is deliberately NOT here. Photographic polyester is
# biaxially oriented and therefore BIREFRINGENT - 3M's own figures
# are 1.65 in-plane against 1.50 through-thickness (N22) - so a ray
# crossing the base at 40 degrees does not see one index, and no
# single number would be honest. Note the trap the same entry
# names: refractiveindex.info's PET row (1.569) is the UNORIENTED
# value and understates the in-plane index by ~0.08. Every
# ESTAR-based format on the shelf refuses on this ground.
_BIREFRINGENT = ("estar", "pet", "polyester")


# (support_um, index, ah_density, what it is) per stock: the
# thickness of the support the light must cross, its refractive
# index, and the optical density of the antihalation measure IN
# ONE PASS - the model spends it twice, down and back. A stock
# absent from this table refuses, by name, with its own reason in
# REFUSALS; the geometry is so willing that any thickness at all
# yields a plausible halo, which is exactly why none is invented.
SUPPORTS = {
    # The 1890 dry plates, and the only stocks on the shelf whose
    # antihalation state is POSITIVELY sourced rather than merely
    # unstated: they had none. Backing in the 1890s was a darkroom
    # operation the photographer did himself (Wall 1897 gives only
    # recipes and accepts unbacked working), anti-halation
    # substrata "have failed to answer in practice" as of 1897
    # (Oakley), ready-backed plates are still "A REVOLUTION IN
    # PHOTOGRAPHY" in 1900, Ilford backs "to order" in 1903, and
    # only by 1912 is it "almost all commercial plates" (Wall).
    # So ah_density 0.0 here is a sourced zero, not a default.
    #
    # Thickness is a BRACKET, like these plates' grain: 0.8-3 mm
    # for gelatin dry plates (Lavedrine/Getty via AIC PMG),
    # narrowed to "less than 2 mm" for factory-cut (Illinois
    # PSAP), so 1-2 mm is the defensible range and 1.3 mm its
    # middle. No period source prints a figure at all - the trade
    # sold glass thickness as a grade name ("extra thin glass to
    # order") - and ISO 14548's table is paywalled. H&D's own 1890
    # paper says nothing about the glass either, which is also why
    # their curves cannot be double-counting a halo they never
    # imaged: they exposed by contact to a controlled source.
    "manchester": (1300.0, INDICES["plate-glass-blue"], 0.0,
                   "gelatin dry plate on unbacked soda-lime glass, "
                   "1.3 mm (bracket 1-2 mm)"),
    "hd22": (1300.0, INDICES["plate-glass-blue"], 0.0,
             "gelatin dry plate on unbacked soda-lime glass, "
             "1.3 mm (bracket 1-2 mm)"),
}

# Why each other stock does not halate. Refusals are per stock and
# per reason, because the reasons genuinely differ - and a reader
# deserves to know whether a sheet is silent, a material is
# unmodellable, or the physics itself says no.
REFUSALS = {
    "collodion":
        "the wet plate refuses BY MECHANISM, not for want of a "
        "number: Abney ranks gelatin as halating WORSE than "
        "collodion despite collodion's glass being two to four "
        "times thicker (3.175-6.35 mm), because collodion's "
        "spread is scatter inside the film - 'the particles of "
        "liquid lying in the film' - not reflection off the back. "
        "Substituting the bigger thickness would move it the "
        "wrong way with confidence (halation-plates R4, R22)",
    "trix":
        "TRI-X's antihalation measure is a dyed GREY BASE and no "
        "sheet of any era states its density; the geometry is in "
        "hand (5-mil grey acetate in 135, 3.9-mil in 120) but "
        "10^(-2D) has no D (halation-supports Q3, Q4)",
    "tmax100":
        "no Kodak T-MAX document of any era contains the word "
        "halation - the undercoat everyone repeats is folklore "
        "with no primary support, and Kodak demonstrably says it "
        "when it applies (7266, 'an additional antihalation "
        "undercoat'). Silence, not zero (halation-supports Q10)",
    "tmax400":
        "as T-MAX 100: the whole T-MAX literature is silent on "
        "halation, in four publications across seventeen years "
        "(halation-supports Q10)",
    "p3200":
        "the most complete silence on the shelf - F-4001 states "
        "no base, no thickness and no antihalation measure, and "
        "the archived F-32 tables never covered P3200 either "
        "(halation-supports Q9)",
    "plusx":
        "Plus-X's grey base has no published density, exactly as "
        "TRI-X's has none; its thicknesses are known (5-mil in "
        "135, 3.6-mil in 120, the thinnest on the shelf) and its "
        "attenuation is not (halation-supports Q11)",
    "fp4":
        "Harman states an antihalation BACKING for rollfilm and "
        "sheet and none at all for 35mm, and gives a density for "
        "neither; the sheet format is polyester besides, which "
        "refuses on birefringence (halation-supports Q13-Q15)",
    "hp5":
        "as FP4 Plus - the same three supports, the same stated "
        "backing without a density, the same systematic silence "
        "for 35mm (halation-supports Q13-Q15)",
    "5222":
        "Double-X names a grey acetate safety base and NO "
        "thickness, in two editions seven years apart; no Kodak "
        "motion-picture sheet dimensions its support at all "
        "(halation-supports Q17, Q22)",
}

# THE ONE NUMBER NOBODY PUBLISHES. Lane N searched archive.org
# full text, Google Patents across the Kodak/Agfa/Fuji/3M/Konica
# antihalation families, Kodak's H-1 and H-24 series, SMPTE,
# JOSA/JOSA A/Applied Optics, IS&T, MDPI, PMC, HathiTrust and
# Gallica, and found NO measured radial density profile across a
# halation ring in real photographic film, NO published ratio of
# halation exposure to direct exposure, and NO measured fraction
# of incident light returning to a real emulsion (N28). So the
# gain is refused as a constant and exposed as the operator's,
# measured against the sourced ceiling below - which is the lane's
# own recommendation, not a convenience.
SCATTER = None


def fresnel_pair(theta_in, index_in, index_out):
    """Unpolarised reflectance at an interface, for light arriving
    at theta_in from inside index_in. Unity at and beyond the
    critical angle, which it reaches CONTINUOUSLY."""
    theta_in = np.asarray(theta_in, np.float64)
    s = (index_in / index_out) * np.sin(theta_in)
    tir = s >= 1.0
    out_angle = np.arcsin(np.where(tir, 0.0, s))
    ci, co = np.cos(theta_in), np.cos(out_angle)
    rs = ((index_in * ci - index_out * co)
          / (index_in * ci + index_out * co)) ** 2
    rp = ((index_in * co - index_out * ci)
          / (index_in * co + index_out * ci)) ** 2
    return np.where(tir, 1.0, 0.5 * (rs + rp))


def fresnel(theta, index):
    """Reflectance at the support/AIR boundary at the back."""
    return fresnel_pair(theta, index, 1.0)


def critical_radius(support_um, index):
    """r_c = 2 t tan(theta_c): where the spread peaks."""
    return 2.0 * float(support_um) * np.tan(np.arcsin(1.0 / index))


def reach_radius(support_um, index, emulsion_index=EMULSION_INDEX):
    """The largest radius the halo can reach at all. Infinite when
    the support is optically THINNER than the emulsion (grazing
    rays exist in the support); finite when it is DENSER, because
    then refraction at the emulsion/support interface compresses
    every ray inward and the halo has a hard outer edge. ESTAR
    (1.64) under gelatin (1.54) is the second case."""
    if index <= emulsion_index:
        return np.inf
    return 2.0 * float(support_um) * np.tan(
        np.arcsin(emulsion_index / index))


def spread(r_um, support_um, index, emulsion_index=EMULSION_INDEX):
    """p(r), the returned light's surface density (per um^2).

    The light is Lambertian in the EMULSION, not in the support -
    it is the emulsion's turbidity that scatters it - so it must
    refract across the emulsion/support interface on the way in.
    Carrying that through Snell leaves the SHAPE untouched and
    scales it by (n_support/n_emulsion)^2, which is exactly the
    share of the hemisphere that gets into the support at all:
    the rest is turned back at that interface and never crosses.
    """
    t = float(support_um)
    th2 = np.arctan2(np.asarray(r_um, np.float64), 2.0 * t)
    ratio = float(index) / float(emulsion_index)
    sin1 = ratio * np.sin(th2)
    reachable = sin1 <= 1.0
    th1 = np.arcsin(np.clip(sin1, 0.0, 1.0))
    into = 1.0 - fresnel_pair(th1, emulsion_index, index)
    return np.where(reachable,
                    ratio ** 2 * into * fresnel(th2, index)
                    * np.cos(th2) ** 4 / (4.0 * np.pi * t * t),
                    0.0)


def reflected_fraction(index, emulsion_index=EMULSION_INDEX,
                       quad=200001):
    """The share of emulsion-scattered light that comes back up.
    About 0.53 for acetate, 0.59 for plate glass, 0.61 for ESTAR -
    over half, either way, which is what a century of antihalation
    chemistry was for."""
    th2max = np.arcsin(min(1.0, emulsion_index / index))
    th2 = np.linspace(0.0, th2max, quad)
    ratio = float(index) / float(emulsion_index)
    th1 = np.arcsin(np.clip(ratio * np.sin(th2), 0.0, 1.0))
    into = 1.0 - fresnel_pair(th1, emulsion_index, index)
    return float(ratio ** 2 * np.trapezoid(
        into * fresnel(th2, index) * np.sin(2.0 * th2), th2))


def _j0(x):
    """Bessel J0 by Abramowitz & Stegun 9.4.1 / 9.4.3 - the
    standard rational approximations, |error| < 5e-8 and < 1.6e-8
    respectively. Kept local so the package stays pure numpy."""
    x = np.abs(np.asarray(x, np.float64))
    small = x < 3.0
    y = np.where(small, (x / 3.0) ** 2, 0.0)
    near = ((((((0.00021 * y - 0.0039444) * y + 0.0444479) * y
               - 0.3163866) * y + 1.2656208) * y - 2.2499997) * y + 1.0)
    z = np.where(small, 1.0, 3.0 / np.where(small, 1.0, x))
    f0 = (((((0.00014476 * z - 0.00072805) * z + 0.00137237) * z
            - 0.00009512) * z - 0.0055274) * z - 0.00000077) * z + 0.79788456
    t0 = (((((0.00013558 * z - 0.00029333) * z - 0.00054125) * z
            + 0.00262573) * z - 0.00003954) * z - 0.04166397) * z
    far = f0 * np.cos(x - 0.78539816 + t0) / np.sqrt(np.where(small, 1.0, x))
    return np.where(small, near, far)


# K depends only on the DIMENSIONLESS product nu = f * t (the
# spread scales with the support thickness and nothing else), so
# one table per index serves every stock and every pitch.
_K_CACHE = {}
_NU_MAX = 12.0


def _k_table(index, emulsion_index=EMULSION_INDEX,
             n_nu=512, quad=30001, u_max=400.0):
    # Grid chosen by measured convergence, not by taste: against a
    # 768 x 60001 / u<600 reference (and against an independent
    # scipy J0 run) this settles K to 6.9e-5 worst case over
    # 0.25-10 c/mm, for a fifth of the cost. Halving it again
    # still holds 2.9e-4, so there is margin either way.
    key = (round(float(index), 6), round(float(emulsion_index), 6))
    if key not in _K_CACHE:
        # r = t*u, so the transform is over u alone. Sample u on a
        # square grid: dense through the peak at u = r_c/t, sparse
        # out along the 16/u^3 tail.
        u = np.linspace(0.0, np.sqrt(u_max), quad) ** 2
        w = spread(u, 1.0, index, emulsion_index) * 2.0 * np.pi * u
        norm = np.trapezoid(w, u)
        # nu cubed-spaced: K falls from 1 to near zero inside
        # nu < 0.2, so a uniform grid would miss the whole event.
        nu = _NU_MAX * np.linspace(0.0, 1.0, n_nu) ** 3
        k = np.array([np.trapezoid(w * _j0(2.0 * np.pi * v * u), u) / norm
                      for v in nu])
        _K_CACHE[key] = (nu, k)
    return _K_CACHE[key]


def kernel_transfer(f_cycles_per_mm, support_um, index,
                    emulsion_index=EMULSION_INDEX):
    """K(f): the transform of the unit-normalised spread. K(0)=1."""
    nu_grid, k_grid = _k_table(index, emulsion_index)
    nu = np.abs(np.asarray(f_cycles_per_mm, np.float64)) / 1000.0 * float(support_um)
    return np.interp(nu, nu_grid, k_grid, left=1.0, right=0.0)


def transfer(f_cycles_per_mm, support_um, index, strength,
             emulsion_index=EMULSION_INDEX):
    """The halation transfer (1 - g) + g K(f). Unity at DC."""
    g = float(strength)
    return (1.0 - g) + g * kernel_transfer(
        f_cycles_per_mm, support_um, index, emulsion_index)


def ceiling(index, ah_density, emulsion_index=EMULSION_INDEX):
    """The SOURCED upper bound on g: every photon the emulsion
    scatters downward does so Lambertianly, crosses the support
    without absorption, and returns. Reached only if nothing is
    lost on the way, which is why it is a ceiling and not a value.

    The Lambertian premise is not this repo's: Cornu treats the
    lit point as "une veritable source lumineuse rayonnant dans
    tous les sens" and demonstrated the ring forms identically
    through a lens, a compound objective, a concave mirror and a
    pierced card (N4); Mees says the same in English (N9); and
    Vendrovsky & Pakoushko's 1964 equations are derived "on the
    presumption of complete light diffusion in photographic
    layers" and their "experimental trials show the equations are
    valid" - the one photographic paper on this, known only by its
    abstract because it is paywalled, and the standing acquisition
    target for this organ.

    The antihalation measure is spent TWICE, down and back. The
    only source stating that arithmetic is BBC T-075, on the CRT
    analogue: the returned light "is reduced in intensity by a
    factor always greater than I squared" (N13). The photographic
    literature describes the double pass and never algebraises it,
    so 10^(-2D) is cited to a television report, deliberately.
    """
    return (reflected_fraction(index, emulsion_index)
            * 10.0 ** (-2.0 * float(ah_density)))


def for_stock(name, gain):
    """(support_um, index, g) for a named stock at the operator's
    `gain` - a fraction of the sourced ceiling, because the gain
    itself is the one quantity nobody has published (SCATTER).
    Raises for a stock the record cannot support, with that
    stock's own reason."""
    spec = SUPPORTS.get(name)
    if spec is None:
        raise ValueError(
            f"{name} does not halate: " + REFUSALS.get(
                name, "no lane has sourced this stock's support "
                      "(organ 10; see docs/sources/dossiers/"
                      "halation-supports.md)"))
    g = float(gain)
    if not 0.0 <= g <= 1.0:
        raise ValueError(
            "halation= is a fraction of the SOURCED CEILING - the "
            "share of emulsion-scattered light that comes back if "
            "nothing is absorbed on the way - so it lies in "
            f"[0, 1]; got {g}. Above 1 would return more light "
            "than the support reflects")
    support_um, index, ah_density, _what = spec
    return (support_um, index, g * ceiling(index, ah_density))


def apply(img, *, pitch_um, support_um, index, strength,
          emulsion_index=EMULSION_INDEX):
    """Redistribute a 2-D linear-exposure image by the support's
    halation at the given pitch - exact, in frequency space,
    reflect-padded. A support far thinner than one pixel cannot
    displace light between pixels and returns the image untouched
    (a true no-op, bit-identical)."""
    g = float(strength)
    pitch = float(pitch_um)
    if g <= 0.0 or critical_radius(support_um, index) / pitch < 0.05:
        return img
    x = np.asarray(img, np.float64)
    if x.ndim != 2:
        # padding a (..., 3) field would pad the CHANNEL axis too and
        # quietly mix the records. Callers project first, or loop.
        raise ValueError(
            "halation applies to a single 2-D exposure plane; got "
            f"shape {x.shape}")
    # the halo holds 98% inside 10 r_c; pad to that, clamped to the
    # frame (a halo wider than the sheet has nothing outside it to
    # come from, and reflect is the least-bad statement of that).
    reach = 10.0 * critical_radius(support_um, index) / pitch
    pad = int(min(max(reach, 4.0), min(x.shape) - 1))
    xp = np.pad(x, pad, mode="reflect")
    fy = np.fft.fftfreq(xp.shape[0], d=pitch / 1000.0)
    fx = np.fft.rfftfreq(xp.shape[1], d=pitch / 1000.0)
    fr = np.sqrt(fy[:, None] ** 2 + fx[None, :] ** 2)
    out = np.fft.irfft2(
        np.fft.rfft2(xp) * transfer(fr, support_um, index, g,
                                    emulsion_index), s=xp.shape)
    return np.maximum(out[pad:pad + x.shape[0], pad:pad + x.shape[1]], 0.0)
