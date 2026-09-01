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

THE GEOMETRY IS DERIVED, NOT DIALLED. With the emulsion scattering
into the support as a Lambertian source, the flux into angle theta
is sin(2 theta) d theta, the return lands at r = 2 t tan(theta),
and the surface density follows in three lines:

    p(r) = R(theta) cos^4(theta) / (4 pi t^2),  theta = atan(r/2t)

- the cos^4 law arriving from a direction it was not expected
from. R is the Fresnel reflectance from inside the support. So
the spread is a weak disc (a few percent, at normal incidence)
climbing to a peak at

    r_c = 2 t tan(asin(1/n))

and decaying outward. NOT a discontinuity: Fresnel climbs to unity
CONTINUOUSLY as theta approaches the critical angle. But not a
smooth maximum either - 1 - R goes as sqrt(theta_c - theta), so
the profile arrives at its peak (7.9x the centre value at
n = 1.48) with an INFINITE derivative. A cusp, which is what a
sharply defined ring edge looks like written down. The halo is
much broader than r_c alone suggests - only 6.9% of the returned
light falls inside r_c, 60% inside 2 r_c, 98% inside 10 r_c -
which is why period interiors veil rather than merely ring.

There is no free shape parameter anywhere above: thickness and
index are datasheet facts and the profile follows. The one number
that must be SOURCED is the strength - what share of the exposing
light enters the support as wide-angle scatter, and what the
antihalation measure removes from it on the double pass down and
back (a dyed undercoat of density D attenuates by 10^(-2D)).
Stocks whose sheets are silent on their support refuse by name.

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

# (support_um, index, ah_density) per stock: thickness of the
# support the light must cross, its refractive index, and the
# optical density of the antihalation measure IN ONE PASS (the
# model applies it twice - down and back). 0.0 means no measure
# at all; a stock absent from this table refuses.
#
# EMPTY UNTIL THE LANES LAND. Constants are sourced or refused,
# and an invented base thickness would be the most inviting lie
# in the organ - the geometry is so willing that any number at
# all produces a plausible halo.
SUPPORTS = {}

# The share of exposing light that leaves the emulsion downward as
# wide-angle scatter. Sourced or refused; see the dossier.
SCATTER = None


def fresnel(theta, index):
    """Unpolarised reflectance at the support/air boundary, for
    light arriving from INSIDE a support of the given index.
    Unity at and beyond the critical angle."""
    theta = np.asarray(theta, np.float64)
    s = index * np.sin(theta)
    tir = s >= 1.0
    out_angle = np.arcsin(np.where(tir, 0.0, s))
    ct, co = np.cos(theta), np.cos(out_angle)
    rs = ((index * ct - co) / (index * ct + co)) ** 2
    rp = ((index * co - ct) / (index * co + ct)) ** 2
    return np.where(tir, 1.0, 0.5 * (rs + rp))


def critical_radius(support_um, index):
    """r_c = 2 t tan(theta_c): where the spread peaks."""
    return 2.0 * float(support_um) * np.tan(np.arcsin(1.0 / index))


def spread(r_um, support_um, index):
    """p(r), the returned light's surface density (per um^2)."""
    t = float(support_um)
    theta = np.arctan2(np.asarray(r_um, np.float64), 2.0 * t)
    return fresnel(theta, index) * np.cos(theta) ** 4 / (4.0 * np.pi * t * t)


def reflected_fraction(index, quad=200001):
    """The share of downward-scattered light that comes back up:
    the angular integral of R over the Lambertian hemisphere.
    0.584 for acetate, 0.608 for plate glass, 0.670 for ESTAR -
    high enough to explain a century of antihalation chemistry."""
    th = np.linspace(0.0, 0.5 * np.pi, quad)
    return float(np.trapezoid(fresnel(th, index) * np.sin(2.0 * th), th))


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


def _k_table(index, n_nu=512, quad=30001, u_max=400.0):
    # Grid chosen by measured convergence, not by taste: against a
    # 768 x 60001 / u<600 reference (and against an independent
    # scipy J0 run) this settles K to 6.9e-5 worst case over
    # 0.25-10 c/mm, for a fifth of the cost. Halving it again
    # still holds 2.9e-4, so there is margin either way.
    key = round(float(index), 6)
    if key not in _K_CACHE:
        # r = t*u, so the transform is over u alone. Sample u on a
        # square grid: dense through the peak at u = r_c/t, sparse
        # out along the 16/u^3 tail.
        u = np.linspace(0.0, np.sqrt(u_max), quad) ** 2
        w = spread(u, 1.0, index) * 2.0 * np.pi * u
        norm = np.trapezoid(w, u)
        # nu cubed-spaced: K falls from 1 to near zero inside
        # nu < 0.2, so a uniform grid would miss the whole event.
        nu = _NU_MAX * np.linspace(0.0, 1.0, n_nu) ** 3
        k = np.array([np.trapezoid(w * _j0(2.0 * np.pi * v * u), u) / norm
                      for v in nu])
        _K_CACHE[key] = (nu, k)
    return _K_CACHE[key]


def kernel_transfer(f_cycles_per_mm, support_um, index):
    """K(f): the transform of the unit-normalised spread. K(0)=1."""
    nu_grid, k_grid = _k_table(index)
    nu = np.abs(np.asarray(f_cycles_per_mm, np.float64)) / 1000.0 * float(support_um)
    return np.interp(nu, nu_grid, k_grid, left=1.0, right=0.0)


def transfer(f_cycles_per_mm, support_um, index, strength):
    """The halation transfer (1 - g) + g K(f). Unity at DC."""
    g = float(strength)
    return (1.0 - g) + g * kernel_transfer(f_cycles_per_mm, support_um, index)


def strength(support_um, index, ah_density, scatter):
    """g: the share of exposing light that ends up redistributed -
    scattered down, reflected back, and surviving the antihalation
    measure on BOTH passes."""
    return (float(scatter) * reflected_fraction(index)
            * 10.0 ** (-2.0 * float(ah_density)))


def for_stock(name):
    """(support_um, index, g) for a named stock, or None if its
    sheet is silent on its base - in which case the stock does not
    halate, rather than halating by invention."""
    spec = SUPPORTS.get(name)
    if spec is None or SCATTER is None:
        return None
    support_um, index, ah_density = spec
    return (support_um, index,
            strength(support_um, index, ah_density, SCATTER))


def apply(img, *, pitch_um, support_um, index, strength):
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
        np.fft.rfft2(xp) * transfer(fr, support_um, index, g), s=xp.shape)
    return np.maximum(out[pad:pad + x.shape[0], pad:pad + x.shape[1]], 0.0)
