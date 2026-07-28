"""The darkroom proper: turn a linear float32 negative into a print file.

Development never touches the renderer — re-develop a negative as many
times as you like. The tone pipeline mirrors the atlas (so the screen
look is reproducible) but everything is a dial here.
"""
import numpy as np
import tifffile


def auto_exposure(neg, percentile=99.5, target=0.85):
    """Choose E so that the given luminance percentile maps to `target`
    after the filmic curve 1 - exp(-x*E)."""
    lum = 0.299 * neg[..., 0] + 0.587 * neg[..., 1] + 0.114 * neg[..., 2]
    ref = np.percentile(lum[lum > 0], percentile) if (lum > 0).any() else 1.0
    if ref <= 0:
        return 1.0
    return -np.log(max(1.0 - target, 1e-6)) / ref


TONES = {
    "gray":      ["#000000", "#ffffff"],
    "selenium":  ["#000000", "#2a2733", "#6e6a78", "#c9c6ce", "#ffffff"],
    "platinum":  ["#000000", "#3a332c", "#8a7d6a", "#d6ccbc", "#fffdf6"],
    "sepia":     ["#000000", "#3c2a1a", "#8a5c34", "#d8b184", "#fff3e0"],
    "cyanotype": ["#00060e", "#0c2f52", "#2f6b9e", "#8fb9d8", "#f2f8ff"],
    "gold":      ["#000000", "#2b2013", "#7a5a28", "#d4a94e", "#fff3d6"],
    "copper":    ["#000000", "#33180f", "#8a4a2c", "#d99a6b", "#ffeadb"],
    "split":     ["#000000", "#1d2735", "#5a6273", "#c9b394", "#fff8ea"],
    "ember":     ["#000000", "#3d0d06", "#a03511", "#e8933c", "#fff1c4"],
    "moonlight": ["#00030a", "#141d33", "#3d517a", "#93a9c9", "#eef3ff"],
}


def _hex(c):
    c = c.lstrip("#")
    return np.array([int(c[i:i + 2], 16) / 255.0 for i in (0, 2, 4)], np.float32)


def _boxn(img, r, n=3):
    """n box passes per axis via cumsum; three approximates a gaussian."""
    out = img
    for _ in range(n):
        for axis in (0, 1):
            n = out.shape[axis]
            cs = np.cumsum(out, axis=axis, dtype=np.float32)
            cs = np.concatenate([np.zeros_like(np.take(cs, [0], axis=axis)), cs],
                                axis=axis)
            hi = np.minimum(np.arange(n) + r + 1, n)
            lo = np.maximum(np.arange(n) - r, 0)
            out = (np.take(cs, hi, axis=axis) - np.take(cs, lo, axis=axis))                 / (hi - lo).reshape([-1 if a == axis else 1
                                     for a in range(out.ndim)])
    return out


def _box3(img, r):
    return _boxn(img, r, 3)


def _boxn_frac(img, r, n=3):
    """Box passes with a REAL radius, by weighting the two edge samples.

    `_boxn` rounds its radius down to a whole pixel, which is invisible at
    a radius of twenty and fatal at a radius of two. The annulus below is
    the difference of two blurs; when both round to the same box the
    difference is not approximate but exactly zero, and halation silently
    disappears at small print sizes. Measured before this existed: a
    140-micron ring vanished entirely on 900px and 1800px sheets and
    appeared only at 3600.

    `_boxn` is deliberately left alone. Bloom, sharpening and the backdrop
    all run through it, and changing their radii by a fraction of a pixel
    would alter every grade ever approved.
    """
    r = float(max(r, 0.0))
    if r < 1e-3:
        return img
    r0 = int(r)
    frac = r - r0
    if frac < 1e-6:
        return _boxn(img, r0, n)
    out = img
    for _ in range(n):
        for axis in (0, 1):
            m = out.shape[axis]
            idx = np.arange(m)
            cs = np.cumsum(out, axis=axis, dtype=np.float32)
            cs = np.concatenate([np.zeros_like(np.take(cs, [0], axis=axis)), cs],
                                axis=axis)
            hi = np.minimum(idx + r0 + 1, m)
            lo = np.maximum(idx - r0, 0)
            acc = np.take(cs, hi, axis=axis) - np.take(cs, lo, axis=axis)
            cnt = (hi - lo).astype(np.float32)
            # the partial pixel just outside the whole-pixel window, on
            # each side, counted only where it is really there
            hi_i, lo_i = idx + r0 + 1, idx - r0 - 1
            wh = (hi_i <= m - 1).astype(np.float32)
            wl = (lo_i >= 0).astype(np.float32)
            shape = [-1 if a == axis else 1 for a in range(out.ndim)]
            acc = acc + frac * (
                np.take(out, np.clip(hi_i, 0, m - 1), axis=axis)
                * wh.reshape(shape)
                + np.take(out, np.clip(lo_i, 0, m - 1), axis=axis)
                * wl.reshape(shape))
            cnt = cnt + frac * (wh + wl)
            out = acc / cnt.reshape(shape)
    return out


def _blur(img, sigma_px):
    """Blur, computed at whatever resolution the result can actually hold.

    A blur destroys exactly the detail that makes it expensive, so running
    it at full size is wasted work: for a wide radius the answer is
    downsampled, blurred small, and resampled back. At print size this is
    the difference between five seconds and a fraction of one, and the
    output is indistinguishable because the discarded frequencies were
    about to be discarded anyway.
    """
    h, w = img.shape[:2]
    # keep ~4 samples per sigma after shrinking; below that there is
    # nothing to gain and blockiness starts to show through
    f = max(1, min(int(sigma_px / 4.0), h // 32, w // 32))
    if f <= 1:
        return _box3(img, max(1, int(sigma_px * 0.55)))

    ph, pw = (-h) % f, (-w) % f
    if ph or pw:
        img = np.pad(img, ((0, ph), (0, pw), (0, 0)), mode="edge")
    H, W = img.shape[0] // f, img.shape[1] // f
    small = img.reshape(H, f, W, f, img.shape[2]).mean(axis=(1, 3))

    small = _box3(small, max(1, int((sigma_px / f) * 0.55)))

    big = np.repeat(np.repeat(small, f, axis=0), f, axis=1)
    # one pass, not three: the small image is already smooth, so this only
    # has to hide the resample steps. Three passes here cost more than the
    # full-resolution blur they were meant to replace.
    big = _boxn(big, max(1, f // 2), n=1)
    return big[:h, :w]


def _blur_full(img, sigma_px):
    """The original full-resolution path, kept for reference."""
    r = max(1, int(sigma_px * 0.55))
    out = img
    for _ in range(3):
        for axis in (0, 1):
            n = out.shape[axis]
            cs = np.cumsum(out, axis=axis, dtype=np.float32)
            cs = np.concatenate([np.zeros_like(np.take(cs, [0], axis=axis)), cs],
                                axis=axis)
            hi = np.minimum(np.arange(n) + r + 1, n)
            lo = np.maximum(np.arange(n) - r, 0)
            out = (np.take(cs, hi, axis=axis) - np.take(cs, lo, axis=axis)) \
                / (hi - lo).reshape([-1 if a == axis else 1
                                     for a in range(out.ndim)])
    return out


# ---------------------------------------------------------------------
# What happens to the light before the film is developed.
#
# Order matters and is physical: the aerial image is formed by the lens
# (diffraction), then light enters the emulsion and some of it bounces off
# the film base and re-exposes from behind (halation). Both act on the
# latent image, so both belong to the negative rather than the print.
#
# Sizes are given on the SHEET, in microns, and converted to pixels using
# the sheet's own dimensions. A given lens therefore behaves identically
# whether the file is printed at twelve inches or forty-eight: the same
# physical blur simply lands on a different number of pixels.

SENSOR_H_UM = 96_000.0                 # the 4x5 sheet, short dimension
SENSOR_W_UM = 120_000.0
LAMBDA_UM = (0.610, 0.545, 0.465)      # where each channel lives, roughly


def _pitch_um(width, height):
    """Microns of sheet per pixel of file."""
    sensor = SENSOR_H_UM if width >= height else SENSOR_W_UM
    return sensor / max(min(width, height), 1)


def _box_variance(r, n=3):
    """Variance in px^2 of n box passes of real half-width r."""
    r0 = int(r)
    f = r - r0
    w = (2 * r0 + 1) + 2 * f
    v1 = (r0 * (r0 + 1) * (2 * r0 + 1) / 3.0 + 2 * f * (r0 + 1) ** 2) / w
    return n * v1


def _radius_for_sigma(sigma_px, n=3, tol=1e-6):
    """The box half-width whose n passes have the wanted standard deviation.

    Needed because a FRACTIONAL box is not a narrow kernel: its side taps
    sit at plus and minus one whole pixel no matter how small the radius,
    only their weight shrinks. So the usual `radius = 0.55 * sigma` rule,
    which is calibrated for whole-pixel boxes, badly over-blurs below a
    pixel - measured at f/64 on a 900px sheet, it widened an edge by 89
    microns where the aperture calls for 43. Solving for the variance
    directly is exact at every scale and costs one scalar bisection.
    """
    want = sigma_px * sigma_px
    if want <= 0:
        return 0.0
    lo, hi = 0.0, 1.0
    while _box_variance(hi, n) < want:
        hi *= 2.0
        if hi > 1e6:
            break
    while hi - lo > tol:
        mid = 0.5 * (lo + hi)
        if _box_variance(mid, n) < want:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def blur_microns(img, sigma_um):
    """Gaussian-ish blur of a width measured on the SHEET, not on the file.

    This is the whole reason the optical stages reproduce across formats.
    A 140-micron halation radius is 140 microns whether the negative is
    read out at 3600 pixels or 14400; expressed in pixels it would be 5 or
    21, and the same two numbers would describe two different lenses.
    Anything with a physical size belongs here.
    """
    px = sigma_um / _pitch_um(img.shape[1], img.shape[0])
    if px < 0.03:
        return img
    if px >= 12.0:
        # wide enough that whole-pixel rounding is a rounding error, so
        # take the multiresolution path and its speed. `_blur` works in
        # its own units, where sigma comes out about 0.55x the argument.
        return _blur(img, px / 0.55)
    return _boxn_frac(img, _radius_for_sigma(px), n=3)


def diffraction_blur(neg, fstop, *, amount=1.0):
    """Soften by the diffraction limit of the aperture actually used.

    The Airy disk's first zero sits at 1.22*lambda*N. At f/64 and green
    light that is 43 microns, which on a 4x5 sheet read out at 7200 pixels
    is about three pixels - small, real, and exactly the reason large
    format stops down only so far. Each channel gets its own wavelength,
    so the softening carries a faint chromatic edge for free.

    This is the instrument constrained by its own optics: Plate XLIX
    computes these patterns from first principles.
    """
    if not fstop or amount <= 0:
        return neg
    out = np.empty_like(neg)
    for ch in range(3):
        # 1.22*lambda*N is the first zero; the gaussian that best fits an
        # Airy disk has sigma about 0.42 of it, which is the number to
        # blur by
        sigma_um = 0.42 * 1.22 * LAMBDA_UM[ch] * float(fstop) * amount
        out[..., ch] = blur_microns(neg[..., ch:ch+1], sigma_um)[..., 0]
    return out


def halation(neg, *, strength=0.35, radius_um=140.0, inner=0.45):
    """The ring, not the glow.

    Light that reaches the emulsion partly passes through it, reflects off
    the film base and comes back to expose from behind - displaced by
    twice the base thickness. What that leaves is an offset ANNULUS around
    a highlight, not a symmetric haze, and it is strongest in red because
    red penetrates deepest. That is why halation reads as a red-orange
    corona rather than a soft bloom. (`bloom` remains available for the
    symmetric kind; this is the one film actually does.)
    """
    if strength <= 0:
        return neg
    per_channel = (1.0, 0.42, 0.16)          # red goes deepest
    ring = (blur_microns(neg, radius_um)
            - blur_microns(neg, radius_um * inner))
    ring = np.clip(ring, 0.0, None)
    return neg + ring * (strength * np.array(per_channel, np.float32))


def _field(shape, kind, angle, center):
    """The scalar 0..1 that drives a backdrop gradient."""
    h, w = shape[:2]
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    u, v = xx / w, yy / h
    cx, cy = center
    if kind == "linear":
        a = np.radians(angle)
        t = (u - cx) * np.cos(a) + (v - cy) * np.sin(a)
        t = np.clip(t * 1.4 + 0.5, 0.0, 1.0)
    elif kind == "floor":
        # A seamless: brightest at the horizon and falling away above and
        # below, so the subject stands in a space instead of floating in a void
        d = v - cy
        t = np.clip(1.0 - np.abs(d) * 2.0, 0.0, 1.0) ** 1.5
        t = np.where(d > 0, t * (1.0 - 0.5 * np.clip(d * 2.2, 0.0, 1.0)), t)
    else:  # radial
        t = 1.0 - np.clip(np.sqrt((u - cx) ** 2 + ((v - cy) * h / w) ** 2) * 1.9,
                          0.0, 1.0)
        t = t * t
    return t.astype(np.float32)


def _backdrop(shape, kind, stops, angle, center, strength):
    """A studio backdrop: any gradient of colour stops, at any angle,
    screened under the render by the caller."""
    t = _field(shape, kind, angle, center)
    xs = np.linspace(0.0, 1.0, len(stops))
    cols = np.stack([_hex(c) for c in stops])
    bg = np.stack([np.interp(t, xs, cols[:, ch]) for ch in range(3)], axis=-1)
    return (strength * bg).astype(np.float32)


def _grain(shape, amount, size, seed=7):
    """Paper grain: noise at a chosen scale, softened so it reads as fibre
    rather than as pixels."""
    rng = np.random.default_rng(seed)
    h, w = shape[:2]
    size = max(float(size), 1.0)
    gh, gw = max(1, int(h / size)), max(1, int(w / size))
    n = rng.random((gh, gw), dtype=np.float32)
    n = np.repeat(np.repeat(n, int(np.ceil(h / gh)), 0),
                  int(np.ceil(w / gw)), 1)[:h, :w]
    n = _blur(n[..., None], max(size * 0.5, 1.0))[..., 0]
    n -= n.mean()
    peak = max(float(np.abs(n).max()), 1e-6)
    return (n / peak * amount).astype(np.float32)


def apply_palette(c, stops, mix=1.0):
    """Gradient-map toning: map display luminance through color stops."""
    lum = 0.299 * c[..., 0] + 0.587 * c[..., 1] + 0.114 * c[..., 2]
    xs = np.linspace(0.0, 1.0, len(stops))
    cols = np.stack([_hex(s) for s in stops])
    mapped = np.stack([np.interp(lum, xs, cols[:, ch]) for ch in range(3)],
                      axis=-1).astype(np.float32)
    return c + (mapped - c) * mix


def unsharp(img, amount, radius_px):
    """Output sharpening: unsharp mask at print resolution. Apply last -
    prints need more sharpening than any screen preview suggests."""
    return np.clip(img + amount * (img - _blur(img, radius_px)), 0.0, 1.0)


# ---------------------------------------------------------------------
# Historic printing processes, derived rather than painted.
#
# A gradient map can imitate the colour of a cyanotype; it cannot imitate
# the reason for it. These build the image the way the process does: the
# accumulated measure is treated as exposing dose, the dose is converted
# to deposited substance through a response curve with a toe and a
# saturating shoulder, and that substance then absorbs light according to
# its own per-channel coefficients. The blue of a cyanotype comes out
# because Prussian blue eats red and spares blue, not because blue was
# chosen - the same way the 22-degree halo in Plate LI comes out of ice
# geometry rather than being drawn at 22 degrees.
#
# Each process is therefore monochrome, as the real ones are: a cyanotype
# made from a colour negative is still blue. Only `dose` carries over.
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
    # dose -> deposited substance. The exponential saturates on its own,
    # which is the shoulder; the toe exponent slows the start.
    d = (pr["dmax"] * dmax_mul) * (1.0 - np.exp(-dose)) ** (pr["toe"] / contrast)
    absorb = np.array(pr["absorb"], np.float32)
    if pigment or pr.get("pigment"):
        # a ground pigment absorbs the complement of its own colour
        absorb = 1.0 - _hex(pigment or pr["pigment"]) * 0.88
    t = np.power(10.0, -(d[..., None] * absorb))       # optical density
    return (_hex(pr["base"]) * t).astype(np.float32)


PAPERS = {
    "bright":  "#ffffff",
    "warm":    "#faf6ec",
    "cream":   "#f3ebd8",
    "rag":     "#efe9dd",
    "newsprint": "#e8e2d2",
}


def paper_print(neg, E, paper="#faf6ec", subtractive=True, ink=None):
    """Render the measure as ink on paper instead of light in darkness.

    The additive pipeline is exact for a luminous object photographed
    against black: brightness accumulates where sample points land. Paper
    is the opposite medium and wants the opposite law - Beer-Lambert
    absorption, where density subtracts from the light the sheet reflects:

        reflected = paper * exp(-density)

    That is the physics of dye on a page rather than an effect applied to
    look like one. Where filaments cross they grow darker, the way
    overlapping ink does in an etching, instead of burning toward white.

    With subtractive colour the plate's own hue becomes the ink's: light
    that was emitted red is absorbed as red ink, which takes out green and
    blue and leaves red. Without it the whole measure prints in one
    neutral ink, which is what a single-plate lithograph would give.
    """
    d = (0.299 * neg[..., 0] + 0.587 * neg[..., 1] + 0.114 * neg[..., 2])
    if subtractive:
        peak = np.maximum(neg.max(axis=-1, keepdims=True), 1e-9)
        hue = np.clip(neg / peak, 0.0, 1.0)          # what colour the light was
        absorb = d[..., None] * (1.0 - hue * 0.85)   # ink takes out the rest
    else:
        absorb = d[..., None] * np.ones(3, np.float32)
    if ink is not None:
        absorb = absorb * (1.0 - _hex(ink) * 0.85)
    t = np.exp(-absorb * E)                          # transmittance of the ink
    return t.astype(np.float32)


def shadow_field(t, *, strength=0.35, radius=0.02, offset=(0.004, 0.010),
                 bite=1.6):
    """A soft shadow cast onto the sheet by the printed form.

    It darkens the paper UNDER the ink rather than being blended against
    it - the first version composited the two by coverage, which for a
    diffuse subject meant nearly the whole frame turned into shadowed
    paper and the picture went grey. The physical order is: light falls on
    the sheet, some is blocked, what remains passes through the ink.

    `bite` raises the matte to a power before blurring, so only substantial
    ink casts anything. Without it, a haze of faint filaments across the
    whole frame throws a shadow the size of the frame."""
    matte = np.clip(1.0 - t.max(axis=-1), 0.0, 1.0) ** bite
    h, w = matte.shape[:2]
    sig = max(radius * min(h, w), 1.0)
    dy, dx = int(offset[1] * h), int(offset[0] * w)
    sh = np.roll(np.roll(matte, dy, axis=0), dx, axis=1)
    sh = _blur(sh[..., None], sig)[..., 0]
    return np.clip(sh * strength, 0.0, 0.92).astype(np.float32)


def tone_curve(c, shoulder=0.0, split_lo=None, split_hi=None, split=0.0):
    """A shoulder that rolls highlights off instead of clipping them, and
    an optional split tone - warm blacks under cool lights, the oldest
    trick in the printing room."""
    if shoulder > 0:
        c = np.clip(c * (1.0 + shoulder) / (1.0 + shoulder * c), 0.0, 1.0)
    if split > 0 and split_lo and split_hi:
        lum = (0.299 * c[..., 0] + 0.587 * c[..., 1]
               + 0.114 * c[..., 2])[..., None]
        tint = _hex(split_lo) + (_hex(split_hi) - _hex(split_lo)) * lum
        c = c * (1.0 - split) + c * tint * 2.0 * split
    return np.clip(c, 0.0, 1.0)


def develop(neg, *, exposure=None, ev=0.0, gamma=0.82, saturation=1.0,
            vignette=0.0, percentile=99.5, target=0.85,
            bloom=0.0, bloom_radius=0.015,
            bg_kind=None, bg_a="#0c0c12", bg_b="#1c1826", bg_stops=None,
            bg_angle=90.0, bg_center=(0.5, 0.45), bg_strength=1.0,
            grain=0.0, grain_size=2.0,
            palette=None, palette_mix=1.0,
            sharpen=0.0, sharpen_radius=1.2,
            paper=None, subtractive=True, ink=None, ink_density=1.0,
            process=None, proc_contrast=1.0, proc_dmax=1.0,
            fstop=0.0, diffraction=0.0,
            halation_strength=0.0, halation_radius=140.0,
            shadow=0.0, shadow_radius=0.02,
            shoulder=0.0, split=0.0,
            split_lo="#3a2c1e", split_hi="#dfe9f5"):
    """Negative to print. The order of stages is fixed and physical:

        diffraction -> halation -> [bloom] -> expose -> process/paper
            -> tone -> backdrop -> grain -> sharpen

    The first two act on the latent image, before there is a print to
    speak of: the lens forms the aerial image and the emulsion scatters
    what reaches it. Everything from `expose` onward is the darkroom.
    Bloom sits between them because it is not an emulsion effect at all -
    it is a glow, kept because it is useful, and `halation` is the one
    film actually does.

    Every optical stage is off at its default, and off is not a lie: a
    virtual camera with ideal optics is an honest instrument. Turning
    diffraction on trades that claim for a different true one - that the
    lens obeys the diffraction limit at the aperture it reports.
    """
    neg = np.nan_to_num(neg, nan=0.0, posinf=0.0, neginf=0.0)
    # the lens, then the emulsion - both before anything is developed
    if diffraction > 0 and fstop:
        neg = diffraction_blur(neg, fstop, amount=diffraction)
    if halation_strength > 0:
        neg = halation(neg, strength=halation_strength,
                       radius_um=halation_radius)
    E = (exposure if exposure is not None
         else auto_exposure(neg, percentile, target)) * (2.0 ** ev)
    if bloom > 0:
        sigma = bloom_radius * min(neg.shape[0], neg.shape[1])
        neg = neg + bloom * _blur(neg, sigma)     # halation in linear light
    if process:
        c = process_print(neg, E * ink_density, process, pigment=ink,
                          contrast=proc_contrast, dmax_mul=proc_dmax)
        if shadow > 0:
            base = _hex(PROCESSES[process]["base"])
            sh = shadow_field(np.clip(c / np.maximum(base, 1e-6), 0, 1),
                              strength=shadow, radius=shadow_radius)
            c = c * (1.0 - sh[..., None])
    elif paper:
        t = paper_print(neg, E * ink_density, paper=paper,
                        subtractive=subtractive, ink=ink)
        sheet = _hex(PAPERS.get(paper, paper)) * np.ones_like(t)
        if shadow > 0:
            sh = shadow_field(t, strength=shadow, radius=shadow_radius)
            sheet = sheet * (1.0 - sh[..., None])
        c = sheet * t                      # light on the sheet, then through the ink
    else:
        c = 1.0 - np.exp(-neg * E)
    lum = (0.299 * c[..., 0] + 0.587 * c[..., 1] + 0.114 * c[..., 2])[..., None]
    c = lum + (c - lum) * saturation
    if palette:
        stops = TONES.get(palette) if isinstance(palette, str) and \
            palette in TONES else palette
        if isinstance(stops, str):
            stops = [s.strip() for s in stops.split(",")]
        c = apply_palette(np.clip(c, 0.0, 1.0), stops, palette_mix)
    if bg_kind:
        stops = bg_stops or [bg_a, bg_b]
        if isinstance(stops, str):
            stops = [x.strip() for x in stops.split(",") if x.strip()]
        if len(stops) < 2:
            stops = [bg_a, bg_b]
        bg = _backdrop(c.shape, bg_kind, stops, bg_angle,
                       bg_center, bg_strength)
        c = bg + c - bg * c                        # screen: light on backdrop
    c = np.clip(c, 0.0, 1.0) ** (1.0 / gamma if (paper or process) else gamma)
    if vignette > 0:
        h, w = c.shape[:2]
        yy, xx = np.mgrid[0:h, 0:w]
        d2 = ((xx / w - 0.5) ** 2 + (yy / h - 0.5) ** 2)
        c *= (1.0 - vignette * 1.1 * d2)[..., None]
    if shoulder > 0 or split > 0:
        c = tone_curve(c, shoulder, split_lo, split_hi, split)
    c = np.clip(c, 0.0, 1.0).astype(np.float32)
    if sharpen > 0:
        c = unsharp(c, sharpen, sharpen_radius)
    if grain > 0:
        # strongest in the midtones, where film grain actually sits
        lum = 0.299 * c[..., 0] + 0.587 * c[..., 1] + 0.114 * c[..., 2]
        mask = (1.0 - np.abs(2.0 * lum - 1.0))[..., None]
        g = _grain(c.shape, grain, grain_size)[..., None]
        c = np.clip(c + g * mask, 0.0, 1.0)
    return c, E


def write_print(path, img01, dpi=300, extratags=None, description=None):
    """16-bit TIFF with resolution tags so editors report the print size."""
    out = (img01 * 65535.0 + 0.5).astype(np.uint16)
    tifffile.imwrite(path, out, photometric="rgb", compression="zlib",
                     resolution=(dpi, dpi), resolutionunit="INCH",
                     description=description,
                     extratags=extratags or [])


def write_preview(path, img01, max_side=2000):
    from PIL import Image
    im = Image.fromarray((img01 * 255.0 + 0.5).astype(np.uint8))
    im.thumbnail((max_side, max_side))
    im.save(path)
