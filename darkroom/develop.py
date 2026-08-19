"""The darkroom proper: turn a linear float32 negative into a print file.

Development never touches the renderer — re-develop a negative as many
times as you like. The tone pipeline mirrors the atlas (so the screen
look is reproducible) but everything is a dial here.
"""
import os
from concurrent.futures import ThreadPoolExecutor

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


# ---------------------------------------------------------------- threads
#
# The elementwise stages run on every core, in row bands.
#
# numpy is single threaded for elementwise work, and develop is mostly
# elementwise, so one core did all of it while eleven sat idle. Measured at
# 2560x1440 on twelve logical cores: the exposure curve, the saturation
# blend, the gamma and the clip together take 0.365 s in one thread and
# 0.079 s in twelve - 4.6x. The transcendentals carry it (`power` alone
# scales 5.0x, `exp` 2.4x); a plain multiply manages only 2.0x because it is
# bandwidth bound, not compute bound.
#
# IT IS BIT-IDENTICAL, and that is the reason to do it this way rather than
# on the GPU. A band computes the same outputs from the same inputs with the
# same arithmetic in the same order, so splitting changes nothing at all -
# which keeps every print reproducible and every test on CI. There is a test
# asserting it.
#
# Reductions are NOT banded. `auto_exposure` takes a percentile of the whole
# frame and grain is generated from one seed for the whole shape; both would
# change if computed per band, and both are cheap (0.05 s and less).
DEVELOP_THREADS = int(os.environ.get("ATLAS_DEVELOP_THREADS", "0")) \
    or min(8, os.cpu_count() or 1)

# Below this a dispatch costs more than the work it hands out.
_MIN_PAR_PIXELS = 1 << 19

_POOL = [None]


def _pool():
    if _POOL[0] is None:
        _POOL[0] = ThreadPoolExecutor(max_workers=DEVELOP_THREADS,
                                      thread_name_prefix="develop")
    return _POOL[0]


def _par(shape, fn, workers=None):
    """Call fn(y0, y1) over row bands, concurrently."""
    rows = int(shape[0])
    w = int(workers or DEVELOP_THREADS)
    if w <= 1 or rows < 2 or int(shape[0]) * int(shape[1]) < _MIN_PAR_PIXELS:
        fn(0, rows)
        return
    w = min(w, rows)
    cuts = [(rows * k // w, rows * (k + 1) // w) for k in range(w)]
    cuts = [c for c in cuts if c[1] > c[0]]
    # list() so an exception in a band propagates rather than being dropped
    list(_pool().map(lambda c: fn(c[0], c[1]), cuts))


def _par_tasks(fns, pixels=None):
    """Run independent WHOLE-ARRAY tasks concurrently.

    Not the same thing as `_par`, which cuts one array into row bands.
    These are separate blurs that happen to be wanted at the same moment -
    the three channels of a diffraction softening, the two legs of a
    halation annulus - each reading the input and writing its own output.

    BIT-IDENTICAL BY CONSTRUCTION, and for a stronger reason than the
    banded helpers can claim: no task shares an accumulator with any
    other, so every floating-point operation happens in the same order on
    the same values no matter what the scheduler does. Only the wall
    clock changes.

    THIS IS DELIBERATELY NOT THREADING THE BOX KERNEL. Doing that needs
    halo-overlapped bands plus a way to stop a band edge being taken for
    the image edge, where the window deliberately shrinks - and a band
    seam in a halation ring is exactly the kind of optical fault this
    project has shipped three times because it looked perfect. The
    structural parallelism here is most of the win for none of that risk.
    """
    fns = list(fns)
    if DEVELOP_THREADS <= 1 or len(fns) < 2 \
            or (pixels is not None and pixels < _MIN_PAR_PIXELS):
        for f in fns:
            f()
        return
    # list() so an exception in a task propagates rather than being dropped
    list(_pool().map(lambda f: f(), fns))


def _par_nan_to_num(a):
    out = np.empty_like(a)

    def band(y0, y1):
        out[y0:y1] = np.nan_to_num(a[y0:y1], nan=0.0, posinf=0.0, neginf=0.0)
    _par(a.shape, band)
    return out


def _par_expose(neg, E):
    """1 - exp(-neg * E), the plain print, band by band."""
    c = np.empty_like(neg)

    def band(y0, y1):
        b = c[y0:y1]
        np.multiply(neg[y0:y1], -E, out=b)
        np.exp(b, out=b)
        np.subtract(1.0, b, out=b)
    _par(neg.shape, band)
    return c


def _par_saturate(c, saturation):
    out = np.empty_like(c)

    def band(y0, y1):
        b = c[y0:y1]
        lum = (0.299 * b[..., 0] + 0.587 * b[..., 1]
               + 0.114 * b[..., 2])[..., None]
        out[y0:y1] = lum + (b - lum) * saturation
    _par(c.shape, band)
    return out


def _par_clip_pow(c, exponent):
    out = np.empty_like(c)

    def band(y0, y1):
        out[y0:y1] = np.clip(c[y0:y1], 0.0, 1.0) ** exponent
    _par(c.shape, band)
    return out


def _par_clip32(c):
    out = np.empty(c.shape, np.float32)

    def band(y0, y1):
        out[y0:y1] = np.clip(c[y0:y1], 0.0, 1.0).astype(np.float32)
    _par(c.shape, band)
    return out


def _hex(c):
    c = c.lstrip("#")
    return np.array([int(c[i:i + 2], 16) / 255.0 for i in (0, 2, 4)], np.float32)


# Below this radius a box pass is summed DIRECTLY rather than differenced
# out of a prefix sum. Two reasons, both measured at 2560x1440:
#
#   SPEED. A prefix-sum box is O(1) in the radius, which sounds like the
#   win and is actually the problem: it costs six full-image sequential
#   scans whatever you asked for, so 1.68 s at radius 1 exactly as at
#   radius 24. Direct sums are O(r) and cost 0.49 s at radius 1, 0.97 s at
#   8, crossing back over around 20.
#
#   ACCURACY. A box value from a prefix sum is the difference of two running
#   totals that ran the length of the row - ~1288 for this data, granular at
#   1.5e-4 in float32 - so it carried 1.3 to 3.4 levels of error at 16 bits.
#   Summing 2r+1 terms directly carries 0.01. Measured against a float64
#   reference of the same convention, direct is 130x to 340x closer.
#
# The two therefore disagree by up to 7 levels of 65535 on small radii, and
# it is the prefix-sum path that was wrong. That is a visible-to-nobody
# change to existing prints and a strictly better one.
#
# Where it matters most is not sharpening but OPTICS: halation at 140 um is
# about 2 px on a 1440p sheet and diffraction at f/32 is 0.3 px, so the
# nine blurs behind the diffraction-and-halation path were all paying the
# full prefix-sum price for a radius of one or two.
DIRECT_BOX_MAX_R = 14

# The SAME choice, made differently for the optical stages, because they
# carry a promise the artistic ones do not: a size specified in microns
# must land the same way on every sheet. `_boxn` feeds bloom, sharpening
# and the backdrop, all specified in pixels, and its threshold stays where
# it is. `_boxn_frac` is what halation, diffraction, Mackie lines and
# tricolour registration go through, and it gets a threshold set by the
# largest print rather than by the speed crossover.
#
# WHY IT IS NOT 14. Prefix-sum error grows with the length of the row it
# ran down, and a print is a very long row. Measured against a float64
# reference on negative-like data - dark ground, sparse bright spikes,
# which is the worst case because a dark window is recovered by
# differencing two large totals:
#
#     row     print @360ppi    direct      prefix     prefix err on
#                                                     the dark ground
#     2560    1440p            4.7e-07     1.6e-06     0.183%
#     14400   40 in            3.8e-07     1.5e-05     1.347%
#     25920   72 in            3.9e-07     3.6e-05     1.742%
#
# Direct is FLAT in the row length; the prefix sum degrades with exactly
# the thing this unit system exists to be invariant to. 32 covers every
# print the studio can make - a 48x72 in sheet asks for a box radius of
# 24.7 - with margin. Above r~20 direct costs about 1.4x on a pass, which
# is the price of the guarantee and is worth it.
DIRECT_BOX_MAX_R_EXACT = 32


def _box_direct(out, r, axis):
    """One box pass along `axis`, summed directly.

    Reproduces the shrinking window at the borders exactly: each output is
    divided by the number of samples actually covered, not by 2r+1.
    """
    m = out.shape[axis]
    acc = np.zeros_like(out)
    cnt = np.zeros(m, np.float32)
    for d in range(-r, r + 1):
        lo_s, hi_s = max(0, d), min(m, m + d)
        if hi_s <= lo_s:
            continue
        src = [slice(None)] * out.ndim
        dst = [slice(None)] * out.ndim
        src[axis] = slice(lo_s, hi_s)
        dst[axis] = slice(lo_s - d, hi_s - d)
        acc[tuple(dst)] += out[tuple(src)]
        cnt[lo_s - d:hi_s - d] += 1.0
    return acc / cnt.reshape([-1 if a == axis else 1
                              for a in range(out.ndim)])


def _boxn(img, r, n=3):
    """n box passes per axis; three approximates a gaussian.

    `r` is whole pixels. Small radii are summed directly and large ones
    differenced from a prefix sum - see DIRECT_BOX_MAX_R for the numbers.
    """
    out = img
    r = int(r)
    for _ in range(int(n)):
        for axis in (0, 1):
            if r <= DIRECT_BOX_MAX_R:
                out = _box_direct(out, r, axis)
                continue
            # `m`, not `n`: this rebound the pass count to the array
            # dimension. It happened to work, because `range(n)` above is
            # evaluated once, and it was a trap sitting there waiting for
            # somebody to convert the outer loop to a while.
            m = out.shape[axis]
            cs = np.cumsum(out, axis=axis, dtype=np.float32)
            cs = np.concatenate([np.zeros_like(np.take(cs, [0], axis=axis)), cs],
                                axis=axis)
            hi = np.minimum(np.arange(m) + r + 1, m)
            lo = np.maximum(np.arange(m) - r, 0)
            out = ((np.take(cs, hi, axis=axis)
                    - np.take(cs, lo, axis=axis))
                   / (hi - lo).reshape([-1 if a == axis else 1
                                        for a in range(out.ndim)]))
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
            if r0 <= DIRECT_BOX_MAX_R_EXACT:
                # THE PATH THE OPTICS ACTUALLY TAKE. Halation at 140 um is
                # ~2 px on a 1440p sheet and diffraction at f/32 is 0.3 px,
                # so every blur behind the eight-second optics path had a
                # radius of one or two and was paying for six full-image
                # prefix scans to get it. At print size the radius is 25
                # rather than 2, and the reason to sum directly stops being
                # speed and becomes accuracy - see DIRECT_BOX_MAX_R_EXACT.
                acc = np.zeros_like(out)
                cnt = np.zeros(m, np.float32)
                for d in range(-r0, r0 + 1):
                    lo_s, hi_s = max(0, d), min(m, m + d)
                    if hi_s <= lo_s:
                        continue
                    src = [slice(None)] * out.ndim
                    dst = [slice(None)] * out.ndim
                    src[axis] = slice(lo_s, hi_s)
                    dst[axis] = slice(lo_s - d, hi_s - d)
                    acc[tuple(dst)] += out[tuple(src)]
                    cnt[lo_s - d:hi_s - d] += 1.0
                # the partial pixel just outside the window on each side,
                # weighted by `frac`, present only where it is on the image
                for d in (r0 + 1, -r0 - 1):
                    lo_s, hi_s = max(0, d), min(m, m + d)
                    if hi_s <= lo_s:
                        continue
                    src = [slice(None)] * out.ndim
                    dst = [slice(None)] * out.ndim
                    src[axis] = slice(lo_s, hi_s)
                    dst[axis] = slice(lo_s - d, hi_s - d)
                    acc[tuple(dst)] += frac * out[tuple(src)]
                    cnt[lo_s - d:hi_s - d] += frac
                out = acc / cnt.reshape([-1 if a == axis else 1
                                         for a in range(out.ndim)])
                continue
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


def _pitch_um(width, height, fmt=None):
    """Microns of sheet per pixel of file.

    The sheet turns in the holder with the print, so its SHORT dimension
    always lies along the file's short axis - 96mm over the short pixel
    count, in either orientation. An earlier version picked the 120mm
    dimension for portrait files while still dividing by the short axis,
    which made every micron-sized stage (halation, diffraction, Mackie
    lines, tricolour registration) run about 20% smaller on a portrait
    print than on the identical sheet rotated - measured: a 140um ring
    was 10.5px on a 9000x7200 file and 8.4px on 7200x9000. That broke
    the one promise this unit system exists to keep.
    """
    sensor = SENSOR_H_UM
    if fmt is not None:
        # the format enters here and nowhere else in the optical stages,
        # which is the right place: an Airy disk is a fixed size in
        # microns and does not care what sheet it lands on. What changes
        # is how many PIXELS it covers, because a smaller sheet read out
        # at the same pixel count has a finer pitch - which is exactly
        # why a phone is diffraction limited by f/8 and a 4x5 is not
        # until f/64.
        from darkroom import formats
        sensor = formats.sensor_um(fmt)
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


def blur_microns(img, sigma_um, fmt=None):
    """Gaussian-ish blur of a width measured on the SHEET, not on the file.

    This is the whole reason the optical stages reproduce across formats.
    A 140-micron halation radius is 140 microns whether the negative is
    read out at 3600 pixels or 14400; expressed in pixels it would be 5 or
    21, and the same two numbers would describe two different lenses.
    Anything with a physical size belongs here.

    THERE IS NO FAST PATH, AND THERE USED TO BE. Above 12 px this called
    `_blur`, the multiresolution shortcut, on the grounds that at that
    width whole-pixel rounding is a rounding error. It is not: `_blur`
    truncates its radius with `int()` at a DOWNSAMPLED scale, so the sigma
    it delivers steps as the downsample factor changes. Measured with an
    impulse, requested against achieved:

        12 px -> +3.8%    13 px -> -4.2%    18 px -> +12.2%    24 px -> +3.8%

    The error changes SIGN, so it is not a calibration constant anybody
    could have divided out. Below 12 px, on this path, it is 0.00%.

    What that did downstream is the part that mattered. Halation is the
    difference of two of these blurs, so between the size where the outer
    leg crosses 12 px and the size where the inner one does, the annulus
    was one approximate blur minus one exact one. Its width held at
    exactly 77.00 um up to 24 in and then wandered between 76.26 and 93.49
    um - non-monotonically, 91.73 at 36 in and 76.26 at 40 in. That is the
    single promise this function exists to keep, kept below 28 in and
    broken above it.

    The cost of removing it is real and was chosen deliberately. Measured
    on a 24 Mpx tile and scaled: at the radius a 48x72 in sheet asks for,
    the exact path is 5.9x the shortcut, which puts a full-size halation
    near six minutes rather than one. Nothing below 12 px changed at all,
    so proofs and Super B are untouched - the cost lands only on the final
    render of a large print, which is not a thing anybody iterates on.
    `halation` and `diffraction_blur` now run their independent legs
    concurrently, which takes some of it back without touching this.
    """
    px = sigma_um / _pitch_um(img.shape[1], img.shape[0], fmt)
    if px < 0.03:
        return img
    return _boxn_frac(img, _radius_for_sigma(px), n=3)


def diffraction_blur(neg, fstop, *, amount=1.0, fmt=None):
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

    def channel(ch):
        # 1.22*lambda*N is the first zero; the gaussian that best fits an
        # Airy disk has sigma about 0.42 of it, which is the number to
        # blur by
        sigma_um = 0.42 * 1.22 * LAMBDA_UM[ch] * float(fstop) * amount

        def run():
            out[..., ch] = blur_microns(neg[..., ch:ch+1], sigma_um,
                                        fmt)[..., 0]
        return run

    # three wavelengths, three independent blurs, three disjoint output
    # channels - concurrent since the exact path made a blur expensive
    _par_tasks([channel(ch) for ch in range(3)], pixels=neg.shape[0] * neg.shape[1])
    return out


def halation(neg, *, strength=0.35, radius_um=140.0, inner=0.45,
             fmt=None):
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
    # the two legs of the annulus are independent blurs of the same input,
    # so they run at the same time. The subtraction below is unchanged and
    # so is every bit of its result.
    legs = [None, None]

    def leg(i, um):
        def run():
            legs[i] = blur_microns(neg, um, fmt)
        return run

    _par_tasks([leg(0, radius_um), leg(1, radius_um * inner)],
               pixels=neg.shape[0] * neg.shape[1])
    ring = np.clip(legs[0] - legs[1], 0.0, None)
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


def solarise(neg, E, *, amount=0.8, fog=0.18, shield=8.0,
             mackie=0.0, mackie_um=90.0, fmt=None):
    """The Sabattier effect: re-expose the print part-way through developing.

    This one is derived rather than imitated, which is worth spelling out
    because most implementations are a folded curve applied to the output.
    What actually happens is that development is interrupted and the whole
    sheet is exposed to light a second time. Silver already formed by the
    first development is opaque, so it SHIELDS the emulsion underneath it
    from the second exposure - Beer-Lambert, through the print's own
    density. Dense areas therefore receive almost nothing and resist;
    thin areas receive the lot and darken. The tonal scale partly reverses
    on its own, from the shielding, with no curve folded by hand:

        D2 = fog * exp(-shield * D1)

    MACKIE LINES are the second half of the effect and a separate
    mechanism: developer is consumed where it works hardest, so a point
    beside a heavily developing region finds less of it left. Density
    therefore gains where it exceeds its own neighbourhood and loses where
    it falls short, which is the adjacency effect, and along a strong edge
    it lays down the bright seam Man Ray built a career on. The
    neighbourhood is measured in microns on the sheet, so the seam is the
    same physical width at any print size.

    Returns a negative, so everything downstream is unchanged.
    """
    if amount <= 0 and mackie <= 0:
        return neg
    d1 = 1.0 - np.exp(-np.maximum(neg, 0.0) * E)      # density so far, 0..1
    d = d1
    if amount > 0:
        d = d1 + amount * fog * np.exp(-shield * d1)
    if mackie > 0:
        # developer exhaustion: what a point gains over its neighbourhood
        d = d + mackie * (d - blur_microns(d, mackie_um, fmt))
    d = np.clip(d, 0.0, 0.999)
    return (-np.log1p(-d) / max(E, 1e-9)).astype(np.float32)


# Three pigments, three separations, in register. The historic colour
# process, and the only one here that is genuinely subtractive rather than
# a tone map: each layer's transmittance multiplies the next, so where two
# pigments overlap the light really is filtered twice.
TRICOLOUR = {
    # carbro: pigmented gelatin, dense and glossy, a long scale
    "carbro": dict(base="#f4efe2", dmax=(1.55, 1.60, 1.45),
                   toe=(1.05, 1.05, 0.95), speed=(1.00, 1.00, 1.05),
                   pigments=("#12b5c8", "#d4176b", "#f2cf1e")),
    # gum bichromate: softer, shorter scale, the pigment sitting in relief
    "gum3":   dict(base="#f2ecdf", dmax=(1.25, 1.30, 1.20),
                   toe=(1.55, 1.55, 1.40), speed=(0.85, 0.85, 0.90),
                   pigments=("#2a9fb8", "#b83a72", "#d8bf4a")),
}


def _shift(img, dx, dy, fill=1.0):
    """Translate by a real number of pixels, bilinearly.

    Whole-pixel rolls would make misregistration a step function of print
    size, which is the same trap the optical stages fell into.

    The plate slides; it does not wrap. `np.roll` alone would carry one
    edge of the frame round to the other, so the image is padded with
    `fill` first and cropped after. For a pigment layer the honest fill
    is transmittance 1.0: beyond the slid plate's edge there is simply no
    pigment, so all the light passes."""
    if abs(dx) < 1e-4 and abs(dy) < 1e-4:
        return img
    m = int(np.ceil(max(abs(dx), abs(dy)))) + 1
    pad = np.full((img.shape[0] + 2 * m, img.shape[1] + 2 * m,
                   img.shape[2]), fill, img.dtype)
    pad[m:-m, m:-m] = img
    x0, y0 = int(np.floor(dx)), int(np.floor(dy))
    fx, fy = dx - x0, dy - y0
    out = np.zeros_like(pad)
    for ox, wx in ((x0, 1 - fx), (x0 + 1, fx)):
        if wx == 0:
            continue
        for oy, wy in ((y0, 1 - fy), (y0 + 1, fy)):
            if wy == 0:
                continue
            out += wx * wy * np.roll(np.roll(pad, oy, axis=0), ox, axis=1)
    return out[m:-m, m:-m]


def tricolour_print(neg, E, name="carbro", *, pigments=None, contrast=1.0,
                    dmax_mul=1.0, registration_um=0.0):
    """Separate through red, green and blue; print three pigment layers.

    We hold full colour data, so the separations are honest: the red
    separation really is what the red channel recorded, not a channel
    invented from a monochrome image. Each layer gets its own response
    curve, and the three transmittances MULTIPLY, which is what makes the
    colour subtractive rather than a palette applied afterwards.

    The one thing worth stating carefully is the SENSE of each layer,
    because getting it backwards is easy and gives a print with no colour
    in it at all - which is what a first attempt here did. The chain is
    subject -> separation negative -> pigment matrix, so the cyan matrix
    is made FROM the red separation and carries pigment in inverse
    proportion to the red light: cyan is what removes red, so it must be
    thin where the subject was red. The deposit is therefore the total
    quantity of light, split among the three pigments by the COMPLEMENT of
    the colour that light had. Light that was red lays down magenta and
    yellow and almost no cyan, and prints red.

    `registration_um` offsets each layer by that distance on the sheet, in
    a different direction per layer - three plates were printed in three
    passes and never landed perfectly, and that near-miss is most of why
    the real thing looks the way it does.
    """
    pr = TRICOLOUR[name]
    pig = pigments or pr["pigments"]
    base = _hex(pr["base"])
    t = np.ones_like(neg)
    px = (registration_um / _pitch_um(neg.shape[1], neg.shape[0])
          if registration_um else 0.0)
    neg = np.maximum(neg, 0.0)
    # how much light there was, and what colour it was
    lum = (0.299 * neg[..., 0] + 0.587 * neg[..., 1] + 0.114 * neg[..., 2])
    hue = neg / np.maximum(neg.max(axis=-1, keepdims=True), 1e-9)
    for i in range(3):
        dose = lum * (1.0 - np.clip(hue[..., i], 0.0, 1.0) * 0.92) \
            * E * pr["speed"][i]
        d = (pr["dmax"][i] * dmax_mul) * \
            (1.0 - np.exp(-dose)) ** (pr["toe"][i] * max(contrast, 1e-6))
        # the pigment absorbs the complement of its own colour
        absorb = 1.0 - _hex(pig[i]) * 0.92
        layer = np.power(10.0, -(d[..., None] * absorb))
        if px:
            ang = i * (2.0 * np.pi / 3.0)
            layer = _shift(layer, px * np.cos(ang), px * np.sin(ang))
        t = t * layer
    return (base * t).astype(np.float32)


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
            tricolour=None, registration_um=0.0, pigments=None,
            fmt=None,
            solar=0.0, solar_fog=0.18, solar_shield=8.0,
            mackie=0.0, mackie_um=90.0,
            fstop=0.0, diffraction=0.0,
            halation_strength=0.0, halation_radius=140.0,
            shadow=0.0, shadow_radius=0.02,
            shoulder=0.0, split=0.0,
            split_lo="#3a2c1e", split_hi="#dfe9f5"):
    """Negative to print. The order of stages is fixed and physical:

        diffraction -> halation -> [bloom] -> expose -> [solarise]
            -> process/paper/tricolour -> tone -> backdrop
            -> grain -> sharpen

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
    neg = _par_nan_to_num(neg)
    # AN F-NUMBER WITH DIFFRACTION OFF IS NEVER DELIBERATE. Off is not a
    # lie and stays the default, but an f-number is only ever passed by
    # a caller who wants the aperture's own optics - and a caller who
    # passes one and forgets `diffraction` gets a picture that looks
    # perfectly good and has no diffraction in it. That happened: a
    # figure whose entire subject was the aperture, captioned as showing
    # the diffraction limit, computed none. Nothing about the output
    # said so, which is the whole reason this warns.
    if fstop and diffraction <= 0:
        import warnings
        warnings.warn(
            f"develop() was given fstop={fstop:g} but diffraction=0, so "
            f"the aperture's Airy blur is NOT being computed. Pass "
            f"diffraction=1.0 to model it, or fstop=0 if the ideal lens "
            f"is what you meant.", RuntimeWarning, stacklevel=2)
    # the lens, then the emulsion - both before anything is developed
    if diffraction > 0 and fstop:
        neg = diffraction_blur(neg, fstop, amount=diffraction, fmt=fmt)
    if halation_strength > 0:
        neg = halation(neg, strength=halation_strength,
                       radius_um=halation_radius, fmt=fmt)
    E = (exposure if exposure is not None
         else auto_exposure(neg, percentile, target)) * (2.0 ** ev)
    if bloom > 0:
        sigma = bloom_radius * min(neg.shape[0], neg.shape[1])
        neg = neg + bloom * _blur(neg, sigma)     # halation in linear light
    # solarisation is a DEVELOPMENT effect: the sheet is re-exposed
    # part-way through developing, so it acts after the exposure is fixed
    # and before there is a print to read
    if solar > 0 or mackie > 0:
        neg = solarise(neg, E, amount=solar, fog=solar_fog,
                       shield=solar_shield, mackie=mackie,
                       mackie_um=mackie_um, fmt=fmt)
    if tricolour:
        c = tricolour_print(neg, E * ink_density, tricolour,
                            pigments=pigments, contrast=proc_contrast,
                            dmax_mul=proc_dmax,
                            registration_um=registration_um)
        if shadow > 0:
            base = _hex(TRICOLOUR[tricolour]["base"])
            sh = shadow_field(np.clip(c / np.maximum(base, 1e-6), 0, 1),
                              strength=shadow, radius=shadow_radius)
            c = c * (1.0 - sh[..., None])
    elif process:
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
        c = _par_expose(neg, E)
    c = _par_saturate(c, saturation)
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
    c = _par_clip_pow(c, (1.0 / gamma if (paper or process or tricolour)
                          else gamma))
    if vignette > 0:
        h, w = c.shape[:2]
        yy, xx = np.mgrid[0:h, 0:w]
        d2 = ((xx / w - 0.5) ** 2 + (yy / h - 0.5) ** 2)
        c *= (1.0 - vignette * 1.1 * d2)[..., None]
    if shoulder > 0 or split > 0:
        c = tone_curve(c, shoulder, split_lo, split_hi, split)
    c = _par_clip32(c)
    if sharpen > 0:
        c = unsharp(c, sharpen, sharpen_radius)
    if grain > 0:
        # strongest in the midtones, where film grain actually sits
        lum = 0.299 * c[..., 0] + 0.587 * c[..., 1] + 0.114 * c[..., 2]
        mask = (1.0 - np.abs(2.0 * lum - 1.0))[..., None]
        g = _grain(c.shape, grain, grain_size)[..., None]
        c = np.clip(c + g * mask, 0.0, 1.0)
    return c, E


def _srgb_icc():
    """The sRGB profile bytes, if Pillow's lcms is present. Cached."""
    global _ICC
    if _ICC is None:
        try:
            from PIL import ImageCms
            _ICC = ImageCms.ImageCmsProfile(
                ImageCms.createProfile("sRGB")).tobytes()
        except Exception:                                     # noqa: BLE001
            _ICC = b""
    return _ICC


_ICC = None


def write_print(path, img01, dpi=300, extratags=None, description=None):
    """16-bit TIFF with resolution tags so editors report the print size.

    An sRGB profile is embedded when available, so a lab's RIP reads the
    file's intent instead of guessing at an untagged TIFF - the pixels
    are unchanged either way."""
    out = (img01 * 65535.0 + 0.5).astype(np.uint16)
    tags = list(extratags or [])
    icc = _srgb_icc()
    if icc and not any(t[0] == 34675 for t in tags):
        tags.append((34675, 7, len(icc), icc, True))
    tifffile.imwrite(path, out, photometric="rgb", compression="zlib",
                     resolution=(dpi, dpi), resolutionunit="INCH",
                     description=description,
                     extratags=tags)


def write_preview(path, img01, max_side=2000, info=None):
    """8-bit preview. `info`, if given, is a dict embedded as PNG text -
    the same provenance the negatives carry, for files that are PNGs."""
    from PIL import Image
    im = Image.fromarray((img01 * 255.0 + 0.5).astype(np.uint8))
    im.thumbnail((max_side, max_side))
    kw = {}
    if info and str(path).lower().endswith(".png"):
        try:
            from PIL.PngImagePlugin import PngInfo
            import json as _json
            pi = PngInfo()
            pi.add_text("atlas", _json.dumps(info, default=str))
            kw["pnginfo"] = pi
        except Exception:                                     # noqa: BLE001
            pass
    im.save(path, **kw)
