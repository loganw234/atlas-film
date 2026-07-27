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


def _blur(img, sigma_px):
    """Separable gaussian-ish blur: three box passes via cumsum. O(n)."""
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


def _backdrop(shape, kind, col_a, col_b, center, strength):
    """A studio backdrop in linear-ish display space: col_a at the edges,
    col_b at the hot spot. Screened under the render by the caller."""
    h, w = shape[:2]
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    u, v = xx / w, yy / h
    cx, cy = center
    if kind == "linear":
        t = 1.0 - np.clip(np.abs(v - cy) * 1.6, 0.0, 1.0)
    else:  # radial
        t = 1.0 - np.clip(np.sqrt((u - cx) ** 2 + ((v - cy) * h / w) ** 2) * 1.9,
                          0.0, 1.0)
    t = (t * t)[..., None]
    a, b = _hex(col_a), _hex(col_b)
    return strength * (a + (b - a) * t)


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


def develop(neg, *, exposure=None, ev=0.0, gamma=0.82, saturation=1.0,
            vignette=0.0, percentile=99.5, target=0.85,
            bloom=0.0, bloom_radius=0.015,
            bg_kind=None, bg_a="#0c0c12", bg_b="#1c1826",
            bg_center=(0.5, 0.45), bg_strength=1.0,
            palette=None, palette_mix=1.0,
            sharpen=0.0, sharpen_radius=1.2):
    E = (exposure if exposure is not None
         else auto_exposure(neg, percentile, target)) * (2.0 ** ev)
    if bloom > 0:
        sigma = bloom_radius * min(neg.shape[0], neg.shape[1])
        neg = neg + bloom * _blur(neg, sigma)     # halation in linear light
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
        bg = _backdrop(c.shape, bg_kind, bg_a, bg_b, bg_center, bg_strength)
        c = bg + c - bg * c                        # screen: light on backdrop
    c = np.clip(c, 0.0, 1.0) ** gamma
    if vignette > 0:
        h, w = c.shape[:2]
        yy, xx = np.mgrid[0:h, 0:w]
        d2 = ((xx / w - 0.5) ** 2 + (yy / h - 0.5) ** 2)
        c *= (1.0 - vignette * 1.1 * d2)[..., None]
    c = np.clip(c, 0.0, 1.0).astype(np.float32)
    if sharpen > 0:
        c = unsharp(c, sharpen, sharpen_radius)
    return c, E


def write_print(path, img01, dpi=300):
    """16-bit TIFF with resolution tags so editors report the print size."""
    out = (img01 * 65535.0 + 0.5).astype(np.uint16)
    tifffile.imwrite(path, out, photometric="rgb", compression="zlib",
                     resolution=(dpi, dpi), resolutionunit="INCH")


def write_preview(path, img01, max_side=2000):
    from PIL import Image
    im = Image.fromarray((img01 * 255.0 + 0.5).astype(np.uint8))
    im.thumbnail((max_side, max_side))
    im.save(path)
