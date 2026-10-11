"""Art generation algorithms and the pattern registry.

Each algorithm takes coordinate grids (x, y), image size, a color
key, and an array module (numpy or cupy).  Some additionally need
the backend name and a cancellation flag.
"""

from __future__ import annotations

import numpy as np

from config import ParamDef, PatternDef
from backends import (
    _dispatch_blur, _dispatch_mandelbrot, _dispatch_julia,
    HAS_CUPY, cp,
)
from colorizer import colorizer

__all__ = ["PATTERNS", "PATTERN_MAP", "JULIA_PRESETS"]


# ═══════════════════════════════════════════════════════════════
#  ALGORITHMS
# ═══════════════════════════════════════════════════════════════

def spiral_art(x, y, size, ck, xp, frequency=6.0, tightness=4.0):
    h = size * 0.5
    r = xp.sqrt((x - h) ** 2 + (y - h) ** 2)
    a = xp.arctan2(y - h, x - h)
    v = xp.sin(r / tightness + a * frequency) * 127 + 128
    return colorizer.colorize(v, ck, xp)


def noise_art(x, y, size, ck, xp, backend, cancel, seed=42):
    rng = np.random.RandomState(seed)
    n = rng.rand(size, size).astype(np.float32) * 255
    n = _dispatch_blur(n, 3, backend)
    return colorizer.colorize(n, ck, np)


def maze_art(x, y, size, ck, xp, cell_size=16):
    v = (np.bitwise_xor((x // cell_size) & 1, (y // cell_size) & 1)).astype(np.float64) * 255
    return colorizer.colorize(v, ck, np)


def mandelbrot_art(x, y, size, ck, xp, backend, cancel,
                   max_iter=255, center_x=-0.5, center_y=0.0, zoom=1.0):
    inv_z = 1.0 / zoom
    xr, yr = 3.5 * inv_z, 2.0 * inv_z
    inv_s = 1.0 / size
    cx = x * inv_s * xr + center_x - xr * 0.5
    cy = y * inv_s * yr + center_y - yr * 0.5
    div = _dispatch_mandelbrot(cx, cy, max_iter, cancel, backend)
    if div is None:
        return None
    mx = div.max()
    v = (div / mx * 255) if mx > 0 else div
    return colorizer.colorize(v, ck, np)


def julia_art(x, y, size, ck, xp, backend, cancel,
              max_iter=255, julia_r=-0.7, julia_i=0.27015, zoom=1.0):
    inv_z = 1.0 / zoom
    span = 3.0 * inv_z
    inv_s = 1.0 / size
    zr = x * inv_s * span - span * 0.5
    zi = y * inv_s * span - span * 0.5
    div = _dispatch_julia(zr, zi, julia_r, julia_i, max_iter, cancel, backend)
    if div is None:
        return None
    mx = div.max()
    v = (div / mx * 255) if mx > 0 else div
    return colorizer.colorize(v, ck, np)


def wave_art(x, y, size, ck, xp, wavelength=10.0, sources=1):
    pts = ((0.25, 0.25), (0.75, 0.25), (0.50, 0.75))
    v = xp.zeros((size, size), dtype=np.float64)
    for sx, sy in pts[:sources]:
        dx, dy = x - size * sx, y - size * sy
        v += xp.sin(xp.sqrt(dx * dx + dy * dy) / wavelength)
    return colorizer.colorize(v / max(sources, 1) * 127 + 128, ck, xp)


def checkerboard_art(x, y, size, ck, xp, tile_size=32):
    return colorizer.colorize(((x // tile_size + y // tile_size) & 1) * 255.0, ck, xp)


def rings_art(x, y, size, ck, xp, ring_width=6.0):
    h = size * 0.5
    r = xp.sqrt((x - h) ** 2 + (y - h) ** 2)
    return colorizer.colorize(((xp.sin(r / ring_width) > 0) * 255), ck, xp)


def plasma_art(x, y, size, ck, xp, freq_x=4.0, freq_y=4.0, phase=0.0):
    PI = xp.pi
    v = (xp.sin(x / size * freq_x * PI + phase)
         + xp.sin(y / size * freq_y * PI + phase * 0.7)
         + xp.sin((x + y) / size * freq_x * 0.5 * PI + phase * 1.3)
         + xp.sin(xp.sqrt((x - size / 2) ** 2 + (y - size / 2) ** 2)
                  / size * freq_x * PI)) / 4.0
    return colorizer.colorize(v * 127 + 128, ck, xp)


# ═══════════════════════════════════════════════════════════════
#  JULIA SET PRESETS  (used by randomizer for good results)
# ═══════════════════════════════════════════════════════════════

JULIA_PRESETS: list[tuple[float, float]] = [
    (-0.7, 0.27015), (-0.8, 0.156), (0.285, 0.01),
    (-0.4, 0.6), (0.355, 0.355), (-0.54, 0.54),
    (0.37, 0.1), (-0.12, 0.75), (-0.75, 0.11),
    (0.28, 0.008), (-0.62, 0.42), (-0.1, 0.65),
]


# ═══════════════════════════════════════════════════════════════
#  PATTERN REGISTRY
# ═══════════════════════════════════════════════════════════════

PATTERNS: tuple[PatternDef, ...] = (
    PatternDef("Spiral", spiral_art,
               "Hypnotic spiral with adjustable frequency and tightness.",
               (ParamDef("frequency", "Frequency", True, 1, 20, 6, 0.5),
                ParamDef("tightness", "Tightness", True, 1, 20, 4, 0.5))),
    PatternDef("Noise", noise_art,
               "Smoothed random noise — accelerated with Numba or GPU blur.",
               (ParamDef("seed", "Seed", False, 0, 9999, 42, 1),),
               needs_backend_dispatch=True),
    PatternDef("Maze", maze_art,
               "XOR-based maze grid pattern.",
               (ParamDef("cell_size", "Cell Size", False, 4, 64, 16, 2),)),
    PatternDef("Mandelbrot", mandelbrot_art,
               "Classic fractal — Numba parallel or GPU accelerated.",
               (ParamDef("max_iter", "Iterations", False, 32, 512, 255, 1),
                ParamDef("zoom", "Zoom", True, 0.5, 100, 1, 0.5),
                ParamDef("center_x", "Center X", True, -2.5, 1.5, -0.5, 0.05),
                ParamDef("center_y", "Center Y", True, -1.5, 1.5, 0, 0.05)),
               needs_backend_dispatch=True),
    PatternDef("Julia Set", julia_art,
               "Julia set fractal — explore beautiful constant variations.",
               (ParamDef("max_iter", "Iterations", False, 32, 512, 255, 1),
                ParamDef("julia_r", "C (real)", True, -1.5, 1.5, -0.7, 0.01),
                ParamDef("julia_i", "C (imag)", True, -1.5, 1.5, 0.27015, 0.01),
                ParamDef("zoom", "Zoom", True, 0.5, 50, 1, 0.5)),
               needs_backend_dispatch=True),
    PatternDef("Wave", wave_art,
               "Interference wave patterns from point sources.",
               (ParamDef("wavelength", "Wavelength", True, 2, 40, 10, 1),
                ParamDef("sources", "Sources", False, 1, 3, 1, 1))),
    PatternDef("Checkerboard", checkerboard_art,
               "Classic checkerboard with adjustable tile size.",
               (ParamDef("tile_size", "Tile Size", False, 4, 128, 32, 4),)),
    PatternDef("Rings", rings_art,
               "Concentric rings radiating from center.",
               (ParamDef("ring_width", "Ring Width", True, 1, 30, 6, 1),)),
    PatternDef("Plasma", plasma_art,
               "Classic plasma effect from layered sine waves.",
               (ParamDef("freq_x", "Freq X", True, 1, 12, 4, 0.5),
                ParamDef("freq_y", "Freq Y", True, 1, 12, 4, 0.5),
                ParamDef("phase", "Phase", True, 0, 6.28, 0, 0.1))),
)

PATTERN_MAP: dict[str, PatternDef] = {p.name: p for p in PATTERNS}