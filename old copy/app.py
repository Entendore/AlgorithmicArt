#!/usr/bin/env python3
"""
Algorithmic Art Studio — Numba JIT + CuPy GPU + Palette System.

Backend priority: GPU (CuPy) → CPU (Numba) → CPU (NumPy)
Gracefully degrades when optional deps are missing.
"""

import sys
import time
import random
from dataclasses import dataclass
from typing import Callable

import numpy as np
from numpy.typing import NDArray

from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QComboBox, QSlider, QPushButton, QFrame,
    QFileDialog, QStatusBar, QSizePolicy, QGridLayout
)
from PySide6.QtGui import (
    QPixmap, QImage, QFont, QColor, QPainter, QKeySequence, QShortcut
)
from PySide6.QtCore import Qt, QSize, QThread, Signal, QTimer

# ═══════════════════════════════════════════════════════════════
#  OPTIONAL DEPENDENCY DETECTION
# ═══════════════════════════════════════════════════════════════

try:
    from numba import njit, prange as _prange
    HAS_NUMBA = True
except ImportError:
    HAS_NUMBA = False

try:
    import cupy as cp
    _cp_test = cp.zeros(1)
    del _cp_test
    HAS_CUPY = True
except Exception:
    HAS_CUPY = False

if HAS_NUMBA:
    def _numba_jit(fn):
        return njit(parallel=True, fastmath=True, cache=True)(fn)
else:
    def _numba_jit(fn):
        return fn

_prange = _prange if HAS_NUMBA else range


# ═══════════════════════════════════════════════════════════════
#  DATA STRUCTURES
# ═══════════════════════════════════════════════════════════════

@dataclass(frozen=True, slots=True)
class ParamDef:
    key: str
    label: str
    is_float: bool
    min_val: float
    max_val: float
    default: float
    step: float


@dataclass(frozen=True, slots=True)
class PatternDef:
    name: str
    fn: Callable
    description: str
    params: tuple[ParamDef, ...] = ()
    needs_backend_dispatch: bool = False


# ═══════════════════════════════════════════════════════════════
#  PALETTE DEFINITIONS  (Nx4 arrays: [pos, R, G, B] in [0,1])
# ═══════════════════════════════════════════════════════════════

def _p(*rows):
    return np.array(rows, dtype=np.float32)

PALETTES: dict[str, NDArray] = {
    "inferno": _p(
        (0.00, 0.001, 0.000, 0.014), (0.13, 0.106, 0.036, 0.260),
        (0.25, 0.258, 0.023, 0.463), (0.38, 0.506, 0.016, 0.506),
        (0.50, 0.735, 0.106, 0.370), (0.63, 0.902, 0.301, 0.175),
        (0.75, 0.973, 0.557, 0.067), (0.88, 0.988, 0.798, 0.183),
        (1.00, 0.988, 0.998, 0.645)),
    "ocean": _p(
        (0.00, 0.004, 0.020, 0.110), (0.20, 0.000, 0.075, 0.290),
        (0.40, 0.004, 0.180, 0.467), (0.60, 0.020, 0.357, 0.627),
        (0.80, 0.133, 0.580, 0.788), (1.00, 0.494, 0.851, 0.973)),
    "viridis": _p(
        (0.00, 0.267, 0.004, 0.329), (0.13, 0.283, 0.141, 0.458),
        (0.25, 0.254, 0.265, 0.530), (0.38, 0.163, 0.471, 0.558),
        (0.50, 0.128, 0.567, 0.551), (0.63, 0.134, 0.659, 0.517),
        (0.75, 0.267, 0.749, 0.441), (0.88, 0.478, 0.821, 0.318),
        (1.00, 0.993, 0.906, 0.144)),
    "plasma": _p(
        (0.00, 0.050, 0.030, 0.529), (0.13, 0.234, 0.014, 0.635),
        (0.25, 0.405, 0.016, 0.651), (0.38, 0.580, 0.043, 0.596),
        (0.50, 0.741, 0.132, 0.497), (0.63, 0.862, 0.283, 0.373),
        (0.75, 0.940, 0.467, 0.247), (0.88, 0.973, 0.682, 0.141),
        (1.00, 0.940, 0.975, 0.131)),
    "magma": _p(
        (0.00, 0.001, 0.000, 0.014), (0.13, 0.106, 0.031, 0.267),
        (0.25, 0.223, 0.090, 0.408), (0.38, 0.366, 0.133, 0.435),
        (0.50, 0.529, 0.169, 0.404), (0.63, 0.702, 0.263, 0.341),
        (0.75, 0.855, 0.396, 0.286), (0.88, 0.945, 0.592, 0.263),
        (1.00, 0.987, 0.991, 0.750)),
    "cividis": _p(
        (0.00, 0.152, 0.224, 0.369), (0.25, 0.275, 0.443, 0.533),
        (0.50, 0.393, 0.640, 0.553), (0.75, 0.694, 0.839, 0.602),
        (1.00, 0.941, 0.912, 0.549)),
    "turbo": _p(
        (0.00, 0.302, 0.000, 0.544), (0.10, 0.235, 0.265, 0.729),
        (0.20, 0.167, 0.478, 0.804), (0.30, 0.101, 0.671, 0.757),
        (0.40, 0.075, 0.820, 0.635), (0.50, 0.176, 0.925, 0.455),
        (0.60, 0.383, 0.973, 0.259), (0.70, 0.620, 0.969, 0.133),
        (0.80, 0.831, 0.902, 0.067), (0.90, 0.976, 0.745, 0.051),
        (1.00, 0.735, 0.016, 0.016)),
    "hot": _p(
        (0.00, 0.0, 0.0, 0.0), (0.35, 0.8, 0.0, 0.0),
        (0.67, 1.0, 0.6, 0.0), (1.00, 1.0, 1.0, 1.0)),
    "cool": _p(
        (0.00, 0.0, 1.0, 1.0), (1.00, 1.0, 0.0, 1.0)),
    "bone": _p(
        (0.00, 0.0, 0.0, 0.0), (0.25, 0.253, 0.253, 0.376),
        (0.50, 0.569, 0.569, 0.694), (0.75, 0.855, 0.855, 0.933),
        (1.00, 1.0, 1.0, 1.0)),
    "spring": _p(
        (0.00, 1.0, 0.0, 1.0), (1.00, 1.0, 1.0, 0.0)),
    "summer": _p(
        (0.00, 0.0, 0.5, 0.4), (1.00, 1.0, 1.0, 0.4)),
    "autumn": _p(
        (0.00, 1.0, 0.0, 0.0), (1.00, 1.0, 1.0, 0.0)),
    "winter": _p(
        (0.00, 0.0, 0.0, 1.0), (1.00, 0.0, 1.0, 0.5)),
    "cubehelix": _p(
        (0.00, 0.0, 0.0, 0.0), (0.17, 0.051, 0.180, 0.425),
        (0.33, 0.188, 0.408, 0.553), (0.50, 0.364, 0.636, 0.510),
        (0.67, 0.588, 0.753, 0.369), (0.83, 0.800, 0.780, 0.220),
        (1.00, 1.0, 1.0, 1.0)),
    "twilight": _p(
        (0.00, 0.164, 0.169, 0.498), (0.14, 0.584, 0.133, 0.588),
        (0.28, 0.867, 0.204, 0.514), (0.42, 0.973, 0.439, 0.349),
        (0.57, 0.890, 0.675, 0.216), (0.71, 0.635, 0.863, 0.224),
        (0.85, 0.380, 0.922, 0.449), (1.00, 0.164, 0.169, 0.498)),
}

_ALGO_COLORS = [
    ("Grayscale", "grayscale"),
    ("Channel Shift", "channel_shift"),
    ("Neon", "neon"),
    ("Spectrum", "spectrum"),
]
_PAL_COLORS = [
    ("Inferno", "inferno"), ("Ocean", "ocean"), ("Viridis", "viridis"),
    ("Plasma", "plasma"), ("Magma", "magma"), ("Cividis", "cividis"),
    ("Turbo", "turbo"), ("Hot", "hot"), ("Cool", "cool"),
    ("Bone", "bone"), ("Spring", "spring"), ("Summer", "summer"),
    ("Autumn", "autumn"), ("Winter", "winter"), ("Cubehelix", "cubehelix"),
    ("Twilight", "twilight"),
]


# ═══════════════════════════════════════════════════════════════
#  LUT COLORIZER  (O(1) per-pixel, GPU-aware)
# ═══════════════════════════════════════════════════════════════

class Colorizer:
    """Precomputed 256-entry LUTs; caches GPU copies."""

    def __init__(self) -> None:
        self._cpu: dict[str, NDArray] = {}
        self._gpu: dict[str, object] = {}
        self._build_all()

    @staticmethod
    def _from_stops(stops: NDArray) -> NDArray:
        t = np.linspace(0.0, 1.0, 256, dtype=np.float32)
        lut = np.empty((256, 3), dtype=np.float32)
        for ch in range(3):
            lut[:, ch] = np.interp(t, stops[:, 0], stops[:, ch + 1])
        return lut

    @staticmethod
    def _hsv_lut(h, s, v):
        h6 = (h % 1.0) * 6.0
        sec = np.floor(h6).astype(np.int32) % 6
        f = h6 - np.floor(h6)
        p, q, t = v * (1 - s), v * (1 - s * f), v * (1 - s * (1 - f))
        rc = [v, q, p, p, t, v]
        gc = [t, v, v, q, p, p]
        bc = [p, p, t, v, v, q]
        lut = np.empty((256, 3), dtype=np.float32)
        for i in range(6):
            m = sec == i
            lut[m, 0] = rc[i][m]
            lut[m, 1] = gc[i][m]
            lut[m, 2] = bc[i][m]
        return lut

    def _build_all(self):
        t = np.linspace(0, 1, 256, dtype=np.float32)
        self._cpu["spectrum"] = self._hsv_lut(t, np.full(256, .85, np.float32),
                                               np.clip(t * 1.4 + .15, 0, 1))
        self._cpu["neon"] = self._hsv_lut((t * 2.5 + .55) % 1, np.ones(256, np.float32),
                                           np.clip(t * 1.8 + .25, 0, 1))
        for name, stops in PALETTES.items():
            self._cpu[name] = self._from_stops(stops)

    def get_preview_lut(self, key: str) -> NDArray | None:
        return self._cpu.get(key)

    def colorize(self, base, key: str, xp=np):
        if key == "grayscale":
            return xp.clip(base, 0, 255).astype(xp.uint8)
        if key == "channel_shift":
            r = base
            g = xp.roll(base, 15, axis=0)
            b = xp.roll(base, -15, axis=1)
            return xp.clip(xp.stack([r, g, b], axis=2), 0, 255).astype(xp.uint8)
        lut = self._lut(key, xp)
        idx = xp.clip(base, 0, 255).astype(xp.uint8)
        return (xp.clip(lut[idx], 0.0, 1.0) * 255).astype(xp.uint8)

    def _lut(self, key, xp):
        if xp is np:
            return self._cpu[key]
        if key not in self._gpu:
            self._gpu[key] = cp.asarray(self._cpu[key])
        return self._gpu[key]

_colorizer = Colorizer()


# ═══════════════════════════════════════════════════════════════
#  COORDINATE CACHE  (LRU, GPU-aware)
# ═══════════════════════════════════════════════════════════════

class CoordCache:
    def __init__(self, cap: int = 8) -> None:
        self._c: dict[tuple[int, bool], tuple] = {}
        self._o: list[tuple[int, bool]] = []
        self._cap = cap

    def get(self, size: int, gpu: bool = False):
        k = (size, gpu)
        if k in self._c:
            self._o.remove(k)
            self._o.append(k)
            return self._c[k]
        if len(self._c) >= self._cap:
            del self._c[self._o.pop(0)]
        x, y = np.indices((size, size), dtype=np.float64)
        if gpu:
            x, y = cp.asarray(x), cp.asarray(y)
        self._c[k] = (x, y)
        self._o.append(k)
        return x, y

_coord_cache = CoordCache()


# ═══════════════════════════════════════════════════════════════
#  NUMBA-ACCELERATED FUNCTIONS
# ═══════════════════════════════════════════════════════════════

@_numba_jit
def _box_blur_numba(arr: NDArray, passes: int) -> NDArray:
    h, w = arr.shape
    out = arr.copy()
    for _ in range(passes):
        tmp = np.empty_like(out)
        for i in _prange(h):
            i0 = 0 if i == 0 else i - 1
            i1 = h if i == h - 1 else i + 2
            for j in range(w):
                j0 = 0 if j == 0 else j - 1
                j1 = w if j == w - 1 else j + 2
                s = 0.0
                for ii in range(i0, i1):
                    for jj in range(j0, j1):
                        s += out[ii, jj]
                tmp[i, j] = s / ((i1 - i0) * (j1 - j0))
        out = tmp
    return out


@_numba_jit
def _mandelbrot_numba(cx: NDArray, cy: NDArray,
                      max_iter: int, cancel: NDArray) -> NDArray:
    h, w = cx.shape
    div = np.empty((h, w), dtype=np.float64)
    LN2 = 0.6931471805599453
    for i in _prange(h):
        for j in range(w):
            if cancel[0]:
                div[i, j] = 0.0
                continue
            cr, ci = cx[i, j], cy[i, j]
            zr, zi = 0.0, 0.0
            n = 0
            while n < max_iter:
                zr2 = zr * zr
                zi2 = zi * zi
                if zr2 + zi2 > 4.0:
                    break
                zi = 2.0 * zr * zi + ci
                zr = zr2 - zi2 + cr
                n += 1
            if n < max_iter:
                mag2 = zr * zr + zi * zi
                log_zn = np.log(max(mag2, 1.0)) * 0.5
                nu = np.log(max(log_zn / LN2, 1e-10)) / LN2
                div[i, j] = n + 1.0 - nu
            else:
                div[i, j] = float(max_iter)
    return div


# Warm up Numba at import time
if HAS_NUMBA:
    _w = np.zeros((2, 2), dtype=np.float64)
    _mandelbrot_numba(_w, _w, 2, np.array([False], dtype=np.bool_))
    _box_blur_numba(np.zeros((4, 4), dtype=np.float32), 1)
    del _w


# ═══════════════════════════════════════════════════════════════
#  CUPY-ACCELERATED FUNCTIONS
# ═══════════════════════════════════════════════════════════════

def _box_blur_gpu(arr, passes: int = 3):
    a = cp.asarray(arr, dtype=cp.float32)
    for _ in range(passes):
        p = cp.pad(a, 1, mode="reflect")
        a = (p[:-2, :-2] + p[:-2, 1:-1] + p[:-2, 2:] +
             p[1:-1, :-2] + p[1:-1, 1:-1] + p[1:-1, 2:] +
             p[2:, :-2] + p[2:, 1:-1] + p[2:, 2:]) / 9.0
    return a


def _mandelbrot_gpu(cx, cy, max_iter: int, cancel: NDArray):
    c = cp.asarray(cx) + 1j * cp.asarray(cy)
    z = cp.zeros(c.shape, dtype=cp.complex128)
    div = cp.full(c.shape, float(max_iter), dtype=cp.float64)
    esc = cp.zeros(c.shape, dtype=cp.bool_)
    for i in range(max_iter):
        if cancel[0]:
            return None
        z_new = z * z + c
        hit = ~esc & (cp.abs(z_new) > 2.0)
        safe_abs = cp.maximum(cp.abs(z_new), 1.0)
        smooth = i + 1.0 - cp.log2(cp.maximum(cp.log2(safe_abs), 0.001))
        div = cp.where(hit, smooth, div)
        esc = esc | hit
        z = cp.where(esc, z, z_new)
    return cp.asnumpy(div)


# ═══════════════════════════════════════════════════════════════
#  NUMPY FALLBACK FUNCTIONS
# ═══════════════════════════════════════════════════════════════

def _box_blur_numpy(arr, passes: int = 3):
    a = arr.astype(np.float32)
    for _ in range(passes):
        p = np.pad(a, 1, mode="reflect")
        a = (p[:-2, :-2] + p[:-2, 1:-1] + p[:-2, 2:] +
             p[1:-1, :-2] + p[1:-1, 1:-1] + p[1:-1, 2:] +
             p[2:, :-2] + p[2:, 1:-1] + p[2:, 2:]) / 9.0
    return a


def _mandelbrot_numpy(cx, cy, max_iter: int, cancel: NDArray):
    c = cx + 1j * cy
    z = np.zeros_like(c, dtype=np.complex128)
    div = np.full(c.shape, float(max_iter), dtype=np.float64)
    esc = np.zeros(c.shape, dtype=bool)
    for i in range(max_iter):
        if cancel[0]:
            return None
        active = ~esc & (np.abs(z) <= 2.0)
        if not np.any(active):
            break
        z[active] = z[active] ** 2 + c[active]
        hit = active & (np.abs(z) > 2.0)
        if np.any(hit):
            safe = np.maximum(np.abs(z[hit]), 1.0)
            div[hit] = i + 1.0 - np.log2(np.maximum(np.log2(safe), 0.001))
            esc[hit] = True
    return div


# ═══════════════════════════════════════════════════════════════
#  UNIFIED BACKEND DISPATCH
# ═══════════════════════════════════════════════════════════════

def _dispatch_blur(arr, passes: int, backend: str):
    if backend == "gpu" and HAS_CUPY:
        return cp.asnumpy(_box_blur_gpu(arr, passes))
    if backend == "numba" and HAS_NUMBA:
        return _box_blur_numba(arr.astype(np.float32), passes)
    return _box_blur_numpy(arr, passes)


def _dispatch_mandelbrot(cx, cy, max_iter, cancel, backend):
    if backend == "gpu" and HAS_CUPY:
        return _mandelbrot_gpu(cx, cy, max_iter, cancel)
    if backend == "numba" and HAS_NUMBA:
        return _mandelbrot_numba(cx, cy, max_iter, cancel)
    return _mandelbrot_numpy(cx, cy, max_iter, cancel)


# ═══════════════════════════════════════════════════════════════
#  ART ALGORITHMS
# ═══════════════════════════════════════════════════════════════

def spiral_art(x, y, size, ck, xp, frequency=6.0, tightness=4.0):
    h = size * 0.5
    r = xp.sqrt((x - h) ** 2 + (y - h) ** 2)
    a = xp.arctan2(y - h, x - h)
    v = xp.sin(r / tightness + a * frequency) * 127 + 128
    return _colorizer.colorize(v, ck, xp)


def noise_art(x, y, size, ck, xp, backend, seed=42):
    rng = np.random.RandomState(seed)
    n = rng.rand(size, size).astype(np.float32) * 255
    n = _dispatch_blur(n, 3, backend)
    return _colorizer.colorize(n, ck, np)


def maze_art(x, y, size, ck, xp, cell_size=16):
    v = (np.bitwise_xor((x // cell_size) & 1, (y // cell_size) & 1)).astype(np.float64) * 255
    return _colorizer.colorize(v, ck, np)


def mandelbrot_art(x_cpu, y_cpu, size, ck, backend, cancel,
                   max_iter=255, center_x=-0.5, center_y=0.0, zoom=1.0):
    inv_z = 1.0 / zoom
    xr, yr = 3.5 * inv_z, 2.0 * inv_z
    inv_s = 1.0 / size
    cx = x_cpu * inv_s * xr + center_x - xr * 0.5
    cy = y_cpu * inv_s * yr + center_y - yr * 0.5
    div = _dispatch_mandelbrot(cx, cy, max_iter, cancel, backend)
    if div is None:
        return None
    mx = div.max()
    v = (div / mx * 255) if mx > 0 else div
    return _colorizer.colorize(v, ck, np)


def wave_art(x, y, size, ck, xp, wavelength=10.0, sources=1):
    pts = ((0.25, 0.25), (0.75, 0.25), (0.50, 0.75))
    v = xp.zeros((size, size), dtype=np.float64)
    for sx, sy in pts[:sources]:
        dx, dy = x - size * sx, y - size * sy
        v += xp.sin(xp.sqrt(dx * dx + dy * dy) / wavelength)
    return _colorizer.colorize(v / max(sources, 1) * 127 + 128, ck, xp)


def checkerboard_art(x, y, size, ck, xp, tile_size=32):
    return _colorizer.colorize(((x // tile_size + y // tile_size) & 1) * 255.0, ck, xp)


def rings_art(x, y, size, ck, xp, ring_width=6.0):
    h = size * 0.5
    r = xp.sqrt((x - h) ** 2 + (y - h) ** 2)
    return _colorizer.colorize(((xp.sin(r / ring_width) > 0) * 255), ck, xp)


# ═══════════════════════════════════════════════════════════════
#  PATTERN REGISTRY
# ═══════════════════════════════════════════════════════════════

PATTERNS: tuple[PatternDef, ...] = (
    PatternDef("Spiral", spiral_art,
               "Hypnotic spiral with adjustable frequency and tightness.",
               (ParamDef("frequency", "Frequency", True, 1, 20, 6, .5),
                ParamDef("tightness", "Tightness", True, 1, 20, 4, .5))),
    PatternDef("Noise", noise_art,
               "Smoothed random noise — Numba blur or GPU blur.",
               (ParamDef("seed", "Seed", False, 0, 9999, 42, 1),),
               needs_backend_dispatch=True),
    PatternDef("Maze", maze_art,
               "XOR-based maze grid pattern.",
               (ParamDef("cell_size", "Cell Size", False, 4, 64, 16, 2),)),
    PatternDef("Mandelbrot", mandelbrot_art,
               "Fractal — Numba parallel or GPU accelerated.",
               (ParamDef("max_iter", "Iterations", False, 32, 512, 255, 1),
                ParamDef("zoom", "Zoom", True, 0.5, 100, 1, .5),
                ParamDef("center_x", "Center X", True, -2.5, 1.5, -.5, .05),
                ParamDef("center_y", "Center Y", True, -1.5, 1.5, 0, .05)),
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
)
_PAT = {p.name: p for p in PATTERNS}


# ═══════════════════════════════════════════════════════════════
#  BACKGROUND WORKER THREAD  (safe cancel via shared flag)
# ═══════════════════════════════════════════════════════════════

class ArtWorker(QThread):
    result = Signal(np.ndarray, float, str)
    error = Signal(str)

    def __init__(self, pattern, size, ck, params, backend):
        super().__init__()
        self._pat = pattern
        self._size = size
        self._ck = ck
        self._params = params
        self._backend = backend
        self._cancel = np.array([False], dtype=np.bool_)

    def cancel(self):
        self._cancel[0] = True

    def run(self):
        try:
            t0 = time.perf_counter()
            gpu = self._backend == "gpu" and HAS_CUPY
            x, y = _coord_cache.get(self._size, gpu=gpu)
            xp = cp if gpu else np

            pat = self._pat
            if pat.needs_backend_dispatch:
                img = pat.fn(x, y, self._size, self._ck, xp,
                             self._backend, self._cancel, **self._params)
            else:
                img = pat.fn(x, y, self._size, self._ck, xp, **self._params)

            elapsed = time.perf_counter() - t0

            if img is None or self._cancel[0]:
                return

            if gpu and isinstance(img, (cp.ndarray,)):
                img = cp.asnumpy(img)

            bname = ("GPU (CuPy)" if gpu
                     else "CPU (Numba)" if self._backend == "numba" and HAS_NUMBA
                     else "CPU (NumPy)")
            self.result.emit(img, elapsed, bname)

        except cp.cuda.OutOfMemoryError:
            self.error.emit("GPU out of memory — switch to CPU backend")
        except Exception as e:
            if not self._cancel[0]:
                self.error.emit(str(e))
        finally:
            if gpu:
                try:
                    cp.get_default_memory_pool().free_all_blocks()
                except Exception:
                    pass


# ═══════════════════════════════════════════════════════════════
#  IMAGE UTILITY
# ═══════════════════════════════════════════════════════════════

def _to_pixmap(arr: NDArray) -> QPixmap:
    arr = np.ascontiguousarray(arr)
    h, w = arr.shape[:2]
    gray = arr.ndim == 2
    fmt = QImage.Format.Format_Grayscale8 if gray else QImage.Format.Format_RGB888
    stride = w if gray else w * 3
    return QPixmap.fromImage(QImage(arr.data, w, h, stride, fmt).copy())


# ═══════════════════════════════════════════════════════════════
#  CUSTOM WIDGETS
# ═══════════════════════════════════════════════════════════════

class ArtCanvas(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self._px = None
        self._loading = False
        self.setMinimumSize(300, 300)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)

    def setArt(self, px):
        self._px, self._loading = px, False
        self.update()

    def setLoading(self, v):
        self._loading = v
        self.update()

    def paintEvent(self, _):
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform)
        p.fillRect(self.rect(), QColor("#0d1117"))
        if self._loading:
            p.setPen(QColor("#8b949e"))
            p.setFont(QFont("Segoe UI", 14))
            p.drawText(self.rect(), Qt.AlignmentFlag.AlignCenter, "⏳  Generating …")
            p.end()
            return
        if self._px and not self._px.isNull():
            m = 16
            sc = self._px.scaled(self.size() - QSize(m, m),
                                 Qt.AspectRatioMode.KeepAspectRatio,
                                 Qt.TransformationMode.SmoothTransformation)
            x = (self.width() - sc.width()) // 2
            y = (self.height() - sc.height()) // 2
            p.setPen(Qt.PenStyle.NoPen)
            for i in range(8, 0, -1):
                p.setBrush(QColor(0, 0, 0, 25 * (9 - i)))
                p.drawRoundedRect(x - i, y - i, sc.width() + 2 * i, sc.height() + 2 * i, 4, 4)
            p.drawPixmap(x, y, sc)
        else:
            p.setPen(QColor("#30363d"))
            p.setFont(QFont("Segoe UI", 14))
            p.drawText(self.rect(), Qt.AlignmentFlag.AlignCenter, "Generate art to begin")
        p.end()


class PalettePreview(QWidget):
    """Horizontal gradient bar showing the current colorization LUT."""
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFixedHeight(22)
        self._px: QPixmap | None = None

    def setLut(self, lut: NDArray | None):
        if lut is None:
            g = np.linspace(0, 255, 256, dtype=np.uint8).reshape(1, -1)
            g = np.stack([g, g, g], axis=2)
        else:
            g = (np.clip(lut, 0, 1) * 255).astype(np.uint8).reshape(1, 256, 3)
        g = np.ascontiguousarray(g)
        qimg = QImage(g.data, 256, 1, 256 * 3, QImage.Format.Format_RGB888).copy()
        self._px = QPixmap.fromImage(qimg)
        self._rescale()
        self.update()

    def _rescale(self):
        if self._px and self.width() > 0:
            self._px = self._px.scaled(self.width(), self.height(),
                                       Qt.AspectRatioMode.IgnoreAspectRatio,
                                       Qt.TransformationMode.SmoothTransformation)

    def resizeEvent(self, _):
        self._rescale()
        self.update()

    def paintEvent(self, _):
        p = QPainter(self)
        if self._px:
            p.drawPixmap(0, 0, self._px)
        else:
            p.fillRect(self.rect(), QColor("#30363d"))
        p.end()


class ParamSlider(QWidget):
    changed = Signal()

    def __init__(self, pdef: ParamDef, parent=None):
        super().__init__(parent)
        self._step = pdef.step
        self._scale = int(round(1.0 / self._step))
        self._dec = len(str(self._step).rstrip("0").split(".")[-1]) if "." in str(self._step) else 0
        self._fmt = f".{self._dec}f" if self._dec else "d"
        lay = QHBoxLayout(self)
        lay.setContentsMargins(0, 2, 0, 2)
        lay.setSpacing(8)
        lbl = QLabel(pdef.label)
        lbl.setFixedWidth(80)
        lbl.setStyleSheet("color:#c9d1d9;")
        lay.addWidget(lbl)
        self.slider = QSlider(Qt.Orientation.Horizontal)
        self.slider.setRange(int(pdef.min_val * self._scale), int(pdef.max_val * self._scale))
        self.slider.setValue(int(pdef.default * self._scale))
        self.slider.valueChanged.connect(self._upd)
        lay.addWidget(self.slider, 1)
        self.vl = QLabel(format(pdef.default, self._fmt))
        self.vl.setFixedWidth(52)
        self.vl.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        self.vl.setStyleSheet("color:#58a6ff;font-weight:bold;")
        lay.addWidget(self.vl)

    def _upd(self, v):
        self.vl.setText(format(v / self._scale, self._fmt))
        self.changed.emit()

    def value(self):
        return self.slider.value() / self._scale

    def setValue(self, v):
        self.slider.blockSignals(True)
        self.slider.setValue(int(v * self._scale))
        self.vl.setText(format(v, self._fmt))
        self.slider.blockSignals(False)


# ═══════════════════════════════════════════════════════════════
#  MAIN WINDOW
# ═══════════════════════════════════════════════════════════════

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Algorithmic Art Studio")
        self.resize(1220, 820)
        self._worker = None
        self._gid = 0
        self._cur: QPixmap | None = None
        self._sliders: dict[str, ParamSlider] = {}
        self._auto = True
        self._timer: QTimer | None = None
        self._build_ui()
        self._shortcuts()
        self._on_pattern(0)
        QTimer.singleShot(150, self.generate)

    # ── build ───────────────────────────────────────────

    def _build_ui(self):
        cw = QWidget()
        self.setCentralWidget(cw)
        root = QHBoxLayout(cw)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)

        sb = QFrame()
        sb.setFixedWidth(310)
        sb.setObjectName("sidebar")
        sl = QVBoxLayout(sb)
        sl.setContentsMargins(16, 20, 16, 16)
        sl.setSpacing(10)

        title = QLabel("🎨  Art Studio")
        title.setObjectName("title")
        sl.addWidget(title)
        sl.addWidget(self._hr())

        # Pattern
        sl.addWidget(self._sec("PATTERN"))
        self.pat_combo = QComboBox()
        self.pat_combo.addItems(p.name for p in PATTERNS)
        self.pat_combo.currentIndexChanged.connect(self._on_pattern)
        sl.addWidget(self.pat_combo)
        self.desc = QLabel("")
        self.desc.setObjectName("desc")
        self.desc.setWordWrap(True)
        sl.addWidget(self.desc)
        sl.addWidget(self._hr())

        # Size
        sl.addWidget(self._sec("IMAGE SIZE"))
        row = QHBoxLayout()
        self.sz_slider = QSlider(Qt.Orientation.Horizontal)
        self.sz_slider.setRange(2, 16)
        self.sz_slider.setValue(8)
        self.sz_slider.valueChanged.connect(self._on_size)
        self.sz_lbl = QLabel("512")
        self.sz_lbl.setFixedWidth(42)
        self.sz_lbl.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        self.sz_lbl.setObjectName("szVal")
        row.addWidget(self.sz_slider, 1)
        row.addWidget(self.sz_lbl)
        sl.addLayout(row)
        sl.addWidget(self._hr())

        # Compute backend
        sl.addWidget(self._sec("COMPUTE"))
        self.be_combo = QComboBox()
        self.be_combo.addItem("Auto")
        if HAS_NUMBA:
            self.be_combo.addItem("CPU (Numba)")
        self.be_combo.addItem("CPU (NumPy)")
        if HAS_CUPY:
            self.be_combo.addItem("GPU (CuPy)")
        self.be_combo.setCurrentIndex(0)
        sl.addWidget(self.be_combo)
        avail = []
        if HAS_CUPY:
            avail.append("CuPy")
        if HAS_NUMBA:
            avail.append("Numba")
        avail.append("NumPy")
        hint = QLabel(f"Available: {', '.join(avail)}")
        hint.setObjectName("desc")
        sl.addWidget(hint)
        sl.addWidget(self._hr())

        # Color mode
        sl.addWidget(self._sec("COLOR MODE"))
        self.cm_combo = QComboBox()
        for label, key in _ALGO_COLORS:
            self.cm_combo.addItem(label, key)
        self.cm_combo.insertSeparator(self.cm_combo.count())
        for label, key in _PAL_COLORS:
            self.cm_combo.addItem(label, key)
        self.cm_combo.currentIndexChanged.connect(self._on_color)
        sl.addWidget(self.cm_combo)
        self.pal_prev = PalettePreview()
        sl.addWidget(self.pal_prev)
        sl.addWidget(self._hr())

        # Parameters
        sl.addWidget(self._sec("PARAMETERS"))
        self.par_box = QWidget()
        self.par_lay = QVBoxLayout(self.par_box)
        self.par_lay.setContentsMargins(0, 0, 0, 0)
        self.par_lay.setSpacing(4)
        sl.addWidget(self.par_box)
        sl.addWidget(self._hr())

        # Auto-generate
        self.auto_btn = QPushButton("⟳  Auto-Generate: ON")
        self.auto_btn.setCheckable(True)
        self.auto_btn.setChecked(True)
        self.auto_btn.setObjectName("autoBtn")
        self.auto_btn.toggled.connect(self._on_auto)
        sl.addWidget(self.auto_btn)

        # Buttons
        bg = QGridLayout()
        bg.setSpacing(8)
        self.gen_btn = QPushButton("⚡ Generate")
        self.gen_btn.setObjectName("genBtn")
        self.gen_btn.clicked.connect(self.generate)
        bg.addWidget(self.gen_btn, 0, 0)
        self.rnd_btn = QPushButton("🎲 Random")
        self.rnd_btn.setObjectName("rndBtn")
        self.rnd_btn.clicked.connect(self.randomize)
        bg.addWidget(self.rnd_btn, 0, 1)
        self.sav_btn = QPushButton("💾 Save")
        self.sav_btn.setObjectName("actBtn")
        self.sav_btn.clicked.connect(self.save_img)
        bg.addWidget(self.sav_btn, 1, 0)
        self.cpy_btn = QPushButton("📋 Copy")
        self.cpy_btn.setObjectName("actBtn")
        self.cpy_btn.clicked.connect(self.copy_img)
        bg.addWidget(self.cpy_btn, 1, 1)
        sl.addLayout(bg)
        sl.addStretch()
        root.addWidget(sb)

        # Canvas
        cf = QFrame()
        cf.setObjectName("canvasFrame")
        cl = QVBoxLayout(cf)
        cl.setContentsMargins(16, 16, 16, 16)
        self.canvas = ArtCanvas()
        cl.addWidget(self.canvas)
        root.addWidget(cf, 1)

        # Status
        self.sbar = QStatusBar()
        self.setStatusBar(self.sbar)
        self.stlbl = QLabel("Ready")
        self.sbar.addWidget(self.stlbl, 1)

        # Init palette preview
        self._on_color(0)

    @staticmethod
    def _hr():
        f = QFrame()
        f.setFrameShape(QFrame.Shape.HLine)
        f.setObjectName("sep")
        return f

    @staticmethod
    def _sec(t):
        l = QLabel(t)
        l.setObjectName("secLbl")
        return l

    def _shortcuts(self):
        QShortcut(QKeySequence("Ctrl+G"), self, self.generate)
        QShortcut(QKeySequence("Ctrl+S"), self, self.save_img)
        QShortcut(QKeySequence("Ctrl+R"), self, self.randomize)
        QShortcut(QKeySequence("Ctrl+Shift+C"), self, self.copy_img)

    # ── backend resolve ─────────────────────────────────

    def _backend(self) -> str:
        txt = self.be_combo.currentText()
        if "CuPy" in txt:
            return "gpu"
        if "Numba" in txt:
            return "numba"
        if "NumPy" in txt:
            return "numpy"
        # Auto
        if HAS_CUPY:
            return "gpu"
        if HAS_NUMBA:
            return "numba"
        return "numpy"

    # ── handlers ────────────────────────────────────────

    def _on_pattern(self, idx):
        pat = _PAT[self.pat_combo.currentText()]
        self.desc.setText(pat.description)
        while self.par_lay.count():
            w = self.par_lay.takeAt(0).widget()
            if w:
                w.deleteLater()
        self._sliders.clear()
        for pd in pat.params:
            s = ParamSlider(pd)
            s.changed.connect(self._sched)
            self.par_lay.addWidget(s)
            self._sliders[pd.key] = s
        self._sched()

    def _on_size(self, v):
        self.sz_lbl.setText(str(v * 64))
        self._sched()

    def _on_color(self, idx):
        key = self.cm_combo.itemData(idx)
        if key:
            self.pal_prev.setLut(_colorizer.get_preview_lut(key))
        self._sched()

    def _on_auto(self, on):
        self._auto = on
        self.auto_btn.setText(f"⟳  Auto-Generate: {'ON' if on else 'OFF'}")
        if on:
            self._sched()

    def _sched(self):
        if not self._auto:
            return
        if self._timer is not None:
            self._timer.stop()
        self._timer = QTimer(self)
        self._timer.setSingleShot(True)
        self._timer.setInterval(60)
        self._timer.timeout.connect(self.generate)
        self._timer.start()

    # ── generation ──────────────────────────────────────

    def generate(self):
        if self._worker is not None and self._worker.isRunning():
            self._worker.cancel()
            self._worker.wait(500)

        self._gid += 1
        gid = self._gid

        pat = _PAT[self.pat_combo.currentText()]
        size = self.sz_slider.value() * 64
        ck = self.cm_combo.currentData() or self.cm_combo.currentText().lower().replace(" ", "_")
        params = {k: s.value() for k, s in self._sliders.items()}
        backend = self._backend()

        self.canvas.setLoading(True)
        self.gen_btn.setEnabled(False)
        self.stlbl.setText("Generating …")

        self._worker = ArtWorker(pat, size, ck, params, backend)
        self._worker.result.connect(lambda img, t, b, g=gid: self._done(img, t, b, g))
        self._worker.error.connect(lambda msg, g=gid: self._err(msg, g))
        self._worker.start()

    def _done(self, img, elapsed, backend_name, gid):
        if gid != self._gid:
            return
        self._cur = _to_pixmap(img)
        self.canvas.setArt(self._cur)
        self.gen_btn.setEnabled(True)
        h, w = img.shape[:2]
        self.stlbl.setText(
            f"✓  {self.pat_combo.currentText()}   |   {w}×{h}   |   "
            f"{self.cm_combo.currentText()}   |   {elapsed:.3f}s   |   {backend_name}"
        )

    def _err(self, msg, gid):
        if gid != self._gid:
            return
        self.canvas.setLoading(False)
        self.gen_btn.setEnabled(True)
        self.stlbl.setText(f"✗  Error: {msg}")

    # ── actions ─────────────────────────────────────────

    def randomize(self):
        pat = random.choice(PATTERNS)
        self.pat_combo.blockSignals(True)
        self.pat_combo.setCurrentText(pat.name)
        self.pat_combo.blockSignals(False)
        self._on_pattern(list(_PAT.keys()).index(pat.name))

        for pd in pat.params:
            s = self._sliders.get(pd.key)
            if s:
                if pd.is_float:
                    val = round(random.uniform(pd.min_val, pd.max_val) / pd.step) * pd.step
                else:
                    val = random.randint(int(pd.min_val), int(pd.max_val))
                s.setValue(val)

        self.cm_combo.blockSignals(True)
        self.cm_combo.setCurrentIndex(random.randint(0, self.cm_combo.count() - 1))
        self.cm_combo.blockSignals(False)
        self._on_color(self.cm_combo.currentIndex())

        self.sz_slider.blockSignals(True)
        self.sz_slider.setValue(random.randint(2, 16))
        self.sz_slider.blockSignals(False)
        self.sz_lbl.setText(str(self.sz_slider.value() * 64))

        self.generate()

    def save_img(self):
        if not self._cur:
            self.stlbl.setText("Nothing to save — generate art first.")
            return
        path, _ = QFileDialog.getSaveFileName(
            self, "Save Art", "art.png",
            "PNG (*.png);;JPEG (*.jpg);;BMP (*.bmp)",
        )
        if path:
            self._cur.save(path)
            self.stlbl.setText(f"Saved → {path}")

    def copy_img(self):
        if not self._cur:
            self.stlbl.setText("Nothing to copy — generate art first.")
            return
        QApplication.clipboard().setPixmap(self._cur)
        self.stlbl.setText("Copied to clipboard ✓")


# ═══════════════════════════════════════════════════════════════
#  STYLESHEET
# ═══════════════════════════════════════════════════════════════

QSS = """
QMainWindow { background:#0d1117; }
#sidebar { background:#161b22; border-right:1px solid #30363d; }
#canvasFrame { background:#0d1117; }
#title { font-size:20px; font-weight:bold; color:#f0f6fc; padding:4px 0; }
#desc { font-size:11px; color:#8b949e; padding:2px 0; line-height:1.4; }
#secLbl { font-size:11px; font-weight:bold; color:#8b949e;
          letter-spacing:1px; padding-top:4px; }
#sep { color:#21262d; max-height:1px; }
#szVal { color:#58a6ff; font-weight:bold; font-size:13px; }

QComboBox {
    background:#0d1117; border:1px solid #30363d; border-radius:6px;
    padding:6px 12px; color:#c9d1d9; font-size:13px; min-height:20px; }
QComboBox:hover { border-color:#58a6ff; }
QComboBox::drop-down { border:none; width:24px; }
QComboBox::down-arrow {
    image:none; border-left:5px solid transparent;
    border-right:5px solid transparent;
    border-top:6px solid #8b949e; margin-right:8px; }
QComboBox QAbstractItemView {
    background:#161b22; border:1px solid #30363d; color:#c9d1d9;
    selection-background-color:#1f6feb; selection-color:#fff;
    outline:none; padding:4px; }

QSlider::groove:horizontal {
    height:4px; background:#30363d; border-radius:2px; }
QSlider::handle:horizontal {
    background:#58a6ff; width:16px; height:16px; margin:-6px 0;
    border-radius:8px; border:2px solid #0d1117; }
QSlider::handle:horizontal:hover { background:#79c0ff; }
QSlider::sub-page:horizontal { background:#1f6feb; border-radius:2px; }

QPushButton {
    background:#21262d; border:1px solid #30363d; border-radius:6px;
    padding:8px 12px; color:#c9d1d9; font-size:12px; font-weight:bold; }
QPushButton:hover { background:#30363d; border-color:#8b949e; }
QPushButton:pressed { background:#0d1117; }
QPushButton:disabled { color:#484f58; border-color:#21262d; }

#genBtn { background:#238636; border-color:#2ea043; color:#fff; }
#genBtn:hover { background:#2ea043; }
#genBtn:disabled { background:#1a3a2a; border-color:#1a3a2a; color:#484f58; }
#rndBtn { background:#8957e5; border-color:#a371f7; color:#fff; }
#rndBtn:hover { background:#a371f7; }
#actBtn { background:#1f6feb; border-color:#388bfd; color:#fff; }
#actBtn:hover { background:#388bfd; }

#autoBtn {
    background:transparent; border:1px solid #30363d;
    text-align:left; padding:8px 12px; font-size:12px; }
#autoBtn:checked { border-color:#238636; color:#3fb950; }

QStatusBar {
    background:#161b22; border-top:1px solid #30363d;
    color:#8b949e; font-size:12px; padding:2px 8px; }
"""


# ═══════════════════════════════════════════════════════════════
#  ENTRY POINT
# ═══════════════════════════════════════════════════════════════

if __name__ == "__main__":
    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    app.setStyleSheet(QSS)
    win = MainWindow()
    win.show()
    sys.exit(app.exec())