#!/usr/bin/env python3
"""Algorithmic Art Studio — single-file edition.

A PySide6 desktop application that generates algorithmic art using multiple
compute backends (NumPy, Numba JIT, CuPy GPU).  Supports 11 pattern types,
26 color palettes, undo/redo, fullscreen, and real-time parameter tweaking.
"""

from __future__ import annotations

import sys
import time
import random
import math
import logging
import numpy as np
from numpy.typing import NDArray
from dataclasses import dataclass
from typing import Callable

from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QComboBox, QSlider, QPushButton, QFrame,
    QFileDialog, QStatusBar, QSizePolicy, QGridLayout,
    QScrollArea, QMessageBox,
)
from PySide6.QtGui import (QPixmap, QImage, QFont, QColor, QPainter,
    QWheelEvent, QMouseEvent, QKeySequence, QShortcut)
from PySide6.QtCore import Qt, QSize, QThread, Signal, QTimer, QPointF

# ── Logging Setup ──────────────────────────────────────────────────
# Configures a file handler with mode='w' (overwrites on each startup)
# and a console handler for real-time feedback.
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

file_handler = logging.FileHandler("art_studio.log", mode='w', encoding='utf-8')
file_handler.setFormatter(logging.Formatter(
    '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
))
logger.addHandler(file_handler)

console_handler = logging.StreamHandler(sys.stdout)
console_handler.setFormatter(logging.Formatter(
    '%(levelname)s: %(message)s'
))
logger.addHandler(console_handler)


# ═══════════════════════════════════════════════════════════════════════
#  § 1. CONFIG — Data structures, palettes, constants
# ═══════════════════════════════════════════════════════════════════════

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


# ── Palette stop helper ────────────────────────────────────────────

def _p(*rows):
    return np.array(rows, dtype=np.float32)


PALETTES: dict[str, NDArray] = {
    # --- Scientific ---
    "inferno": _p(
        (0.00, 0.001, 0.000, 0.014), (0.13, 0.106, 0.036, 0.260),
        (0.25, 0.258, 0.023, 0.463), (0.38, 0.506, 0.016, 0.506),
        (0.50, 0.735, 0.106, 0.370), (0.63, 0.902, 0.301, 0.175),
        (0.75, 0.973, 0.557, 0.067), (0.88, 0.988, 0.798, 0.183),
        (1.00, 0.988, 0.998, 0.645)),
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
    # --- Classic ---
    "hot": _p(
        (0.00, 0.0, 0.0, 0.0), (0.35, 0.8, 0.0, 0.0),
        (0.67, 1.0, 0.6, 0.0), (1.00, 1.0, 1.0, 1.0)),
    "cool": _p((0.00, 0.0, 1.0, 1.0), (1.00, 1.0, 0.0, 1.0)),
    "bone": _p(
        (0.00, 0.0, 0.0, 0.0), (0.25, 0.253, 0.253, 0.376),
        (0.50, 0.569, 0.569, 0.694), (0.75, 0.855, 0.855, 0.933),
        (1.00, 1.0, 1.0, 1.0)),
    "spring": _p((0.00, 1.0, 0.0, 1.0), (1.00, 1.0, 1.0, 0.0)),
    "summer": _p((0.00, 0.0, 0.5, 0.4), (1.00, 1.0, 1.0, 0.4)),
    "autumn": _p((0.00, 1.0, 0.0, 0.0), (1.00, 1.0, 1.0, 0.0)),
    "winter": _p((0.00, 0.0, 0.0, 1.0), (1.00, 0.0, 1.0, 0.5)),
    # --- Creative ---
    "ocean": _p(
        (0.00, 0.004, 0.020, 0.110), (0.20, 0.000, 0.075, 0.290),
        (0.40, 0.004, 0.180, 0.467), (0.60, 0.020, 0.357, 0.627),
        (0.80, 0.133, 0.580, 0.788), (1.00, 0.494, 0.851, 0.973)),
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
    # --- Themed ---
    "ember": _p(
        (0.00, 0.05, 0.00, 0.00), (0.25, 0.30, 0.02, 0.00),
        (0.50, 0.70, 0.15, 0.00), (0.75, 0.95, 0.50, 0.05),
        (1.00, 1.00, 0.95, 0.40)),
    "arctic": _p(
        (0.00, 0.02, 0.02, 0.10), (0.25, 0.05, 0.10, 0.30),
        (0.50, 0.15, 0.35, 0.60), (0.75, 0.50, 0.70, 0.85),
        (1.00, 0.92, 0.96, 1.00)),
    "forest": _p(
        (0.00, 0.02, 0.06, 0.02), (0.25, 0.05, 0.18, 0.05),
        (0.50, 0.12, 0.42, 0.08), (0.75, 0.32, 0.65, 0.18),
        (1.00, 0.62, 0.92, 0.38)),
    "sunset": _p(
        (0.00, 0.10, 0.02, 0.22), (0.25, 0.42, 0.05, 0.32),
        (0.50, 0.82, 0.20, 0.15), (0.75, 0.96, 0.52, 0.10),
        (1.00, 1.00, 0.88, 0.42)),
    "rose": _p(
        (0.00, 0.15, 0.02, 0.12), (0.25, 0.45, 0.05, 0.28),
        (0.50, 0.82, 0.18, 0.52), (0.75, 0.96, 0.48, 0.78),
        (1.00, 1.00, 0.88, 0.96)),
    "mint": _p(
        (0.00, 0.02, 0.06, 0.06), (0.25, 0.04, 0.18, 0.16),
        (0.50, 0.12, 0.38, 0.32), (0.75, 0.32, 0.68, 0.58),
        (1.00, 0.72, 0.96, 0.88)),
    # --- Pastel (stops-based, used by ALGO_COLORS "Pastel") ---
    "pastel": _p(
        (0.00, 0.96, 0.84, 0.91), (0.17, 0.78, 0.86, 0.96),
        (0.33, 0.82, 0.93, 0.80), (0.50, 0.96, 0.96, 0.78),
        (0.67, 0.96, 0.84, 0.78), (0.83, 0.93, 0.80, 0.90),
        (1.00, 0.88, 0.84, 0.96)),
}

ALGO_COLORS: list[tuple[str, str]] = [
    ("Grayscale", "grayscale"),
    ("Channel Shift", "channel_shift"),
    ("Neon", "neon"),
    ("Spectrum", "spectrum"),
    ("Pastel", "pastel"),
]

PAL_COLORS: list[tuple[str, str]] = [
    ("Inferno", "inferno"), ("Viridis", "viridis"), ("Plasma", "plasma"),
    ("Magma", "magma"), ("Cividis", "cividis"), ("Turbo", "turbo"),
    ("Hot", "hot"), ("Cool", "cool"), ("Bone", "bone"),
    ("Spring", "spring"), ("Summer", "summer"), ("Autumn", "autumn"),
    ("Winter", "winter"), ("Ocean", "ocean"), ("Cubehelix", "cubehelix"),
    ("Twilight", "twilight"),
    ("Ember", "ember"), ("Arctic", "arctic"), ("Forest", "forest"),
    ("Sunset", "sunset"), ("Rose", "rose"), ("Mint", "mint"),
]


# ═══════════════════════════════════════════════════════════════════════
#  § 2. BACKENDS — Numba / CuPy detection, JIT kernels, dispatch
# ═══════════════════════════════════════════════════════════════════════

# ── Optional dependency detection ──────────────────────────────────

try:
    from numba import njit, prange as _numba_prange
    HAS_NUMBA = True
except ImportError:
    HAS_NUMBA = False
    _numba_prange = range  # type: ignore[assignment]

try:
    import cupy as cp
    _cp_sanity = cp.zeros(1, dtype=np.float32)
    del _cp_sanity
    HAS_CUPY = True
except Exception:
    cp = None  # type: ignore[assignment]
    HAS_CUPY = False

_numba_warmed = False


def _ensure_numba_warmup():
    """Deferred Numba JIT warm-up — runs only once on first use."""
    global _numba_warmed
    if _numba_warmed or not HAS_NUMBA:
        return
    logger.info("Warming up Numba JIT kernels...")
    _tiny = np.zeros((2, 2), dtype=np.float64)
    _tiny_cancel = np.array([False], dtype=np.bool_)
    _mandelbrot_numba(_tiny, _tiny, 2, _tiny_cancel)
    _julia_numba(_tiny, _tiny, -0.7, 0.27, 2, _tiny_cancel)
    _burning_ship_numba(_tiny, _tiny, 2, _tiny_cancel)
    _box_blur_numba(np.zeros((4, 4), dtype=np.float32), 1)
    del _tiny, _tiny_cancel
    _numba_warmed = True
    logger.info("Numba warm-up complete.")


def _numba_jit(fn):
    """Decorator: apply Numba JIT when available, else pass through."""
    if HAS_NUMBA:
        return njit(parallel=True, fastmath=True, cache=True)(fn)
    return fn


# ── Coordinate cache (LRU, GPU-aware) ──────────────────────────────

class CoordCache:
    """Caches (x, y) index grids keyed by (size, on_gpu)."""

    def __init__(self, cap: int = 8) -> None:
        self._data: dict[tuple[int, bool], tuple] = {}
        self._order: list[tuple[int, bool]] = []
        self._cap = cap

    def get(self, size: int, gpu: bool = False):
        key = (size, gpu)
        if key in self._data:
            self._order.remove(key)
            self._order.append(key)
            return self._data[key]
        if len(self._data) >= self._cap:
            del self._data[self._order.pop(0)]
        x, y = np.indices((size, size), dtype=np.float64)
        if gpu and HAS_CUPY:
            x, y = cp.asarray(x), cp.asarray(y)
        self._data[key] = (x, y)
        self._order.append(key)
        return x, y


_coord_cache = CoordCache()


# ── Numba-accelerated kernels ──────────────────────────────────────

@_numba_jit
def _box_blur_numba(arr: NDArray, passes: int) -> NDArray:
    h, w = arr.shape
    out = arr.copy()
    for _ in range(passes):
        tmp = np.empty_like(out)
        for i in _numba_prange(h):
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
    for i in _numba_prange(h):
        for j in range(w):
            if cancel[0]:
                div[i, j] = 0.0
                continue
            cr, ci = cx[i, j], cy[i, j]
            zr, zi = 0.0, 0.0
            n = 0
            while n < max_iter:
                zr2, zi2 = zr * zr, zi * zi
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


@_numba_jit
def _julia_numba(zr_arr: NDArray, zi_arr: NDArray,
                 cr: float, ci: float,
                 max_iter: int, cancel: NDArray) -> NDArray:
    h, w = zr_arr.shape
    div = np.empty((h, w), dtype=np.float64)
    LN2 = 0.6931471805599453
    for i in _numba_prange(h):
        for j in range(w):
            if cancel[0]:
                div[i, j] = 0.0
                continue
            zr, zi = zr_arr[i, j], zi_arr[i, j]
            n = 0
            while n < max_iter:
                zr2, zi2 = zr * zr, zi * zi
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


@_numba_jit
def _burning_ship_numba(cx: NDArray, cy: NDArray,
                        max_iter: int, cancel: NDArray) -> NDArray:
    """Burning Ship fractal — uses abs(z) at each iteration."""
    h, w = cx.shape
    div = np.empty((h, w), dtype=np.float64)
    LN2 = 0.6931471805599453
    for i in _numba_prange(h):
        for j in range(w):
            if cancel[0]:
                div[i, j] = 0.0
                continue
            cr, ci = cx[i, j], cy[i, j]
            zr, zi = 0.0, 0.0
            n = 0
            while n < max_iter:
                zr2, zi2 = zr * zr, zi * zi
                if zr2 + zi2 > 4.0:
                    break
                zi = 2.0 * abs(zr) * abs(zi) + ci
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


# ── CuPy-accelerated kernels ───────────────────────────────────────

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
        smooth = i + 1.0 - cp.log2(cp.maximum(cp.log2(safe_abs), 1e-3))
        div = cp.where(hit, smooth, div)
        esc |= hit
        z = cp.where(esc, z, z_new)
    return cp.asnumpy(div)


def _julia_gpu(zr, zi, cr: float, ci: float,
               max_iter: int, cancel: NDArray):
    z = cp.asarray(zr) + 1j * cp.asarray(zi)
    c = complex(cr, ci)
    div = cp.full(z.shape, float(max_iter), dtype=cp.float64)
    esc = cp.zeros(z.shape, dtype=cp.bool_)
    for i in range(max_iter):
        if cancel[0]:
            return None
        z_new = z * z + c
        hit = ~esc & (cp.abs(z_new) > 2.0)
        safe_abs = cp.maximum(cp.abs(z_new), 1.0)
        smooth = i + 1.0 - cp.log2(cp.maximum(cp.log2(safe_abs), 1e-3))
        div = cp.where(hit, smooth, div)
        esc |= hit
        z = cp.where(esc, z, z_new)
    return cp.asnumpy(div)


def _burning_ship_gpu(cx, cy, max_iter: int, cancel: NDArray):
    """Burning Ship on GPU — uses |Re(z)| and |Im(z)| component-wise."""
    c = cp.asarray(cx) + 1j * cp.asarray(cy)
    z = cp.zeros(c.shape, dtype=cp.complex128)
    div = cp.full(c.shape, float(max_iter), dtype=cp.float64)
    esc = cp.zeros(c.shape, dtype=cp.bool_)
    for i in range(max_iter):
        if cancel[0]:
            return None
        zr_abs = cp.abs(z.real)
        zi_abs = cp.abs(z.imag)
        z_new = cp.where(esc, z, (zr_abs + 1j * zi_abs) ** 2 + c)
        hit = ~esc & (cp.abs(z_new) > 2.0)
        safe_abs = cp.maximum(cp.abs(z_new), 1.0)
        smooth = i + 1.0 - cp.log2(cp.maximum(cp.log2(safe_abs), 1e-3))
        div = cp.where(hit, smooth, div)
        esc |= hit
        z = z_new
    return cp.asnumpy(div)


# ── NumPy fallback kernels ─────────────────────────────────────────

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
            div[hit] = i + 1.0 - np.log2(np.maximum(np.log2(safe), 1e-3))
            esc[hit] = True
    return div


def _julia_numpy(zr, zi, cr: float, ci: float,
                 max_iter: int, cancel: NDArray):
    z = zr + 1j * zi
    c = complex(cr, ci)
    div = np.full(z.shape, float(max_iter), dtype=np.float64)
    esc = np.zeros(z.shape, dtype=bool)
    for i in range(max_iter):
        if cancel[0]:
            return None
        active = ~esc & (np.abs(z) <= 2.0)
        if not np.any(active):
            break
        z[active] = z[active] ** 2 + c
        hit = active & (np.abs(z) > 2.0)
        if np.any(hit):
            safe = np.maximum(np.abs(z[hit]), 1.0)
            div[hit] = i + 1.0 - np.log2(np.maximum(np.log2(safe), 1e-3))
            esc[hit] = True
    return div


def _burning_ship_numpy(cx, cy, max_iter: int, cancel: NDArray):
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
        z_a = z[active]
        z[active] = (np.abs(z_a.real) + 1j * np.abs(z_a.imag)) ** 2 + c[active]
        hit = active & (np.abs(z) > 2.0)
        if np.any(hit):
            safe = np.maximum(np.abs(z[hit]), 1.0)
            div[hit] = i + 1.0 - np.log2(np.maximum(np.log2(safe), 1e-3))
            esc[hit] = True
    return div


# ── Unified dispatch functions ─────────────────────────────────────

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


def _dispatch_julia(zr, zi, cr, ci, max_iter, cancel, backend):
    if backend == "gpu" and HAS_CUPY:
        return _julia_gpu(zr, zi, cr, ci, max_iter, cancel)
    if backend == "numba" and HAS_NUMBA:
        return _julia_numba(zr, zi, cr, ci, max_iter, cancel)
    return _julia_numpy(zr, zi, cr, ci, max_iter, cancel)


def _dispatch_burning_ship(cx, cy, max_iter, cancel, backend):
    if backend == "gpu" and HAS_CUPY:
        return _burning_ship_gpu(cx, cy, max_iter, cancel)
    if backend == "numba" and HAS_NUMBA:
        return _burning_ship_numba(cx, cy, max_iter, cancel)
    return _burning_ship_numpy(cx, cy, max_iter, cancel)


# ═══════════════════════════════════════════════════════════════════════
#  § 3. COLORIZER — LUT-based color mapping with GPU awareness
# ═══════════════════════════════════════════════════════════════════════

class Colorizer:
    """Precomputed 256-entry LUT colorizer; caches GPU copies."""

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
    def _hsv_lut(h, s, v) -> NDArray:
        h6 = (h % 1.0) * 6.0
        sec = np.floor(h6).astype(np.int32) % 6
        f = h6 - np.floor(h6)
        p = v * (1.0 - s)
        q = v * (1.0 - s * f)
        t = v * (1.0 - s * (1.0 - f))
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

    def _build_all(self) -> None:
        t = np.linspace(0, 1, 256, dtype=np.float32)
        ones = np.ones(256, dtype=np.float32)
        self._cpu["spectrum"] = self._hsv_lut(
            t, np.full(256, 0.85, np.float32),
            np.clip(t * 1.4 + 0.15, 0, 1))
        self._cpu["neon"] = self._hsv_lut(
            (t * 2.5 + 0.55) % 1, ones,
            np.clip(t * 1.8 + 0.25, 0, 1))
        for name, stops in PALETTES.items():
            self._cpu[name] = self._from_stops(stops)

    def get_preview_lut(self, key: str) -> NDArray | None:
        return self._cpu.get(key)

    def colorize(self, base, key: str, xp=np):
        """Map a 2-D value array to an RGB uint8 image via LUT."""
        if key == "grayscale":
            return xp.clip(base, 0, 255).astype(xp.uint8)
        if key == "channel_shift":
            r = base
            g = xp.roll(base, 15, axis=0)
            b = xp.roll(base, -15, axis=1)
            return xp.clip(
                xp.stack([r, g, b], axis=2), 0, 255
            ).astype(xp.uint8)
        lut = self._lut(key, xp)
        idx = xp.clip(base, 0, 255).astype(xp.uint8)
        return (xp.clip(lut[idx], 0.0, 1.0) * 255).astype(xp.uint8)

    def _lut(self, key: str, xp):
        if xp is np:
            return self._cpu[key]
        if key not in self._gpu:
            self._gpu[key] = cp.asarray(self._cpu[key])
        return self._gpu[key]


colorizer = Colorizer()


# ═══════════════════════════════════════════════════════════════════════
#  § 4. ALGORITHMS — Pattern functions and registry
# ═══════════════════════════════════════════════════════════════════════

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
    # BUG FIX: Cast to int before bitwise operations
    xs = (x // cell_size).astype(np.int64)
    ys = (y // cell_size).astype(np.int64)
    v = np.bitwise_xor(xs & 1, ys & 1).astype(np.float64) * 255
    return colorizer.colorize(v, ck, np)


def mandelbrot_art(x, y, size, ck, xp, backend, cancel,
                   max_iter=255, center_x=-0.5, center_y=0.0, zoom=1.0):
    _ensure_numba_warmup()
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
    _ensure_numba_warmup()
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


def burning_ship_art(x, y, size, ck, xp, backend, cancel,
                     max_iter=255, center_x=-0.4, center_y=-0.6, zoom=1.0):
    """Burning Ship fractal — a stunning variant with 'ship' structures."""
    _ensure_numba_warmup()
    inv_z = 1.0 / zoom
    xr, yr = 3.5 * inv_z, 2.5 * inv_z
    inv_s = 1.0 / size
    cx = x * inv_s * xr + center_x - xr * 0.5
    cy = y * inv_s * yr + center_y - yr * 0.5
    div = _dispatch_burning_ship(cx, cy, max_iter, cancel, backend)
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
    v = ((x // tile_size + y // tile_size).astype(np.int64) & 1) * 255.0
    return colorizer.colorize(v, ck, xp)


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


def sierpinski_art(x, y, size, ck, xp, scale=6):
    """Sierpinski triangle — point is inside if (x & y) == 0.

    The scale parameter controls the recursion depth: the pattern
    tiles with period 2^scale, creating finer detail at higher values.
    """
    xs = x.astype(np.int64)
    ys = y.astype(np.int64)
    mask = (1 << scale) - 1
    v = (((xs & ys) & mask) == 0).astype(np.float64) * 255
    return colorizer.colorize(v, ck, np)


# ── Julia presets for good randomization results ────────────────────

JULIA_PRESETS: list[tuple[float, float]] = [
    (-0.7, 0.27015), (-0.8, 0.156), (0.285, 0.01),
    (-0.4, 0.6), (0.355, 0.355), (-0.54, 0.54),
    (0.37, 0.1), (-0.12, 0.75), (-0.75, 0.11),
    (0.28, 0.008), (-0.62, 0.42), (-0.1, 0.65),
]

# ── Burning Ship interesting regions ────────────────────────────────

BURNING_SHIP_PRESETS: list[tuple[float, float, float]] = [
    (-0.4, -0.6, 1.0),        # Full view
    (-1.755, -0.028, 40.0),   # Antenna detail
    (-1.762, -0.028, 200.0),  # Deep antenna zoom
    (-0.45, -0.52, 5.0),      # Ship hull
    (-1.78, -0.02, 500.0),    # Mini-ship
]


# ── Pattern registry ───────────────────────────────────────────────

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
                ParamDef("zoom", "Zoom", True, 0.5, 500, 1, 0.5),
                ParamDef("center_x", "Center X", True, -2.5, 1.5, -0.5, 0.05),
                ParamDef("center_y", "Center Y", True, -1.5, 1.5, 0, 0.05)),
               needs_backend_dispatch=True),
    PatternDef("Julia Set", julia_art,
               "Julia set fractal — explore beautiful constant variations.",
               (ParamDef("max_iter", "Iterations", False, 32, 512, 255, 1),
                ParamDef("julia_r", "C (real)", True, -1.5, 1.5, -0.7, 0.01),
                ParamDef("julia_i", "C (imag)", True, -1.5, 1.5, 0.27015, 0.01),
                ParamDef("zoom", "Zoom", True, 0.5, 200, 1, 0.5)),
               needs_backend_dispatch=True),
    PatternDef("Burning Ship", burning_ship_art,
               "Burning Ship fractal — stunning 'ship' structures via abs(z).",
               (ParamDef("max_iter", "Iterations", False, 32, 512, 255, 1),
                ParamDef("zoom", "Zoom", True, 0.5, 1000, 1, 0.5),
                ParamDef("center_x", "Center X", True, -2.5, 1.0, -0.4, 0.05),
                ParamDef("center_y", "Center Y", True, -2.0, 0.5, -0.6, 0.05)),
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
    PatternDef("Sierpinski", sierpinski_art,
               "Sierpinski triangle via bitwise coordinate decomposition.",
               (ParamDef("scale", "Depth", False, 1, 12, 6, 1),)),
)

PATTERN_MAP: dict[str, PatternDef] = {p.name: p for p in PATTERNS}


# ═══════════════════════════════════════════════════════════════════════
#  § 5. WIDGETS — Canvas, palette preview, param slider, worker
# ═══════════════════════════════════════════════════════════════════════

def to_pixmap(arr: NDArray) -> QPixmap:
    """Convert an HxW (grayscale) or HxWx3 (RGB) uint8 array to QPixmap."""
    arr = np.ascontiguousarray(arr)
    h, w = arr.shape[:2]
    gray = arr.ndim == 2
    fmt = QImage.Format.Format_Grayscale8 if gray else QImage.Format.Format_RGB888
    stride = w if gray else w * 3
    return QPixmap.fromImage(QImage(arr.data, w, h, stride, fmt).copy())


class ArtCanvas(QWidget):
    """Central image display with drop-shadow, zoom, and pan."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self._px: QPixmap | None = None
        self._loading = False
        self._zoom = 1.0
        self._pan = QPointF(0, 0)
        self._drag_start: QPointF | None = None
        self._pan_start: QPointF | None = None
        self.setMinimumSize(300, 300)
        self.setSizePolicy(QSizePolicy.Policy.Expanding,
                           QSizePolicy.Policy.Expanding)
        self.setCursor(Qt.CursorShape.OpenHandCursor)
        self.setMouseTracking(True)

    def setArt(self, px: QPixmap):
        self._px = px
        self._loading = False
        self._zoom = 1.0
        self._pan = QPointF(0, 0)
        self.update()

    def setLoading(self, v: bool):
        self._loading = v
        self.update()

    def clearArt(self):
        self._px = None
        self._loading = False
        self.update()

    def resetView(self):
        self._zoom = 1.0
        self._pan = QPointF(0, 0)
        self.update()

    def _scaled_pixmap(self) -> QPixmap | None:
        if not self._px or self._px.isNull():
            return None
        target = self.size() - QSize(32, 32)
        if target.width() < 10 or target.height() < 10:
            return None
        base = self._px.scaled(
            target, Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation)
        if abs(self._zoom - 1.0) < 0.01:
            return base
        scaled_size = QSize(
            int(base.width() * self._zoom),
            int(base.height() * self._zoom))
        return base.scaled(
            scaled_size, Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation)

    def paintEvent(self, _) -> None:
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform)
        p.fillRect(self.rect(), QColor("#0d1117"))

        if self._loading:
            p.setPen(QColor("#8b949e"))
            p.setFont(QFont("Segoe UI", 14))
            p.drawText(self.rect(), Qt.AlignmentFlag.AlignCenter,
                       "Generating ...")
            p.end()
            return

        sc = self._scaled_pixmap()
        if sc and not sc.isNull():
            x = (self.width() - sc.width()) // 2 + int(self._pan.x())
            y = (self.height() - sc.height()) // 2 + int(self._pan.y())
            # Drop shadow
            p.setPen(Qt.PenStyle.NoPen)
            for i in range(8, 0, -1):
                p.setBrush(QColor(0, 0, 0, 25 * (9 - i)))
                p.drawRoundedRect(x - i, y - i,
                                  sc.width() + 2 * i, sc.height() + 2 * i, 4, 4)
            p.drawPixmap(x, y, sc)
        else:
            p.setPen(QColor("#30363d"))
            p.setFont(QFont("Segoe UI", 14))
            p.drawText(self.rect(), Qt.AlignmentFlag.AlignCenter,
                       "Generate art to begin")
        p.end()

    def wheelEvent(self, event: QWheelEvent) -> None:
        if not self._px or self._px.isNull():
            return
        delta = event.angleDelta().y()
        factor = 1.15 if delta > 0 else 1.0 / 1.15
        self._zoom = max(0.1, min(20.0, self._zoom * factor))
        self.update()

    def mousePressEvent(self, event: QMouseEvent) -> None:
        if event.button() == Qt.MouseButton.LeftButton:
            self._drag_start = event.position()
            self._pan_start = QPointF(self._pan)
            self.setCursor(Qt.CursorShape.ClosedHandCursor)

    def mouseMoveEvent(self, event: QMouseEvent) -> None:
        if self._drag_start is not None and self._pan_start is not None:
            delta = event.position() - self._drag_start
            self._pan = QPointF(
                self._pan_start.x() + delta.x(),
                self._pan_start.y() + delta.y())
            self.update()

    def mouseReleaseEvent(self, event: QMouseEvent) -> None:
        if event.button() == Qt.MouseButton.LeftButton:
            self._drag_start = None
            self._pan_start = None
            self.setCursor(Qt.CursorShape.OpenHandCursor)

    def mouseDoubleClickEvent(self, _) -> None:
        """Double-click resets zoom and pan."""
        self.resetView()


class PalettePreview(QWidget):
    """Horizontal gradient bar showing the active colorization LUT."""

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
        qimg = QImage(g.data, 256, 1, 256 * 3,
                      QImage.Format.Format_RGB888).copy()
        self._px = QPixmap.fromImage(qimg)
        self._rescale()
        self.update()

    def _rescale(self):
        if self._px and self.width() > 0:
            self._px = self._px.scaled(
                self.width(), self.height(),
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
    """Label + slider + value display for a single pattern parameter."""

    changed = Signal()
    doubleClicked = Signal()

    def __init__(self, pdef: ParamDef, parent=None):
        super().__init__(parent)
        self._pdef = pdef
        self._is_float = pdef.is_float
        self._scale = max(1, int(round(1.0 / pdef.step)))

        if self._is_float:
            step_str = str(pdef.step)
            self._dec = (len(step_str.rstrip("0").split(".")[-1])
                         if "." in step_str else 0)
            self._fmt = f".{max(self._dec, 1)}f"
        else:
            self._dec = 0
            self._fmt = "d"

        lay = QHBoxLayout(self)
        lay.setContentsMargins(0, 2, 0, 2)
        lay.setSpacing(8)

        lbl = QLabel(pdef.label)
        lbl.setFixedWidth(80)
        lbl.setStyleSheet("color:#c9d1d9;")
        lay.addWidget(lbl)

        self.slider = QSlider(Qt.Orientation.Horizontal)
        self.slider.setRange(int(pdef.min_val * self._scale),
                             int(pdef.max_val * self._scale))
        self.slider.setValue(int(pdef.default * self._scale))
        self.slider.valueChanged.connect(self._update_label)
        lay.addWidget(self.slider, 1)

        self.vl = QLabel(format(pdef.default, self._fmt))
        self.vl.setFixedWidth(56)
        self.vl.setAlignment(Qt.AlignmentFlag.AlignRight
                             | Qt.AlignmentFlag.AlignVCenter)
        self.vl.setStyleSheet("color:#58a6ff;font-weight:bold;")
        self.vl.setToolTip("Double-click to reset")
        lay.addWidget(self.vl)

    def _update_label(self, v):
        val = v / self._scale
        if not self._is_float:
            val = int(val)
        self.vl.setText(format(val, self._fmt))
        self.changed.emit()

    def value(self) -> float | int:
        v = self.slider.value() / self._scale
        return v if self._is_float else int(v)

    def setValue(self, v):
        iv = int(v) if not self._is_float else v
        self.slider.blockSignals(True)
        self.slider.setValue(int(iv * self._scale))
        self.vl.setText(format(iv, self._fmt))
        self.slider.blockSignals(False)

    def resetToDefault(self):
        self.setValue(self._pdef.default)

    def mouseDoubleClickEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            self.resetToDefault()
            self.doubleClicked.emit()


class ArtWorker(QThread):
    """Runs a pattern function off the main thread; supports cancellation."""

    result = Signal(np.ndarray, float, str)   # image, elapsed, backend_label
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
        gpu = False
        try:
            t0 = time.perf_counter()
            gpu = self._backend == "gpu" and HAS_CUPY

            if self._pat.needs_backend_dispatch:
                x, y = _coord_cache.get(self._size, gpu=gpu)
                xp = cp if gpu else np
                img = self._pat.fn(
                    x, y, self._size, self._ck, xp,
                    self._backend, self._cancel, **self._params)
            else:
                x, y = _coord_cache.get(self._size, gpu=False)
                xp = np
                img = self._pat.fn(
                    x, y, self._size, self._ck, xp, **self._params)

            elapsed = time.perf_counter() - t0

            if img is None or self._cancel[0]:
                logger.info(f"Worker cancelled or returned None for {self._pat.name}")
                return

            if gpu and HAS_CUPY and isinstance(img, cp.ndarray):
                img = cp.asnumpy(img)

            if self._cancel[0]:
                logger.info(f"Worker cancelled post-computation for {self._pat.name}")
                return

            if gpu:
                label = "GPU (CuPy)"
            elif self._backend == "numba" and HAS_NUMBA:
                label = "CPU (Numba)"
            else:
                label = "CPU (NumPy)"

            logger.info(f"Worker completed: pattern={self._pat.name}, size={self._size}, elapsed={elapsed:.3f}s, backend={label}")
            self.result.emit(img, elapsed, label)

        except Exception as e:
            if not self._cancel[0]:
                err_msg = str(e)
                if "out of memory" in err_msg.lower():
                    err_msg = "GPU out of memory — switch to CPU backend"
                logger.exception(f"Worker failed: {err_msg}")
                self.error.emit(err_msg)
        finally:
            if gpu and HAS_CUPY:
                try:
                    cp.get_default_memory_pool().free_all_blocks()
                except Exception:
                    pass


# ═══════════════════════════════════════════════════════════════════════
#  § 6. MAIN WINDOW — UI, history, event handling
# ═══════════════════════════════════════════════════════════════════════

class HistoryStack:
    """Stores committed states for undo/redo.

    The undo stack holds ALL past states including the current one
    (the top of the stack). Undo moves the current to redo and
    reveals the previous state. Redo moves it back.
    """

    def __init__(self, max_size: int = 30):
        self._undo: list[tuple[QPixmap, str]] = []
        self._redo: list[tuple[QPixmap, str]] = []
        self._max = max_size

    def commit(self, px: QPixmap, status: str):
        """Record a new current state. Clears redo stack."""
        self._undo.append((px, status))
        if len(self._undo) > self._max:
            self._undo.pop(0)
        self._redo.clear()

    def undo(self) -> tuple[QPixmap, str] | None:
        """Move current to redo, return the previous state."""
        if len(self._undo) < 2:
            return None
        current = self._undo.pop()
        self._redo.append(current)
        return self._undo[-1]

    def redo(self) -> tuple[QPixmap, str] | None:
        """Restore a previously undone state."""
        if not self._redo:
            return None
        item = self._redo.pop()
        self._undo.append(item)
        return item

    def can_undo(self) -> bool:
        return len(self._undo) >= 2

    def can_redo(self) -> bool:
        return bool(self._redo)

    def clear(self):
        self._undo.clear()
        self._redo.clear()


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Algorithmic Art Studio")
        self.resize(1220, 820)
        self._worker: ArtWorker | None = None
        self._pending_workers: list[ArtWorker] = []
        self._gen_id = 0
        self._cur: QPixmap | None = None
        self._sliders: dict[str, ParamSlider] = {}
        self._auto = True
        self._timer: QTimer | None = None
        self._history = HistoryStack()
        self._build_ui()
        self._bind_shortcuts()

    # ── UI construction ─────────────────────────────────

    def _build_ui(self):
        cw = QWidget()
        self.setCentralWidget(cw)
        root = QHBoxLayout(cw)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)

        # ── Sidebar (with scroll) ──
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFixedWidth(320)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        scroll.setObjectName("sidebar")

        sidebar = QWidget()
        sl = QVBoxLayout(sidebar)
        sl.setContentsMargins(16, 20, 16, 16)
        sl.setSpacing(10)

        title = QLabel("Art Studio")
        title.setObjectName("title")
        sl.addWidget(title)
        sl.addWidget(self._hsep())

        # Pattern
        sl.addWidget(self._section("PATTERN"))
        self.pat_combo = QComboBox()
        self.pat_combo.addItems(p.name for p in PATTERNS)
        self.pat_combo.currentIndexChanged.connect(self._on_pattern_changed)
        sl.addWidget(self.pat_combo)
        self.desc_lbl = QLabel("")
        self.desc_lbl.setObjectName("desc")
        self.desc_lbl.setWordWrap(True)
        sl.addWidget(self.desc_lbl)
        sl.addWidget(self._hsep())

        # Image size
        sl.addWidget(self._section("IMAGE SIZE"))
        row = QHBoxLayout()
        self.sz_slider = QSlider(Qt.Orientation.Horizontal)
        self.sz_slider.setRange(2, 20)
        self.sz_slider.setValue(8)
        self.sz_slider.valueChanged.connect(self._on_size_changed)
        self.sz_lbl = QLabel("512 x 512")
        self.sz_lbl.setFixedWidth(80)
        self.sz_lbl.setAlignment(Qt.AlignmentFlag.AlignRight
                                 | Qt.AlignmentFlag.AlignVCenter)
        self.sz_lbl.setObjectName("szVal")
        row.addWidget(self.sz_slider, 1)
        row.addWidget(self.sz_lbl)
        sl.addLayout(row)
        sl.addWidget(self._hsep())

        # Compute backend
        sl.addWidget(self._section("COMPUTE"))
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
        sl.addWidget(self._hint(f"Available: {', '.join(avail)}"))
        sl.addWidget(self._hsep())

        # Color mode
        sl.addWidget(self._section("COLOR MODE"))
        self.cm_combo = QComboBox()
        for label, key in ALGO_COLORS:
            self.cm_combo.addItem(label, key)
        self.cm_combo.insertSeparator(self.cm_combo.count())
        for label, key in PAL_COLORS:
            self.cm_combo.addItem(label, key)
        self.cm_combo.currentIndexChanged.connect(self._on_color_changed)
        sl.addWidget(self.cm_combo)
        self.pal_preview = PalettePreview()
        sl.addWidget(self.pal_preview)
        sl.addWidget(self._hsep())

        # Parameters
        sl.addWidget(self._section("PARAMETERS"))
        self.par_box = QWidget()
        self.par_lay = QVBoxLayout(self.par_box)
        self.par_lay.setContentsMargins(0, 0, 0, 0)
        self.par_lay.setSpacing(4)
        sl.addWidget(self.par_box)
        sl.addWidget(self._hsep())

        # Auto-generate toggle
        self.auto_btn = QPushButton("Auto-Generate: ON")
        self.auto_btn.setCheckable(True)
        self.auto_btn.setChecked(True)
        self.auto_btn.setObjectName("autoBtn")
        self.auto_btn.toggled.connect(self._on_auto_toggled)
        sl.addWidget(self.auto_btn)

        # Action buttons — 3-column grid
        grid = QGridLayout()
        grid.setSpacing(6)

        self.gen_btn = QPushButton("Generate")
        self.gen_btn.setObjectName("genBtn")
        self.gen_btn.setToolTip("Generate art (Ctrl+G)")
        self.gen_btn.clicked.connect(self.generate)
        grid.addWidget(self.gen_btn, 0, 0)

        self.rnd_btn = QPushButton("Random")
        self.rnd_btn.setObjectName("rndBtn")
        self.rnd_btn.setToolTip("Randomize all settings (Ctrl+R)")
        self.rnd_btn.clicked.connect(self.randomize)
        grid.addWidget(self.rnd_btn, 0, 1)

        self.rst_btn = QPushButton("Reset")
        self.rst_btn.setObjectName("rstBtn")
        self.rst_btn.setToolTip("Reset parameters to defaults")
        self.rst_btn.clicked.connect(self.reset_params)
        grid.addWidget(self.rst_btn, 0, 2)

        self.sav_btn = QPushButton("Save")
        self.sav_btn.setObjectName("actBtn")
        self.sav_btn.setToolTip("Save image to file (Ctrl+S)")
        self.sav_btn.clicked.connect(self.save_image)
        grid.addWidget(self.sav_btn, 1, 0)

        self.cpy_btn = QPushButton("Copy")
        self.cpy_btn.setObjectName("actBtn")
        self.cpy_btn.setToolTip("Copy image to clipboard (Ctrl+Shift+C)")
        self.cpy_btn.clicked.connect(self.copy_image)
        grid.addWidget(self.cpy_btn, 1, 1)

        self.fit_btn = QPushButton("Fit View")
        self.fit_btn.setObjectName("actBtn")
        self.fit_btn.setToolTip("Reset zoom and pan (double-click canvas)")
        self.fit_btn.clicked.connect(self._reset_canvas_view)
        grid.addWidget(self.fit_btn, 1, 2)

        sl.addLayout(grid)

        sl.addStretch()
        scroll.setWidget(sidebar)
        root.addWidget(scroll)

        # ── Canvas area ──
        canvas_frame = QFrame()
        canvas_frame.setObjectName("canvasFrame")
        cl = QVBoxLayout(canvas_frame)
        cl.setContentsMargins(16, 16, 16, 16)
        self.canvas = ArtCanvas()
        cl.addWidget(self.canvas)
        root.addWidget(canvas_frame, 1)

        # ── Status bar ──
        self.sbar = QStatusBar()
        self.setStatusBar(self.sbar)
        self.st_lbl = QLabel("Ready")
        self.st_lbl.setToolTip(
            "Ctrl+G Generate | Ctrl+R Random | Ctrl+S Save | "
            "Ctrl+Shift+C Copy | Ctrl+Z Undo | Ctrl+Y Redo | "
            "F11 Fullscreen | Esc Cancel | Scroll to zoom | Drag to pan")
        self.sbar.addWidget(self.st_lbl, 1)

        # Init first pattern
        self._on_pattern_changed(0)

    # ── Tiny helpers ────────────────────────────────────

    @staticmethod
    def _hsep() -> QFrame:
        f = QFrame()
        f.setFrameShape(QFrame.Shape.HLine)
        f.setObjectName("sep")
        return f

    @staticmethod
    def _section(text: str) -> QLabel:
        l = QLabel(text)
        l.setObjectName("secLbl")
        return l

    @staticmethod
    def _hint(text: str) -> QLabel:
        l = QLabel(text)
        l.setObjectName("desc")
        return l

    # ── Shortcuts ───────────────────────────────────────

    def _bind_shortcuts(self):
        QShortcut(QKeySequence("Ctrl+G"), self, self.generate)
        QShortcut(QKeySequence("Ctrl+S"), self, self.save_image)
        QShortcut(QKeySequence("Ctrl+R"), self, self.randomize)
        QShortcut(QKeySequence("Ctrl+Shift+C"), self, self.copy_image)
        QShortcut(QKeySequence("Ctrl+Z"), self, self.undo)
        QShortcut(QKeySequence("Ctrl+Y"), self, self.redo)
        QShortcut(QKeySequence("F11"), self, self._toggle_fullscreen)
        QShortcut(QKeySequence("Escape"), self, self._cancel_generation)

    # ── Backend resolution ──────────────────────────────

    def _resolve_backend(self) -> str:
        txt = self.be_combo.currentText()
        if "CuPy" in txt:
            return "gpu"
        if "Numba" in txt:
            return "numba"
        if "NumPy" in txt:
            return "numpy"
        if HAS_CUPY:
            return "gpu"
        if HAS_NUMBA:
            return "numba"
        return "numpy"

    # ── Event handlers ──────────────────────────────────

    def _on_pattern_changed(self, idx):
        if idx < 0 or idx >= len(PATTERNS):
            return
        pat = PATTERNS[idx]
        self.desc_lbl.setText(pat.description)
        # Rebuild parameter sliders
        while self.par_lay.count():
            w = self.par_lay.takeAt(0).widget()
            if w:
                w.deleteLater()
        self._sliders.clear()
        for pd in pat.params:
            s = ParamSlider(pd)
            s.changed.connect(self._schedule_generate)
            self.par_lay.addWidget(s)
            self._sliders[pd.key] = s
        self._schedule_generate()

    def _on_size_changed(self, v):
        px = v * 64
        self.sz_lbl.setText(f"{px} x {px}")
        self._schedule_generate()

    def _on_color_changed(self, idx):
        key = self.cm_combo.itemData(idx)
        if key:
            self.pal_preview.setLut(colorizer.get_preview_lut(key))
        self._schedule_generate()

    def _on_auto_toggled(self, on):
        self._auto = on
        self.auto_btn.setText(f"Auto-Generate: {'ON' if on else 'OFF'}")
        if on:
            self._schedule_generate()

    def _schedule_generate(self):
        if not self._auto:
            return
        if self._timer is not None:
            self._timer.stop()
        self._timer = QTimer(self)
        self._timer.setSingleShot(True)
        self._timer.setInterval(80)
        self._timer.timeout.connect(self.generate)
        self._timer.start()

    # ── Generation ──────────────────────────────────────

    def generate(self):
        if self._worker is not None and self._worker.isRunning():
            self._worker.cancel()
            self._pending_workers.append(self._worker)

        self._gen_id += 1
        gid = self._gen_id

        pat = PATTERN_MAP[self.pat_combo.currentText()]
        size = self.sz_slider.value() * 64
        ck = self.cm_combo.currentData()
        if not ck:
            ck = "grayscale"
        params = {k: s.value() for k, s in self._sliders.items()}
        backend = self._resolve_backend()

        self.canvas.setLoading(True)
        self.gen_btn.setEnabled(False)
        self.st_lbl.setText("Generating...")

        self._worker = ArtWorker(pat, size, ck, params, backend)
        self._worker.result.connect(
            lambda img, t, b, g=gid: self._on_result(img, t, b, g))
        self._worker.error.connect(
            lambda msg, g=gid: self._on_error(msg, g))
        self._worker.finished.connect(
            lambda w=self._worker: self._cleanup_worker(w))
        self._worker.start()

    def _cleanup_worker(self, worker: ArtWorker):
        if worker in self._pending_workers:
            self._pending_workers.remove(worker)
        if worker is self._worker:
            self._worker = None
            if not self.gen_btn.isEnabled():
                self.canvas.setLoading(False)
                self.gen_btn.setEnabled(True)
        worker.deleteLater()

    def _on_result(self, img, elapsed, backend_name, gid):
        if gid != self._gen_id:
            return

        self._cur = to_pixmap(img)
        self.canvas.setArt(self._cur)
        self.gen_btn.setEnabled(True)

        h, w = img.shape[:2]
        mp = w * h / 1_000_000
        mp_str = f"{mp:.2f} MP" if mp >= 0.1 else f"{w * h:,} px"
        status = (
            f"{self.pat_combo.currentText()}   |   "
            f"{w}x{h} ({mp_str})   |   "
            f"{self.cm_combo.currentText()}   |   "
            f"{elapsed:.3f}s   |   {backend_name}")
        self.st_lbl.setText(status)
        self.setWindowTitle(
            f"{self.pat_combo.currentText()} - {w}x{h} - Art Studio")

        self._history.commit(self._cur, status)

    def _on_error(self, msg, gid):
        if gid != self._gen_id:
            return
        self.canvas.setLoading(False)
        self.gen_btn.setEnabled(True)
        self.st_lbl.setText(f"Error: {msg}")

    # ── Actions ─────────────────────────────────────────

    def reset_params(self):
        """Reset all parameter sliders to their defaults."""
        for s in self._sliders.values():
            s.resetToDefault()
        self._schedule_generate()

    def randomize(self):
        pat = random.choice(PATTERNS)
        self.pat_combo.blockSignals(True)
        self.pat_combo.setCurrentText(pat.name)
        self.pat_combo.blockSignals(False)
        self._on_pattern_changed(
            list(PATTERN_MAP.keys()).index(pat.name))

        for pd in pat.params:
            s = self._sliders.get(pd.key)
            if not s:
                continue
            if pd.is_float:
                val = round(random.uniform(pd.min_val, pd.max_val)
                           / pd.step) * pd.step
            else:
                val = random.randint(int(pd.min_val), int(pd.max_val))
            s.setValue(val)

        if pat.name == "Julia Set":
            preset = random.choice(JULIA_PRESETS)
            self._sliders["julia_r"].setValue(preset[0])
            self._sliders["julia_i"].setValue(preset[1])

        if pat.name == "Burning Ship":
            preset = random.choice(BURNING_SHIP_PRESETS)
            self._sliders["center_x"].setValue(preset[0])
            self._sliders["center_y"].setValue(preset[1])
            self._sliders["zoom"].setValue(preset[2])

        self.cm_combo.blockSignals(True)
        idx = random.choice(
            [i for i in range(self.cm_combo.count())
             if self.cm_combo.itemData(i) is not None])
        self.cm_combo.setCurrentIndex(idx)
        self.cm_combo.blockSignals(False)
        self._on_color_changed(self.cm_combo.currentIndex())

        self.sz_slider.blockSignals(True)
        self.sz_slider.setValue(random.randint(2, 16))
        self.sz_slider.blockSignals(False)
        px = self.sz_slider.value() * 64
        self.sz_lbl.setText(f"{px} x {px}")

        if self.be_combo.count() > 1:
            self.be_combo.blockSignals(True)
            self.be_combo.setCurrentIndex(
                random.randint(0, self.be_combo.count() - 1))
            self.be_combo.blockSignals(False)

        self.generate()

    def save_image(self):
        if not self._cur or self._cur.isNull():
            self.st_lbl.setText("Nothing to save - generate art first.")
            return
        path, _ = QFileDialog.getSaveFileName(
            self, "Save Art", "art.png",
            "PNG (*.png);;JPEG (*.jpg);;BMP (*.bmp)")
        if path:
            if self._cur.save(path):
                logger.info(f"Image saved successfully to {path}")
                self.st_lbl.setText(f"Saved: {path}")
            else:
                logger.error(f"Failed to save image to {path}")
                self.st_lbl.setText(f"Failed to save: {path}")

    def copy_image(self):
        if not self._cur or self._cur.isNull():
            self.st_lbl.setText("Nothing to copy - generate art first.")
            return
        QApplication.clipboard().setPixmap(self._cur)
        self.st_lbl.setText("Copied to clipboard")

    def undo(self):
        result = self._history.undo()
        if result is None:
            self.st_lbl.setText("Nothing to undo")
            return
        px, status = result
        self._cur = px
        self.canvas.setArt(px)
        self.st_lbl.setText(status)

    def redo(self):
        result = self._history.redo()
        if result is None:
            self.st_lbl.setText("Nothing to redo")
            return
        px, status = result
        self._cur = px
        self.canvas.setArt(px)
        self.st_lbl.setText(status)

    def _toggle_fullscreen(self):
        if self.isFullScreen():
            self.showNormal()
        else:
            self.showFullScreen()

    def _cancel_generation(self):
        if self._worker and self._worker.isRunning():
            self._worker.cancel()
            logger.info("Generation cancelled by user.")
            self.canvas.setLoading(False)
            self.gen_btn.setEnabled(True)
            self.st_lbl.setText("Cancelled")

    def _reset_canvas_view(self):
        self.canvas.resetView()


# ═══════════════════════════════════════════════════════════════════════
#  § 7. STYLESHEET
# ═══════════════════════════════════════════════════════════════════════

QSS = """
QMainWindow { background:#0d1117; }
QScrollArea { background:#161b22; border:none; }
#sidebar { background:#161b22; }
#sidebar > QWidget { background:#161b22; }
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

QScrollBar:vertical {
    background:#161b22; width:8px; border:none; }
QScrollBar::handle:vertical {
    background:#30363d; border-radius:4px; min-height:20px; }
QScrollBar::handle:vertical:hover { background:#484f58; }
QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical { height:0; }

QSlider::groove:horizontal {
    height:4px; background:#30363d; border-radius:2px; }
QSlider::handle:horizontal {
    background:#58a6ff; width:16px; height:16px; margin:-6px 0;
    border-radius:8px; border:2px solid #0d1117; }
QSlider::handle:horizontal:hover { background:#79c0ff; }
QSlider::sub-page:horizontal { background:#1f6feb; border-radius:2px; }

QPushButton {
    background:#21262d; border:1px solid #30363d; border-radius:6px;
    padding:8px 10px; color:#c9d1d9; font-size:12px; font-weight:bold; }
QPushButton:hover { background:#30363d; border-color:#8b949e; }
QPushButton:pressed { background:#0d1117; }
QPushButton:disabled { color:#484f58; border-color:#21262d; }

#genBtn { background:#238636; border-color:#2ea043; color:#fff; }
#genBtn:hover { background:#2ea043; }
#genBtn:disabled { background:#1a3a2a; border-color:#1a3a2a; color:#484f58; }
#rndBtn { background:#8957e5; border-color:#a371f7; color:#fff; }
#rndBtn:hover { background:#a371f7; }
#rstBtn { background:#6e4b1e; border-color:#9e6a2e; color:#f0c070; }
#rstBtn:hover { background:#9e6a2e; color:#fff; }
#actBtn { background:#1f6feb; border-color:#388bfd; color:#fff; }
#actBtn:hover { background:#388bfd; }

#autoBtn {
    background:transparent; border:1px solid #30363d;
    text-align:left; padding:8px 12px; font-size:12px; }
#autoBtn:checked { border-color:#238636; color:#3fb950; }

QStatusBar {
    background:#161b22; border-top:1px solid #30363d;
    color:#8b949e; font-size:12px; padding:2px 8px; }

QToolTip {
    background:#161b22; border:1px solid #30363d;
    color:#c9d1d9; padding:4px 8px; font-size:11px; }
"""


# ═══════════════════════════════════════════════════════════════════════
#  § 8. ENTRY POINT
# ═══════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    logger.info(f"Starting Algorithmic Art Studio (Numba={HAS_NUMBA}, CuPy={HAS_CUPY})")
    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    app.setStyleSheet(QSS)
    win = MainWindow()
    win.show()
    exit_code = app.exec()
    logger.info(f"Shutting down Algorithmic Art Studio (exit code: {exit_code})")
    sys.exit(exit_code)