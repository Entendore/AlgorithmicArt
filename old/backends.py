"""Compute backends: Numba JIT, CuPy GPU, and NumPy fallback.

Detection is run once at import time.  The public API consists of:
    HAS_NUMBA, HAS_CUPY          — bool flags
    _dispatch_blur()              — box-blur with backend selection
    _dispatch_mandelbrot()        — Mandelbrot fractal
    _dispatch_julia()             — Julia set fractal
    CoordCache                    — LRU cache for coordinate grids
    _coord_cache                  — module-level CoordCache instance
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

__all__ = [
    "HAS_NUMBA", "HAS_CUPY", "cp",
    "_dispatch_blur", "_dispatch_mandelbrot", "_dispatch_julia",
    "CoordCache", "_coord_cache",
]

# ═══════════════════════════════════════════════════════════════
#  OPTIONAL DEPENDENCY DETECTION
# ═══════════════════════════════════════════════════════════════

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


def _numba_jit(fn):
    """Decorator: apply Numba JIT when available, else pass through."""
    if HAS_NUMBA:
        return njit(parallel=True, fastmath=True, cache=True)(fn)
    return fn


# ═══════════════════════════════════════════════════════════════
#  COORDINATE CACHE  (LRU, GPU-aware)
# ═══════════════════════════════════════════════════════════════

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


# Module-level singleton — imported by widgets.py
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


# Warm up Numba at import time
if HAS_NUMBA:
    _tiny = np.zeros((2, 2), dtype=np.float64)
    _tiny_cancel = np.array([False], dtype=np.bool_)
    _mandelbrot_numba(_tiny, _tiny, 2, _tiny_cancel)
    _julia_numba(_tiny, _tiny, -0.7, 0.27, 2, _tiny_cancel)
    _box_blur_numba(np.zeros((4, 4), dtype=np.float32), 1)
    del _tiny, _tiny_cancel


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


def _dispatch_julia(zr, zi, cr, ci, max_iter, cancel, backend):
    if backend == "gpu" and HAS_CUPY:
        return _julia_gpu(zr, zi, cr, ci, max_iter, cancel)
    if backend == "numba" and HAS_NUMBA:
        return _julia_numba(zr, zi, cr, ci, max_iter, cancel)
    return _julia_numpy(zr, zi, cr, ci, max_iter, cancel)