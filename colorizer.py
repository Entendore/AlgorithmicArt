"""LUT-based colorizer with GPU awareness.

Builds 256-entry lookup tables from palette stops or HSV formulas.
Provides O(1) per-pixel colorization and caches GPU copies.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from config import PALETTES

__all__ = ["Colorizer", "colorizer"]


class Colorizer:
    """Precomputed 256-entry LUT colorizer; caches GPU copies."""

    def __init__(self) -> None:
        self._cpu: dict[str, NDArray] = {}
        self._gpu: dict[str, object] = {}
        self._build_all()

    # ── LUT construction ────────────────────────────────

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

    # ── Public API ──────────────────────────────────────

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

    # ── Internal ────────────────────────────────────────

    def _lut(self, key: str, xp):
        if xp is np:
            return self._cpu[key]
        if key not in self._gpu:
            from backends import cp
            self._gpu[key] = cp.asarray(self._cpu[key])
        return self._gpu[key]


# Module-level singleton
colorizer = Colorizer()