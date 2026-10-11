"""Custom Qt widgets: ArtCanvas, PalettePreview, ParamSlider, ArtWorker."""

from __future__ import annotations

import time
import numpy as np
from numpy.typing import NDArray

from PySide6.QtWidgets import (
    QWidget, QHBoxLayout, QLabel, QSlider, QSizePolicy,
)
from PySide6.QtGui import QPixmap, QImage, QFont, QColor, QPainter
from PySide6.QtCore import Qt, QSize, QThread, Signal

from config import ParamDef
from backends import HAS_CUPY, cp, _coord_cache

__all__ = [
    "ArtCanvas", "PalettePreview", "ParamSlider",
    "ArtWorker", "to_pixmap",
]


# ═══════════════════════════════════════════════════════════════
#  IMAGE UTILITY
# ═══════════════════════════════════════════════════════════════

def to_pixmap(arr: NDArray) -> QPixmap:
    """Convert an H×W (grayscale) or H×W×3 (RGB) uint8 array to QPixmap."""
    arr = np.ascontiguousarray(arr)
    h, w = arr.shape[:2]
    gray = arr.ndim == 2
    fmt = QImage.Format.Format_Grayscale8 if gray else QImage.Format.Format_RGB888
    stride = w if gray else w * 3
    return QPixmap.fromImage(QImage(arr.data, w, h, stride, fmt).copy())


# ═══════════════════════════════════════════════════════════════
#  ART CANVAS
# ═══════════════════════════════════════════════════════════════

class ArtCanvas(QWidget):
    """Central image display with drop-shadow and placeholder states."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self._px: QPixmap | None = None
        self._loading = False
        self.setMinimumSize(300, 300)
        self.setSizePolicy(QSizePolicy.Policy.Expanding,
                           QSizePolicy.Policy.Expanding)

    def setArt(self, px: QPixmap):
        self._px = px
        self._loading = False
        self.update()

    def setLoading(self, v: bool):
        self._loading = v
        self.update()

    def clearArt(self):
        self._px = None
        self._loading = False
        self.update()

    def paintEvent(self, _) -> None:
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform)
        p.fillRect(self.rect(), QColor("#0d1117"))

        if self._loading:
            p.setPen(QColor("#8b949e"))
            p.setFont(QFont("Segoe UI", 14))
            p.drawText(self.rect(), Qt.AlignmentFlag.AlignCenter,
                       "⏳  Generating …")
            p.end()
            return

        if self._px and not self._px.isNull():
            margin = 16
            sc = self._px.scaled(
                self.size() - QSize(margin, margin),
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation,
            )
            x = (self.width() - sc.width()) // 2
            y = (self.height() - sc.height()) // 2
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


# ═══════════════════════════════════════════════════════════════
#  PALETTE PREVIEW BAR
# ═══════════════════════════════════════════════════════════════

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
                Qt.TransformationMode.SmoothTransformation,
            )

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


# ═══════════════════════════════════════════════════════════════
#  PARAMETER SLIDER
# ═══════════════════════════════════════════════════════════════

class ParamSlider(QWidget):
    """Label + slider + value display for a single pattern parameter."""

    changed = Signal()

    def __init__(self, pdef: ParamDef, parent=None):
        super().__init__(parent)
        self._is_float = pdef.is_float
        self._scale = int(round(1.0 / pdef.step))
        
        # Derive format strictly from the is_float flag, not the step string
        if self._is_float:
            step_str = str(pdef.step)
            self._dec = (len(step_str.rstrip("0").split(".")[-1])
                         if "." in step_str else 0)
            self._fmt = f".{max(self._dec, 1)}f"  # Force at least 1 decimal place for floats
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
        lay.addWidget(self.vl)

    def _update_label(self, v):
        val = v / self._scale
        # Cast to int for integer sliders before formatting
        if not self._is_float:
            val = int(val)
        self.vl.setText(format(val, self._fmt))
        self.changed.emit()

    def value(self) -> float | int:
        v = self.slider.value() / self._scale
        # Return strict int for integer params so slices/ranges don't crash
        return v if self._is_float else int(v)

    def setValue(self, v):
        iv = int(v) if not self._is_float else v
        self.slider.blockSignals(True)
        self.slider.setValue(int(iv * self._scale))
        self.vl.setText(format(iv, self._fmt))
        self.slider.blockSignals(False)


# ═══════════════════════════════════════════════════════════════
#  BACKGROUND WORKER THREAD
# ═══════════════════════════════════════════════════════════════

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
        try:
            t0 = time.perf_counter()
            gpu = self._backend == "gpu" and HAS_CUPY
            x, y = _coord_cache.get(self._size, gpu=gpu)
            xp = cp if gpu else np

            if self._pat.needs_backend_dispatch:
                img = self._pat.fn(
                    x, y, self._size, self._ck, xp,
                    self._backend, self._cancel, **self._params)
            else:
                img = self._pat.fn(
                    x, y, self._size, self._ck, xp, **self._params)

            elapsed = time.perf_counter() - t0

            if img is None or self._cancel[0]:
                return

            if gpu and HAS_CUPY and isinstance(img, cp.ndarray):
                img = cp.asnumpy(img)

            if self._cancel[0]:
                return

            if gpu:
                label = "GPU (CuPy)"
            elif self._backend == "numba" and HAS_NUMBA:
                label = "CPU (Numba)"
            else:
                label = "CPU (NumPy)"

            self.result.emit(img, elapsed, label)

        except Exception as e:
            if not self._cancel[0]:
                err_msg = str(e)
                if "out of memory" in err_msg.lower():
                    err_msg = "GPU out of memory — switch to CPU backend"
                self.error.emit(err_msg)
        finally:
            if gpu and HAS_CUPY:
                try:
                    cp.get_default_memory_pool().free_all_blocks()
                except Exception:
                    pass