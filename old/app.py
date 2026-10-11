#!/usr/bin/env python3
"""Algorithmic Art Studio — main window, stylesheet, and entry point.

Improvements over the original:
  • Julia Set fractal with curated presets
  • Plasma pattern from layered sine waves
  • 6 new themed palettes (Ember, Arctic, Forest, Sunset, Rose, Mint)
  • Undo history (Ctrl+Z, up to 12 steps)
  • Fullscreen toggle (F11)
  • ESC to cancel in-progress generation
  • Megapixel count in status bar
  • Tooltips on all action buttons
  • Resolution extended to 1280 px
  • Fixed thread lifecycle (no more "Destroyed while thread is running")
"""

import sys
import random

from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QComboBox, QSlider, QPushButton, QFrame,
    QFileDialog, QStatusBar, QSizePolicy, QGridLayout,
)
from PySide6.QtGui import QPixmap, QFont, QKeySequence, QShortcut
from PySide6.QtCore import Qt, QTimer

from config import ALGO_COLORS, PAL_COLORS
from algorithms import PATTERNS, PATTERN_MAP, JULIA_PRESETS
from backends import HAS_NUMBA, HAS_CUPY
from colorizer import colorizer
from widgets import (
    ArtCanvas, PalettePreview, ParamSlider, ArtWorker, to_pixmap,
)


# ═══════════════════════════════════════════════════════════════
#  HISTORY STACK
# ═══════════════════════════════════════════════════════════════

class HistoryStack:
    """Stores recent (pixmap, status_text) pairs for undo."""

    def __init__(self, max_size: int = 12):
        self._items: list[tuple[QPixmap, str]] = []
        self._max = max_size

    def push(self, px: QPixmap, status: str):
        self._items.append((px, status))
        if len(self._items) > self._max:
            self._items.pop(0)

    def pop(self) -> tuple[QPixmap, str] | None:
        return self._items.pop() if self._items else None

    def can_undo(self) -> bool:
        return bool(self._items)


# ═══════════════════════════════════════════════════════════════
#  MAIN WINDOW
# ═══════════════════════════════════════════════════════════════

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
        self._on_pattern_changed(0)
        QTimer.singleShot(150, self.generate)

    # ── UI construction ─────────────────────────────────

    def _build_ui(self):
        cw = QWidget()
        self.setCentralWidget(cw)
        root = QHBoxLayout(cw)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)

        # ── Sidebar ──
        sidebar = QFrame()
        sidebar.setFixedWidth(310)
        sidebar.setObjectName("sidebar")
        sl = QVBoxLayout(sidebar)
        sl.setContentsMargins(16, 20, 16, 16)
        sl.setSpacing(10)

        title = QLabel("🎨  Art Studio")
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
        self.sz_lbl = QLabel("512 × 512")
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
        self.auto_btn = QPushButton("⟳  Auto-Generate: ON")
        self.auto_btn.setCheckable(True)
        self.auto_btn.setChecked(True)
        self.auto_btn.setObjectName("autoBtn")
        self.auto_btn.toggled.connect(self._on_auto_toggled)
        sl.addWidget(self.auto_btn)

        # Action buttons
        grid = QGridLayout()
        grid.setSpacing(8)
        self.gen_btn = QPushButton("⚡ Generate")
        self.gen_btn.setObjectName("genBtn")
        self.gen_btn.setToolTip("Generate art (Ctrl+G)")
        self.gen_btn.clicked.connect(self.generate)
        grid.addWidget(self.gen_btn, 0, 0)

        self.rnd_btn = QPushButton("🎲 Random")
        self.rnd_btn.setObjectName("rndBtn")
        self.rnd_btn.setToolTip("Randomize all settings (Ctrl+R)")
        self.rnd_btn.clicked.connect(self.randomize)
        grid.addWidget(self.rnd_btn, 0, 1)

        self.sav_btn = QPushButton("💾 Save")
        self.sav_btn.setObjectName("actBtn")
        self.sav_btn.setToolTip("Save image to file (Ctrl+S)")
        self.sav_btn.clicked.connect(self.save_image)
        grid.addWidget(self.sav_btn, 1, 0)

        self.cpy_btn = QPushButton("📋 Copy")
        self.cpy_btn.setObjectName("actBtn")
        self.cpy_btn.setToolTip("Copy image to clipboard (Ctrl+Shift+C)")
        self.cpy_btn.clicked.connect(self.copy_image)
        grid.addWidget(self.cpy_btn, 1, 1)
        sl.addLayout(grid)

        sl.addStretch()
        root.addWidget(sidebar)

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
            "Ctrl+G Generate · Ctrl+R Random · Ctrl+S Save · "
            "Ctrl+Shift+C Copy · Ctrl+Z Undo · F11 Fullscreen · Esc Cancel")
        self.sbar.addWidget(self.st_lbl, 1)

        # Init palette preview
        self._on_color_changed(0)

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
        pat = PATTERN_MAP[self.pat_combo.currentText()]
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
        self.sz_lbl.setText(f"{px} × {px}")
        self._schedule_generate()

    def _on_color_changed(self, idx):
        key = self.cm_combo.itemData(idx)
        if key:
            self.pal_preview.setLut(colorizer.get_preview_lut(key))
        self._schedule_generate()

    def _on_auto_toggled(self, on):
        self._auto = on
        self.auto_btn.setText(f"⟳  Auto-Generate: {'ON' if on else 'OFF'}")
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
        # Cancel any running worker but DO NOT block the UI with wait()
        if self._worker is not None and self._worker.isRunning():
            self._worker.cancel()
            # Keep a reference so it doesn't get garbage-collected while finishing
            self._pending_workers.append(self._worker)

        self._gen_id += 1
        gid = self._gen_id

        pat = PATTERN_MAP[self.pat_combo.currentText()]
        size = self.sz_slider.value() * 64
        ck = (self.cm_combo.currentData()
              or self.cm_combo.currentText().lower().replace(" ", "_"))
        params = {k: s.value() for k, s in self._sliders.items()}
        backend = self._resolve_backend()

        self.canvas.setLoading(True)
        self.gen_btn.setEnabled(False)
        self.st_lbl.setText("Generating …")

        self._worker = ArtWorker(pat, size, ck, params, backend)
        self._worker.result.connect(
            lambda img, t, b, g=gid: self._on_result(img, t, b, g))
        self._worker.error.connect(
            lambda msg, g=gid: self._on_error(msg, g))
        
        # Schedule safe Qt deletion when the thread finishes
        self._worker.finished.connect(
            lambda w=self._worker: self._cleanup_worker(w))
        
        self._worker.start()

    def _cleanup_worker(self, worker: ArtWorker):
        """Safely delete replaced workers. Keep the active worker alive."""
        # If this is the currently active worker, don't delete it yet
        # so self._worker remains a valid reference for the next generation.
        if worker is self._worker:
            return
        
        if worker in self._pending_workers:
            self._pending_workers.remove(worker)
            
        # Safe to delete because the QThread object lives in the main thread,
        # so deleteLater will be processed by the main event loop.
        worker.deleteLater()

    def _on_result(self, img, elapsed, backend_name, gid):
        if gid != self._gen_id:
            return
        # Push current art to history before replacing
        if self._cur is not None and not self._cur.isNull():
            self._history.push(self._cur, self.st_lbl.text())

        self._cur = to_pixmap(img)
        self.canvas.setArt(self._cur)
        self.gen_btn.setEnabled(True)

        h, w = img.shape[:2]
        mp = w * h / 1_000_000
        mp_str = f"{mp:.2f} MP" if mp >= 0.1 else f"{w * h:,} px"
        self.st_lbl.setText(
            f"✓  {self.pat_combo.currentText()}   |   "
            f"{w}×{h} ({mp_str})   |   "
            f"{self.cm_combo.currentText()}   |   "
            f"{elapsed:.3f}s   |   {backend_name}")
        self.setWindowTitle(
            f"{self.pat_combo.currentText()} — {w}×{h} — Art Studio")

    def _on_error(self, msg, gid):
        if gid != self._gen_id:
            return
        self.canvas.setLoading(False)
        self.gen_btn.setEnabled(True)
        self.st_lbl.setText(f"✗  Error: {msg}")

    # ── Actions ─────────────────────────────────────────

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

        # Use Julia presets for better randomization
        if pat.name == "Julia Set":
            preset = random.choice(JULIA_PRESETS)
            self._sliders["julia_r"].setValue(preset[0])
            self._sliders["julia_i"].setValue(preset[1])

        self.cm_combo.blockSignals(True)
        # Skip separator at index 4
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
        self.sz_lbl.setText(f"{px} × {px}")

        self.generate()

    def save_image(self):
        if not self._cur or self._cur.isNull():
            self.st_lbl.setText("Nothing to save — generate art first.")
            return
        path, _ = QFileDialog.getSaveFileName(
            self, "Save Art", "art.png",
            "PNG (*.png);;JPEG (*.jpg);;BMP (*.bmp)")
        if path:
            self._cur.save(path)
            self.st_lbl.setText(f"Saved → {path}")

    def copy_image(self):
        if not self._cur or self._cur.isNull():
            self.st_lbl.setText("Nothing to copy — generate art first.")
            return
        QApplication.clipboard().setPixmap(self._cur)
        self.st_lbl.setText("Copied to clipboard ✓")

    def undo(self):
        if not self._history.can_undo():
            self.st_lbl.setText("Nothing to undo")
            return
        px, status = self._history.pop()
        self._cur = px
        self.canvas.setArt(px)
        self.st_lbl.setText(f"↩  {status}")

    def _toggle_fullscreen(self):
        if self.isFullScreen():
            self.showNormal()
        else:
            self.showFullScreen()

    def _cancel_generation(self):
        if self._worker and self._worker.isRunning():
            self._worker.cancel()
            self.st_lbl.setText("Cancelled")


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

QToolTip {
    background:#161b22; border:1px solid #30363d;
    color:#c9d1d9; padding:4px 8px; font-size:11px; }
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