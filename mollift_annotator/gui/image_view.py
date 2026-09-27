"""Zoomable projection image view using QGraphicsView."""

from __future__ import annotations

from typing import Dict, Optional, Tuple

from PIL.ImageQt import toqpixmap
from PySide6.QtCore import Signal
from PySide6.QtGui import QBrush, QPen
from PySide6.QtWidgets import QGraphicsEllipseItem, QGraphicsPixmapItem, QGraphicsSimpleTextItem

from mollift_annotator.gui.zoom_view import ZoomableGraphicsView
from mollift_annotator.render import load_projection_image


class ZoomableImageView(ZoomableGraphicsView):
    """Display projection image with zoom/pan; scene coords = original image pixels."""

    clicked = Signal(float, float)

    def __init__(self, image_path: str, parent=None) -> None:
        super().__init__(parent)
        self._orig_image = load_projection_image(image_path)
        self.orig_width, self.orig_height = self._orig_image.size

        self._bindings: Dict[int, Tuple[float, float]] = {}
        self._pending: Optional[Tuple[float, float]] = None
        self._marker_items: list = []

        self._pixmap_item = QGraphicsPixmapItem(toqpixmap(self._orig_image.convert("RGBA")))
        self._scene.addItem(self._pixmap_item)
        self._scene.setSceneRect(0, 0, self.orig_width, self.orig_height)

    def _on_left_click(self, scene_x: float, scene_y: float) -> None:
        ox = max(0.0, min(float(self.orig_width - 1), scene_x))
        oy = max(0.0, min(float(self.orig_height - 1), scene_y))
        self.clicked.emit(ox, oy)

    def set_bindings(self, bindings: Dict[int, Tuple[float, float]]) -> None:
        self._bindings = dict(bindings)
        self._rebuild_markers()

    def set_pending(self, pending: Optional[Tuple[float, float]]) -> None:
        self._pending = pending
        self._rebuild_markers()

    def _rebuild_markers(self) -> None:
        for item in self._marker_items:
            self._scene.removeItem(item)
        self._marker_items.clear()

        for atom_idx, (px, py) in self._bindings.items():
            self._marker_items.extend(
                self._make_marker(px, py, str(atom_idx), "#22AA44")
            )
        if self._pending is not None:
            px, py = self._pending
            self._marker_items.extend(self._make_marker(px, py, "?", "#FF8800"))

    def _make_marker(
        self,
        x: float,
        y: float,
        label: str,
        color: str,
    ) -> list:
        from PySide6.QtCore import Qt

        r = 6.0
        ellipse = QGraphicsEllipseItem(x - r, y - r, 2 * r, 2 * r)
        ellipse.setPen(QPen(Qt.GlobalColor.white, 2))
        ellipse.setBrush(QBrush(color))
        ellipse.setZValue(10)
        text = QGraphicsSimpleTextItem(label)
        text.setPos(x + 8, y - 16)
        text.setBrush(QBrush(Qt.GlobalColor.white))
        text.setZValue(11)
        self._scene.addItem(ellipse)
        self._scene.addItem(text)
        return [ellipse, text]
