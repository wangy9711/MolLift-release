"""Base QGraphicsView with zoom and pan."""

from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtGui import QBrush, QPainter, QWheelEvent
from PySide6.QtWidgets import QGraphicsScene, QGraphicsView

ZOOM_STEP = 1.15
MIN_ZOOM = 0.05
MAX_ZOOM = 20.0


class ZoomableGraphicsView(QGraphicsView):
    """Graphics view with wheel zoom and middle-button / space+drag pan."""

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self._zoom = 1.0
        self._space_pan = False
        self._scene = QGraphicsScene(self)
        self.setScene(self._scene)

        self.setRenderHint(QPainter.RenderHint.Antialiasing)
        self.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform)
        self.setDragMode(QGraphicsView.DragMode.NoDrag)
        self.setTransformationAnchor(QGraphicsView.ViewportAnchor.AnchorUnderMouse)
        self.setResizeAnchor(QGraphicsView.ViewportAnchor.AnchorUnderMouse)
        self.setBackgroundBrush(QBrush(Qt.GlobalColor.darkGray))
        self.setMouseTracking(True)
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)

    def showEvent(self, event) -> None:
        super().showEvent(event)
        if abs(self._zoom - 1.0) < 1e-6 and self._scene.sceneRect().isValid():
            self.fit_to_view()

    def keyPressEvent(self, event) -> None:
        if event.key() == Qt.Key.Key_Space and not event.isAutoRepeat():
            self._space_pan = True
            self.setCursor(Qt.CursorShape.OpenHandCursor)
        super().keyPressEvent(event)

    def keyReleaseEvent(self, event) -> None:
        if event.key() == Qt.Key.Key_Space and not event.isAutoRepeat():
            self._space_pan = False
            self.setCursor(Qt.CursorShape.ArrowCursor)
            self.setDragMode(QGraphicsView.DragMode.NoDrag)
        super().keyReleaseEvent(event)

    def wheelEvent(self, event: QWheelEvent) -> None:
        factor = ZOOM_STEP if event.angleDelta().y() > 0 else 1.0 / ZOOM_STEP
        new_zoom = self._zoom * factor
        if MIN_ZOOM <= new_zoom <= MAX_ZOOM:
            self.scale(factor, factor)
            self._zoom = new_zoom
        event.accept()

    def mousePressEvent(self, event) -> None:
        if event.button() == Qt.MouseButton.MiddleButton or (
            event.button() == Qt.MouseButton.LeftButton and self._space_pan
        ):
            self.setDragMode(QGraphicsView.DragMode.ScrollHandDrag)
            super().mousePressEvent(event)
            if self._space_pan:
                self.setCursor(Qt.CursorShape.ClosedHandCursor)
            return

        if event.button() == Qt.MouseButton.LeftButton:
            scene_pos = self.mapToScene(event.position().toPoint())
            self._on_left_click(scene_pos.x(), scene_pos.y())
            event.accept()
            return

        super().mousePressEvent(event)

    def mouseReleaseEvent(self, event) -> None:
        super().mouseReleaseEvent(event)
        if event.button() in (Qt.MouseButton.MiddleButton, Qt.MouseButton.LeftButton):
            self.setDragMode(QGraphicsView.DragMode.NoDrag)
            if self._space_pan:
                self.setCursor(Qt.CursorShape.OpenHandCursor)
            else:
                self.setCursor(Qt.CursorShape.ArrowCursor)

    def _on_left_click(self, scene_x: float, scene_y: float) -> None:
        """Override in subclasses."""

    def zoom_in(self) -> None:
        self._apply_zoom(ZOOM_STEP)

    def zoom_out(self) -> None:
        self._apply_zoom(1.0 / ZOOM_STEP)

    def fit_to_view(self) -> None:
        self.resetTransform()
        self._zoom = 1.0
        rect = self._scene.sceneRect()
        if rect.isValid():
            self.fitInView(rect, Qt.AspectRatioMode.KeepAspectRatio)
            self._zoom = self.transform().m11()

    def reset_zoom(self) -> None:
        self.resetTransform()
        self._zoom = 1.0

    def _apply_zoom(self, factor: float) -> None:
        new_zoom = self._zoom * factor
        if MIN_ZOOM <= new_zoom <= MAX_ZOOM:
            self.scale(factor, factor)
            self._zoom = new_zoom
