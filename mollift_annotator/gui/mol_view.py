"""Zoomable molecule view with atom click hit-testing."""

from __future__ import annotations

from typing import Optional, Set

from PIL.ImageQt import toqpixmap
from PySide6.QtCore import Signal
from PySide6.QtWidgets import QGraphicsPixmapItem
from rdkit import Chem

from mollift_annotator.gui.zoom_view import ZoomableGraphicsView
from mollift_annotator.render import (
    DEFAULT_HIT_RADIUS,
    DEFAULT_MOL_HEIGHT,
    DEFAULT_MOL_WIDTH,
    atom_at_point,
    compute_mol_render_size,
    render_molecule,
)


class MolView(ZoomableGraphicsView):
    """Display RDKit molecule with zoom/pan; clicks select nearest atom."""

    atomClicked = Signal(int)

    def __init__(
        self,
        mol: Chem.Mol,
        width: int = DEFAULT_MOL_WIDTH,
        height: int = DEFAULT_MOL_HEIGHT,
        hit_radius: float = DEFAULT_HIT_RADIUS,
        parent=None,
    ) -> None:
        super().__init__(parent)
        self.mol = mol
        self.view_width = width
        self.view_height = height
        self._base_hit_radius = hit_radius

        self._render_width, self._render_height, self._render_scale = compute_mol_render_size(
            mol.GetNumAtoms(),
            view_width=width,
            view_height=height,
        )
        self.hit_radius = hit_radius * self._render_scale

        self._annotated: Set[int] = set()
        self._skipped: Set[int] = set()
        self._selected: Optional[int] = None
        self._atom_coords: dict[int, tuple[float, float]] = {}

        self._pixmap_item = QGraphicsPixmapItem()
        self._scene.addItem(self._pixmap_item)
        self._scene.setSceneRect(0, 0, self._render_width, self._render_height)
        self.setMinimumHeight(320)
        self._render()

    def set_annotated(self, atom_indices: Set[int]) -> None:
        self._annotated = set(atom_indices)
        self._render()

    def set_skipped(self, atom_indices: Set[int]) -> None:
        self._skipped = set(atom_indices)
        self._render()

    def set_selected(self, atom_index: Optional[int]) -> None:
        self._selected = atom_index
        self._render()

    def _on_left_click(self, scene_x: float, scene_y: float) -> None:
        idx = atom_at_point(
            scene_x,
            scene_y,
            self._atom_coords,
            self.hit_radius,
        )
        if idx is not None:
            self.atomClicked.emit(idx)

    def _render(self) -> None:
        img, self._atom_coords = render_molecule(
            self.mol,
            width=self._render_width,
            height=self._render_height,
            annotated=self._annotated,
            skipped=self._skipped,
            selected=self._selected,
        )
        pixmap = toqpixmap(img.convert("RGBA"))
        pixmap.setDevicePixelRatio(1.0)
        self._pixmap_item.setPixmap(pixmap)
        self._scene.setSceneRect(0, 0, self._render_width, self._render_height)
