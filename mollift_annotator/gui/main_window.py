"""Main PySide6 window for 2D atom annotation."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QComboBox,
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QSplitter,
    QStatusBar,
    QToolBar,
    QVBoxLayout,
    QWidget,
)

from mollift_annotator.export import load_annotation, save_annotation
from mollift_annotator.gui.image_view import ZoomableImageView
from mollift_annotator.gui.mol_view import MolView
from mollift_annotator.session import AnnotationSession


class AnnotatorWindow(QMainWindow):
    """Desktop annotation UI."""

    def __init__(
        self,
        smiles: str,
        image_path: str,
        output_path: Optional[str] = None,
    ) -> None:
        super().__init__()
        self.setWindowTitle("MolLift 2D Atom Annotator")
        self.resize(1100, 780)
        self.setMinimumSize(900, 640)

        self.session = AnnotationSession(smiles=smiles, image_path=image_path)
        self.session.set_on_change(self._refresh_ui)

        self._pending_pixel: Optional[tuple[float, float]] = None
        self._selected_atom: Optional[int] = None
        self._output_path_str = output_path or "annotation.json"

        self._build_ui()
        self._refresh_ui()
        self._set_status(
            "Mode 1: click image then molecule | Mode 2: select atom then click image | "
            "Wheel: zoom | Middle-drag or Space+drag: pan"
        )

    def _build_ui(self) -> None:
        central = QWidget()
        self.setCentralWidget(central)
        root = QVBoxLayout(central)

        splitter = QSplitter(Qt.Orientation.Horizontal)
        root.addWidget(splitter, stretch=1)

        left = QWidget()
        left_layout = QVBoxLayout(left)
        left_layout.setContentsMargins(4, 4, 4, 4)
        left_layout.addWidget(QLabel("<b>Projection Image</b>"))
        left_layout.addWidget(
            QLabel("Left-click: annotate | Wheel: zoom | Middle/Space+drag: pan (pixel coords)")
        )
        self.image_view = ZoomableImageView(self.session.image_path)
        self.image_view.clicked.connect(self._on_image_click)
        left_layout.addWidget(self.image_view)
        splitter.addWidget(left)

        right = QWidget()
        right_layout = QVBoxLayout(right)
        right_layout.setContentsMargins(4, 4, 4, 4)
        right_layout.addWidget(QLabel("<b>Molecule (SMILES)</b>"))
        smiles_short = self.session.smiles
        if len(smiles_short) > 80:
            smiles_short = smiles_short[:77] + "..."
        smiles_label = QLabel(smiles_short)
        smiles_label.setWordWrap(True)
        right_layout.addWidget(smiles_label)
        right_layout.addWidget(
            QLabel("Left-click: select atom | Wheel: zoom | Middle/Space+drag: pan")
        )
        self.mol_view = MolView(self.session.mol)
        self.mol_view.atomClicked.connect(self._on_mol_click)
        right_layout.addWidget(self.mol_view)
        splitter.addWidget(right)
        splitter.setStretchFactor(0, 3)
        splitter.setStretchFactor(1, 2)

        toolbar = QToolBar("Toolbar")
        toolbar.setMovable(False)
        self.addToolBar(toolbar)

        toolbar.addWidget(QLabel("Atom:"))
        self._atom_combo = QComboBox()
        for i in range(self.session.num_atoms):
            self._atom_combo.addItem(f"{i}: {self.session.atom_symbol(i)}", i)
        if self.session.num_atoms > 0:
            self._atom_combo.setCurrentIndex(0)
            self._selected_atom = 0
        self._atom_combo.currentIndexChanged.connect(self._on_combo_select)
        toolbar.addWidget(self._atom_combo)

        toolbar.addSeparator()
        for text, slot in [
            ("Undo", self._do_undo),
            ("Clear", self._do_clear),
            ("Skip", self._do_skip),
        ]:
            btn = QPushButton(text)
            btn.clicked.connect(slot)
            toolbar.addWidget(btn)

        toolbar.addSeparator()
        toolbar.addWidget(QLabel("Image:"))
        for text, method in [
            ("Zoom In", self.image_view.zoom_in),
            ("Zoom Out", self.image_view.zoom_out),
            ("Fit", self.image_view.fit_to_view),
            ("100%", self.image_view.reset_zoom),
        ]:
            btn = QPushButton(text)
            btn.clicked.connect(method)
            toolbar.addWidget(btn)

        toolbar.addSeparator()
        toolbar.addWidget(QLabel("Mol:"))
        for text, method in [
            ("Zoom In", self.mol_view.zoom_in),
            ("Zoom Out", self.mol_view.zoom_out),
            ("Fit", self.mol_view.fit_to_view),
            ("100%", self.mol_view.reset_zoom),
        ]:
            btn = QPushButton(text)
            btn.clicked.connect(method)
            toolbar.addWidget(btn)

        self._progress_label = QLabel(self.session.progress_text())
        toolbar.addWidget(self._progress_label)

        io_row = QWidget()
        io_layout = QHBoxLayout(io_row)
        io_layout.addWidget(QLabel("Output:"))
        self._output_edit = QLineEdit(self._output_path_str)
        io_layout.addWidget(self._output_edit, stretch=1)
        for text, slot in [
            ("Browse...", self._browse_output),
            ("Save", self._do_save),
            ("Load", self._do_load),
        ]:
            btn = QPushButton(text)
            btn.clicked.connect(slot)
            io_layout.addWidget(btn)
        root.addWidget(io_row)

        self._status = QStatusBar()
        self.setStatusBar(self._status)

    def _on_combo_select(self, index: int) -> None:
        if index >= 0:
            self._selected_atom = self._atom_combo.itemData(index)
            self._refresh_ui()

    def _on_image_click(self, px: float, py: float) -> None:
        pixel = (px, py)
        if self._selected_atom is not None:
            warning = self.session.bind(self._selected_atom, pixel)
            self._pending_pixel = None
            msg = f"Bound atom {self._selected_atom} -> ({px:.1f}, {py:.1f})"
            if warning:
                msg = f"{warning} | {msg}"
            self._set_status(msg)
            return

        self._pending_pixel = pixel
        self.image_view.set_pending(pixel)
        self._set_status(f"Pending ({px:.1f}, {py:.1f}) — click an atom on the molecule panel")

    def _on_mol_click(self, atom_index: int) -> None:
        self._selected_atom = atom_index
        self._atom_combo.setCurrentIndex(atom_index)

        if self._pending_pixel is not None:
            warning = self.session.bind(atom_index, self._pending_pixel)
            self._pending_pixel = None
            px, py = self.session.bindings[atom_index]
            msg = f"Bound atom {atom_index} -> ({px:.1f}, {py:.1f})"
            if warning:
                msg = f"{warning} | {msg}"
            self._set_status(msg)
        else:
            self._refresh_ui()
            self._set_status(
                f"Selected atom {atom_index} ({self.session.atom_symbol(atom_index)})"
            )

    def _do_undo(self) -> None:
        if self.session.undo():
            self._pending_pixel = None
            self._set_status("Undone last action")
        else:
            self._set_status("Nothing to undo")

    def _do_clear(self) -> None:
        if self._selected_atom is None:
            self._set_status("Select an atom to clear")
            return
        self.session.unbind(self._selected_atom)
        self._set_status(f"Cleared binding for atom {self._selected_atom}")

    def _do_skip(self) -> None:
        if self._selected_atom is None:
            self._set_status("Select an atom to skip")
            return
        self.session.skip(self._selected_atom)
        self._pending_pixel = None
        self._set_status(f"Atom {self._selected_atom} marked as skipped (occluded)")

    def _do_save(self) -> None:
        path = self._output_edit.text().strip()
        if not path:
            QMessageBox.warning(self, "Save", "Please enter an output path")
            return
        try:
            save_annotation(path, self.session.to_record())
            self._set_status(f"Saved: {Path(path).resolve()}")
        except Exception as exc:
            QMessageBox.critical(self, "Save Failed", str(exc))

    def _do_load(self) -> None:
        path = self._output_edit.text().strip()
        if not path:
            path, _ = QFileDialog.getOpenFileName(
                self,
                "Load Annotation JSON",
                "",
                "JSON (*.json);;All (*.*)",
            )
            if not path:
                return
            self._output_edit.setText(path)
        try:
            record = load_annotation(path)
            self.session.load_record(record)
            self._pending_pixel = None
            self._set_status(f"Loaded: {Path(path).resolve()}")
        except Exception as exc:
            QMessageBox.critical(self, "Load Failed", str(exc))

    def _browse_output(self) -> None:
        path, _ = QFileDialog.getSaveFileName(
            self,
            "Save Annotation JSON",
            self._output_edit.text().strip() or "annotation.json",
            "JSON (*.json);;All (*.*)",
        )
        if path:
            self._output_edit.setText(path)

    def _refresh_ui(self) -> None:
        bindings = self.session.bindings
        self.image_view.set_bindings(bindings)
        self.image_view.set_pending(self._pending_pixel)
        self.mol_view.set_annotated(set(bindings.keys()))
        self.mol_view.set_skipped(set(self.session.skipped))
        self.mol_view.set_selected(self._selected_atom)
        self._progress_label.setText(self.session.progress_text())

    def _set_status(self, text: str) -> None:
        self._status.showMessage(text)

    def save(self, path: Optional[str] = None) -> None:
        out = path or self._output_edit.text().strip()
        save_annotation(out, self.session.to_record())
