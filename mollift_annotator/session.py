"""Annotation session state machine."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, List, Optional, Tuple

import numpy as np
from PIL import Image
from rdkit import Chem

from mollift_annotator.schema import (
    AnnotationRecord,
    AtomAnnotation,
    make_metadata,
)


@dataclass
class HistoryEntry:
    action: str
    atom_index: Optional[int] = None
    pixel: Optional[Tuple[float, float]] = None
    was_skipped: bool = False
    previous_pixel: Optional[Tuple[float, float]] = None


@dataclass
class AnnotationSession:
    smiles: str
    image_path: str
    mol: Chem.Mol = field(init=False, repr=False)
    bindings: dict[int, tuple[float, float]] = field(default_factory=dict)
    skipped: set[int] = field(default_factory=set)
    _history: List[HistoryEntry] = field(default_factory=list, repr=False)
    _image_size: tuple[int, int] = field(default_factory=lambda: (0, 0), repr=False)
    _on_change: Optional[Callable[[], None]] = field(default=None, repr=False)

    def __post_init__(self) -> None:
        mol = Chem.MolFromSmiles(self.smiles)
        if mol is None:
            raise ValueError(f"Invalid SMILES: {self.smiles!r}")
        self.mol = mol
        self._load_image_size()

    def _load_image_size(self) -> None:
        path = Path(self.image_path)
        if not path.is_file():
            raise FileNotFoundError(f"Image not found: {self.image_path}")
        with Image.open(path) as img:
            self._image_size = img.size

    @property
    def num_atoms(self) -> int:
        return self.mol.GetNumAtoms()

    @property
    def image_width(self) -> int:
        return self._image_size[0]

    @property
    def image_height(self) -> int:
        return self._image_size[1]

    def set_on_change(self, callback: Optional[Callable[[], None]]) -> None:
        self._on_change = callback

    def _notify(self) -> None:
        if self._on_change is not None:
            self._on_change()

    def atom_symbol(self, atom_index: int) -> str:
        return self.mol.GetAtomWithIdx(atom_index).GetSymbol()

    def bind(self, atom_index: int, pixel: tuple[float, float]) -> str | None:
        """Bind atom to pixel. Returns warning message if overwriting another atom at same pixel."""
        self._validate_atom_index(atom_index)
        px, py = float(pixel[0]), float(pixel[1])

        if atom_index in self.skipped:
            self.skipped.discard(atom_index)

        previous_pixel = self.bindings.get(atom_index)
        self._history.append(
            HistoryEntry(
                action="bind",
                atom_index=atom_index,
                pixel=(px, py),
                previous_pixel=previous_pixel,
                was_skipped=False,
            )
        )

        warning = None
        for other_idx, other_px in list(self.bindings.items()):
            if other_idx != atom_index and _pixels_close(other_px, (px, py)):
                del self.bindings[other_idx]
                warning = f"Atom {other_idx} unbound (same pixel as atom {atom_index})"

        self.bindings[atom_index] = (px, py)
        self._notify()
        return warning

    def unbind(self, atom_index: int) -> None:
        self._validate_atom_index(atom_index)
        if atom_index not in self.bindings:
            return
        previous_pixel = self.bindings.pop(atom_index)
        self._history.append(
            HistoryEntry(
                action="unbind",
                atom_index=atom_index,
                previous_pixel=previous_pixel,
            )
        )
        self._notify()

    def skip(self, atom_index: int) -> None:
        self._validate_atom_index(atom_index)
        previous_pixel = self.bindings.pop(atom_index, None)
        was_skipped = atom_index in self.skipped
        self.skipped.add(atom_index)
        self._history.append(
            HistoryEntry(
                action="skip",
                atom_index=atom_index,
                previous_pixel=previous_pixel,
                was_skipped=was_skipped,
            )
        )
        self._notify()

    def unskip(self, atom_index: int) -> None:
        self._validate_atom_index(atom_index)
        if atom_index not in self.skipped:
            return
        self.skipped.discard(atom_index)
        self._history.append(
            HistoryEntry(
                action="unskip",
                atom_index=atom_index,
                was_skipped=True,
            )
        )
        self._notify()

    def undo(self) -> bool:
        if not self._history:
            return False
        entry = self._history.pop()
        if entry.action == "bind":
            assert entry.atom_index is not None
            if entry.previous_pixel is None:
                self.bindings.pop(entry.atom_index, None)
            else:
                self.bindings[entry.atom_index] = entry.previous_pixel
        elif entry.action == "unbind":
            assert entry.atom_index is not None and entry.previous_pixel is not None
            self.bindings[entry.atom_index] = entry.previous_pixel
        elif entry.action == "skip":
            assert entry.atom_index is not None
            self.skipped.discard(entry.atom_index)
            if entry.previous_pixel is not None:
                self.bindings[entry.atom_index] = entry.previous_pixel
        elif entry.action == "unskip":
            assert entry.atom_index is not None
            self.skipped.add(entry.atom_index)
        self._notify()
        return True

    def annotated_count(self) -> int:
        return len(self.bindings)

    def progress_text(self) -> str:
        return f"Annotated {self.annotated_count()} / {self.num_atoms}"

    def to_record(self) -> AnnotationRecord:
        mask = [False] * self.num_atoms
        annotations: list[AtomAnnotation] = []
        for idx in sorted(self.bindings):
            mask[idx] = True
            px, py = self.bindings[idx]
            annotations.append(
                AtomAnnotation(
                    atom_index=idx,
                    symbol=self.atom_symbol(idx),
                    pixel=(px, py),
                )
            )
        return AnnotationRecord(
            smiles=self.smiles,
            image_path=str(Path(self.image_path).as_posix()),
            image_width=self.image_width,
            image_height=self.image_height,
            num_atoms=self.num_atoms,
            annotated_mask=mask,
            annotations=annotations,
            skipped_atoms=sorted(self.skipped),
            metadata=make_metadata(),
        )

    def load_record(self, record: AnnotationRecord) -> None:
        if record.smiles != self.smiles:
            raise ValueError("SMILES mismatch when loading annotation")
        if record.image_width != self.image_width or record.image_height != self.image_height:
            raise ValueError("Image size mismatch when loading annotation")

        self.bindings.clear()
        self.skipped = set(record.skipped_atoms)
        for ann in record.annotations:
            self.bindings[ann.atom_index] = ann.pixel
        self._history.clear()
        self._notify()

    def _validate_atom_index(self, atom_index: int) -> None:
        if atom_index < 0 or atom_index >= self.num_atoms:
            raise ValueError(f"Invalid atom_index: {atom_index}")


def _pixels_close(a: tuple[float, float], b: tuple[float, float], tol: float = 1.0) -> bool:
    return abs(a[0] - b[0]) <= tol and abs(a[1] - b[1]) <= tol
