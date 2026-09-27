"""MolLift_annotator: desktop tool for 2D atom position annotation."""

from __future__ import annotations

from typing import Optional

from mollift_annotator.export import (
    annotations_to_pixel_array,
    load_annotation,
    save_annotation,
)
from mollift_annotator.schema import AnnotationRecord, AtomAnnotation
from mollift_annotator.session import AnnotationSession


def run_gui(smiles: str, image_path: str, output_path: Optional[str] = None) -> None:
    """Launch PySide6 annotation GUI (lazy import)."""
    from mollift_annotator.gui.app import run_gui as _run_gui

    _run_gui(smiles, image_path, output_path)


__all__ = [
    "run_gui",
    "AnnotationSession",
    "AnnotationRecord",
    "AtomAnnotation",
    "load_annotation",
    "save_annotation",
    "annotations_to_pixel_array",
]
