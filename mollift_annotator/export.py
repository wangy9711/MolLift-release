"""Load/save helpers and numpy conversion."""

from __future__ import annotations

from pathlib import Path
from typing import Tuple

import numpy as np

from mollift_annotator.schema import AnnotationRecord, load_record, save_record


def load_annotation(path: str | Path) -> AnnotationRecord:
    return load_record(path)


def save_annotation(path: str | Path, record: AnnotationRecord) -> None:
    save_record(path, record)


def annotations_to_pixel_array(
    record: AnnotationRecord,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Convert record to dense arrays aligned with RDKit atom order.

    Returns
    -------
    coords : (M, 2) float64, NaN for unannotated atoms
    mask : (M,) bool, True where pixel is annotated
    """
    n = record.num_atoms
    coords = np.full((n, 2), np.nan, dtype=np.float64)
    mask = np.zeros(n, dtype=bool)
    for ann in record.annotations:
        coords[ann.atom_index, 0] = ann.pixel[0]
        coords[ann.atom_index, 1] = ann.pixel[1]
        mask[ann.atom_index] = True
    return coords, mask
