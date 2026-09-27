import json
from pathlib import Path

import numpy as np
import pytest

from mollift_annotator.export import annotations_to_pixel_array, load_annotation, save_annotation
from mollift_annotator.schema import AnnotationRecord, AtomAnnotation


def _sample_record() -> AnnotationRecord:
    return AnnotationRecord(
        smiles="CCO",
        image_path="test.png",
        image_width=800,
        image_height=600,
        num_atoms=3,
        annotated_mask=[True, True, False],
        annotations=[
            AtomAnnotation(atom_index=0, symbol="C", pixel=(120.5, 340.2)),
            AtomAnnotation(atom_index=1, symbol="C", pixel=(210.0, 338.0)),
        ],
        skipped_atoms=[2],
        metadata={"tool": "mollift_annotator"},
    )


def test_record_roundtrip_dict():
    record = _sample_record()
    restored = AnnotationRecord.from_dict(record.to_dict())
    assert restored.smiles == record.smiles
    assert restored.annotated_mask == record.annotated_mask
    assert len(restored.annotations) == 2
    assert restored.skipped_atoms == [2]


def test_save_load_json(tmp_path: Path):
    record = _sample_record()
    path = tmp_path / "ann.json"
    save_annotation(path, record)
    loaded = load_annotation(path)
    assert loaded.smiles == "CCO"
    assert loaded.annotations[0].pixel == (120.5, 340.2)


def test_annotations_to_pixel_array():
    record = _sample_record()
    coords, mask = annotations_to_pixel_array(record)
    assert coords.shape == (3, 2)
    assert mask.shape == (3,)
    assert mask.tolist() == [True, True, False]
    assert coords[0, 0] == pytest.approx(120.5)
    assert np.isnan(coords[2, 0])


def test_invalid_atom_index_rejected():
    record = _sample_record()
    record.annotations.append(AtomAnnotation(atom_index=99, symbol="C", pixel=(1.0, 2.0)))
    with pytest.raises(ValueError, match="Invalid atom_index"):
        record.validate()


def test_mask_mismatch_rejected():
    record = _sample_record()
    record.annotated_mask = [True, False, False]
    with pytest.raises(ValueError, match="annotated_mask"):
        record.validate()


def test_json_format_keys(tmp_path: Path):
    record = _sample_record()
    path = tmp_path / "ann.json"
    save_annotation(path, record)
    data = json.loads(path.read_text())
    assert data["format_version"] == "1.0"
    assert "image" in data
    assert data["image"]["width"] == 800
