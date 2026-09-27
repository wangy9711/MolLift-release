from pathlib import Path

import pytest
from PIL import Image

from mollift_annotator.schema import AnnotationRecord, AtomAnnotation
from mollift_annotator.session import AnnotationSession


@pytest.fixture
def image_path(tmp_path: Path) -> str:
    path = tmp_path / "proj.png"
    Image.new("RGB", (400, 300), color=(240, 240, 240)).save(path)
    return str(path)


def test_bind_and_unbind(image_path: str):
    session = AnnotationSession(smiles="CCO", image_path=image_path)
    assert session.num_atoms == 3

    session.bind(0, (10.0, 20.0))
    session.bind(1, (100.0, 50.0))
    assert session.annotated_count() == 2

    record = session.to_record()
    assert record.annotated_mask == [True, True, False]

    session.unbind(0)
    assert session.annotated_count() == 1
    assert 0 not in session.bindings


def test_skip_atom(image_path: str):
    session = AnnotationSession(smiles="CCO", image_path=image_path)
    session.bind(0, (1.0, 2.0))
    session.skip(0)
    assert 0 not in session.bindings
    assert 0 in session.skipped

    record = session.to_record()
    assert 0 not in [a.atom_index for a in record.annotations]
    assert 0 in record.skipped_atoms


def test_undo_bind(image_path: str):
    session = AnnotationSession(smiles="CCO", image_path=image_path)
    session.bind(0, (5.0, 6.0))
    session.bind(0, (7.0, 8.0))
    assert session.bindings[0] == (7.0, 8.0)
    session.undo()
    assert session.bindings[0] == (5.0, 6.0)
    session.undo()
    assert 0 not in session.bindings


def test_load_record(image_path: str):
    session = AnnotationSession(smiles="CCO", image_path=image_path)
    record = AnnotationRecord(
        smiles="CCO",
        image_path="proj.png",
        image_width=400,
        image_height=300,
        num_atoms=3,
        annotated_mask=[True, False, False],
        annotations=[AtomAnnotation(atom_index=0, symbol="C", pixel=(12.0, 34.0))],
        skipped_atoms=[2],
    )
    session.load_record(record)
    assert session.bindings[0] == (12.0, 34.0)
    assert 2 in session.skipped


def test_same_pixel_unbinds_other(image_path: str):
    session = AnnotationSession(smiles="CCO", image_path=image_path)
    session.bind(0, (50.0, 60.0))
    warning = session.bind(1, (50.0, 60.0))
    assert 0 not in session.bindings
    assert 1 in session.bindings
    assert warning is not None
