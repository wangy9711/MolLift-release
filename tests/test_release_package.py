import hashlib
import json
import zipfile

import pytest

from scripts.package_release import package


def test_bundle_contains_only_released_checkpoints(tmp_path):
    root = tmp_path / "source"
    (root / "mollift").mkdir(parents=True)
    (root / "checkpoints").mkdir()
    (root / "README.md").write_text("# MolLift\n")
    registry = {}
    for name in ("mollift-drugs", "mollift-peptide"):
        payload = name.encode()
        filename = f"{name}.pt"
        (root / "checkpoints" / filename).write_bytes(payload)
        entry = {"filename": filename, "sha256": hashlib.sha256(payload).hexdigest()}
        registry[name] = entry
        (root / "checkpoints" / f"{name}.json").write_text(json.dumps(entry))
    (root / "mollift/models.json").write_text(json.dumps(registry))
    for filename in ("latest.pt", "debug.cpt", "extra.safetensors"):
        (root / "checkpoints" / filename).write_bytes(b"not released")

    destination = tmp_path / "release"
    package(root, destination)
    assert {p.name for p in destination.iterdir()} == {
        "mollift.zip", "release_manifest.json", "SHA256SUMS"
    }
    manifest = json.loads((destination / "release_manifest.json").read_text())
    with zipfile.ZipFile(destination / "mollift.zip") as archive:
        assert set(archive.namelist()) == {
            "MolLift/README.md", "MolLift/mollift/models.json",
            "MolLift/checkpoints/mollift-drugs.pt", "MolLift/checkpoints/mollift-drugs.json",
            "MolLift/checkpoints/mollift-peptide.pt", "MolLift/checkpoints/mollift-peptide.json",
        }
        for filename, digest in manifest["source_files_sha256"].items():
            assert hashlib.sha256(archive.read(f"MolLift/{filename}")).hexdigest() == digest
    digest = hashlib.sha256((destination / "mollift.zip").read_bytes()).hexdigest()
    assert (destination / "SHA256SUMS").read_text() == f"{digest}  mollift.zip\n"
    with pytest.raises(FileExistsError):
        package(root, destination)


def test_bundle_rejects_missing_checkpoint(tmp_path):
    root = tmp_path / "source"
    (root / "mollift").mkdir(parents=True)
    (root / "mollift/models.json").write_text(json.dumps({
        "mollift-drugs": {"filename": "mollift-drugs.pt", "sha256": "missing"}
    }))
    with pytest.raises(ValueError, match="Missing/mismatched model"):
        package(root, tmp_path / "release")
