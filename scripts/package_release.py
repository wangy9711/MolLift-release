"""Bundle code and pretrained checkpoints in one ZIP with a checksum manifest."""

import argparse
import hashlib
import json
import re
from pathlib import Path
import zipfile


def digest(path):
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def package(root: Path, destination: Path):
    root, destination = root.resolve(), destination.resolve()
    if destination == root or root in destination.parents:
        raise ValueError("Choose an output directory outside the source repository")
    destination.mkdir(parents=True, exist_ok=True)
    names = ["mollift.zip", "release_manifest.json", "SHA256SUMS"]
    for name in names:
        if (destination / name).exists():
            raise FileExistsError(destination / name)
    registry = json.loads((root / "mollift/models.json").read_text())
    for model in registry.values():
        checkpoint = root / "checkpoints" / model["filename"]
        if not checkpoint.is_file() or digest(checkpoint) != model["sha256"]:
            raise ValueError(f"Missing/mismatched model: {checkpoint}")
    allowed_dirs = {"mollift", "mollift_generator", "mollift_solver", "mollift_refiner",
                    "mollift_annotator", "configs", "docs", "examples", "scripts", "tests", "checkpoints"}
    allowed_root = {"README.md", "LICENSE", "CITATION.cff",
                    "pyproject.toml", ".gitignore"}
    released_weights = {Path("checkpoints") / model["filename"] for model in registry.values()}
    source_files = []
    for path in sorted(root.rglob("*")):
        if not path.is_file():
            continue
        relative = path.relative_to(root)
        if "__pycache__" in relative.parts or path.suffix == ".pyc":
            continue
        if path.suffix in {".pt", ".cpt", ".safetensors"} and relative not in released_weights:
            continue
        if (len(relative.parts) == 1 and relative.name in allowed_root) or (
            len(relative.parts) > 1 and relative.parts[0] in allowed_dirs
        ):
            source_files.append(path)
    # Catch accidental local-machine paths or experiment archives in public text.
    private_path = re.compile(rb"/(?:home|Users|data)/[A-Za-z0-9_.-]+/")
    for path in source_files:
        if path.suffix in {".py", ".md", ".yaml", ".json", ".toml", ".sh"}:
            if private_path.search(path.read_bytes()):
                raise ValueError(f"Private path in public file: {path}")
    files_manifest = {str(p.relative_to(root)): digest(p) for p in source_files}
    with zipfile.ZipFile(destination / names[0], "w", zipfile.ZIP_DEFLATED) as archive:
        for path in source_files:
            archive.write(path, "MolLift/" + str(path.relative_to(root)))
    archives = {name: {"sha256": digest(destination / name), "size_bytes": (destination / name).stat().st_size}
                for name in names[:1]}
    manifest = {"source_file_count": len(source_files), "source_files_sha256": files_manifest,
                "models": registry, "archives": archives}
    (destination / names[1]).write_text(json.dumps(manifest, indent=2) + "\n")
    (destination / names[2]).write_text("".join(f"{row['sha256']}  {name}\n" for name, row in archives.items()))
    print(json.dumps({"source_file_count": len(source_files), "archives": archives}, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    package(Path(__file__).resolve().parents[1], args.output)
