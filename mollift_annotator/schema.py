"""JSON schema v1 for 2D atom annotations."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, List, Optional

from rdkit import Chem

FORMAT_VERSION = "1.0"
TOOL_NAME = "mollift_annotator"


@dataclass
class AtomAnnotation:
    atom_index: int
    symbol: str
    pixel: tuple[float, float]

    def to_dict(self) -> dict[str, Any]:
        return {
            "atom_index": self.atom_index,
            "symbol": self.symbol,
            "pixel": [float(self.pixel[0]), float(self.pixel[1])],
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "AtomAnnotation":
        px, py = data["pixel"]
        return cls(
            atom_index=int(data["atom_index"]),
            symbol=str(data["symbol"]),
            pixel=(float(px), float(py)),
        )


@dataclass
class AnnotationRecord:
    smiles: str
    image_path: str
    image_width: int
    image_height: int
    num_atoms: int
    annotated_mask: List[bool]
    annotations: List[AtomAnnotation] = field(default_factory=list)
    skipped_atoms: List[int] = field(default_factory=list)
    format_version: str = FORMAT_VERSION
    metadata: dict[str, Any] = field(default_factory=dict)

    def validate(self) -> None:
        if self.format_version != FORMAT_VERSION:
            raise ValueError(f"Unsupported format_version: {self.format_version!r}")

        mol = Chem.MolFromSmiles(self.smiles)
        if mol is None:
            raise ValueError(f"Invalid SMILES: {self.smiles!r}")

        n = mol.GetNumAtoms()
        if self.num_atoms != n:
            raise ValueError(f"num_atoms {self.num_atoms} != mol atom count {n}")
        if len(self.annotated_mask) != n:
            raise ValueError(f"annotated_mask length {len(self.annotated_mask)} != {n}")

        seen: set[int] = set()
        for ann in self.annotations:
            if ann.atom_index < 0 or ann.atom_index >= n:
                raise ValueError(f"Invalid atom_index: {ann.atom_index}")
            if ann.atom_index in seen:
                raise ValueError(f"Duplicate atom_index in annotations: {ann.atom_index}")
            seen.add(ann.atom_index)
            if not self.annotated_mask[ann.atom_index]:
                raise ValueError(
                    f"atom_index {ann.atom_index} in annotations but annotated_mask is False"
                )

        for idx in self.skipped_atoms:
            if idx < 0 or idx >= n:
                raise ValueError(f"Invalid skipped atom index: {idx}")
            if self.annotated_mask[idx]:
                raise ValueError(f"skipped atom {idx} is marked annotated in mask")

        for i, flagged in enumerate(self.annotated_mask):
            if flagged and i not in seen:
                raise ValueError(f"annotated_mask[{i}] is True but no annotation entry")

    def to_dict(self) -> dict[str, Any]:
        return {
            "format_version": self.format_version,
            "smiles": self.smiles,
            "image": {
                "path": self.image_path,
                "width": self.image_width,
                "height": self.image_height,
            },
            "num_atoms": self.num_atoms,
            "annotated_mask": list(self.annotated_mask),
            "annotations": [a.to_dict() for a in self.annotations],
            "skipped_atoms": sorted(self.skipped_atoms),
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "AnnotationRecord":
        image = data["image"]
        record = cls(
            format_version=str(data.get("format_version", FORMAT_VERSION)),
            smiles=str(data["smiles"]),
            image_path=str(image["path"]),
            image_width=int(image["width"]),
            image_height=int(image["height"]),
            num_atoms=int(data["num_atoms"]),
            annotated_mask=[bool(x) for x in data["annotated_mask"]],
            annotations=[AtomAnnotation.from_dict(a) for a in data.get("annotations", [])],
            skipped_atoms=[int(x) for x in data.get("skipped_atoms", [])],
            metadata=dict(data.get("metadata", {})),
        )
        record.validate()
        return record


def save_record(path: str | Path, record: AnnotationRecord) -> None:
    record.validate()
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(record.to_dict(), f, indent=2, ensure_ascii=False)
        f.write("\n")


def load_record(path: str | Path) -> AnnotationRecord:
    path = Path(path)
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    return AnnotationRecord.from_dict(data)


def make_metadata(**extra: Any) -> dict[str, Any]:
    meta = {
        "tool": TOOL_NAME,
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    meta.update(extra)
    return meta
