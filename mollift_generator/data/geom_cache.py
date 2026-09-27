"""Molecule-level cache and manifest utilities for GEOM-QM9/GEOM-Drugs."""

from __future__ import annotations

import hashlib
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Literal

import torch

GEOM_CACHE_VERSION = 1
Split = Literal["train", "valid", "test_pool", "test", "unused"]


@dataclass(frozen=True)
class GEOMRecord:
    molecule_id: str
    split: Split
    cache_path: str
    smiles: str
    source_path: str
    n_atoms: int
    n_conformers: int
    atom_feature_dim: int


def molecule_id(smiles: str) -> str:
    """Stable identifier independent of source filename escaping."""
    return hashlib.sha256(smiles.encode("utf-8")).hexdigest()


def cache_path_for(cache_root: str | Path, molecule_key: str) -> Path:
    root = Path(cache_root)
    return root / "molecules" / molecule_key[:2] / f"{molecule_key}.pt"


def manifest_path(cache_root: str | Path) -> Path:
    return Path(cache_root) / "index.sqlite"


def connect_manifest(cache_root: str | Path) -> sqlite3.Connection:
    path = manifest_path(cache_root)
    path.parent.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(path)
    con.execute(
        """
        CREATE TABLE IF NOT EXISTS molecules (
            molecule_id TEXT PRIMARY KEY,
            split TEXT NOT NULL,
            cache_path TEXT NOT NULL,
            smiles TEXT NOT NULL,
            source_path TEXT NOT NULL,
            n_atoms INTEGER NOT NULL,
            n_conformers INTEGER NOT NULL,
            atom_feature_dim INTEGER NOT NULL,
            cache_version INTEGER NOT NULL
        )
        """
    )
    con.execute("CREATE INDEX IF NOT EXISTS idx_molecules_split ON molecules(split)")
    con.commit()
    return con


def upsert_records(con: sqlite3.Connection, records: Iterable[GEOMRecord]) -> None:
    con.executemany(
        """
        INSERT INTO molecules (
            molecule_id, split, cache_path, smiles, source_path, n_atoms,
            n_conformers, atom_feature_dim, cache_version
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(molecule_id) DO UPDATE SET
            split=excluded.split,
            cache_path=excluded.cache_path,
            smiles=excluded.smiles,
            source_path=excluded.source_path,
            n_atoms=excluded.n_atoms,
            n_conformers=excluded.n_conformers,
            atom_feature_dim=excluded.atom_feature_dim,
            cache_version=excluded.cache_version
        """,
        [
            (
                record.molecule_id,
                record.split,
                record.cache_path,
                record.smiles,
                record.source_path,
                record.n_atoms,
                record.n_conformers,
                record.atom_feature_dim,
                GEOM_CACHE_VERSION,
            )
            for record in records
        ],
    )
    con.commit()


def read_records(cache_root: str | Path, split: Split) -> list[GEOMRecord]:
    con = sqlite3.connect(manifest_path(cache_root))
    try:
        rows = con.execute(
            """
            SELECT molecule_id, split, cache_path, smiles, source_path, n_atoms,
                   n_conformers, atom_feature_dim
            FROM molecules
            WHERE split = ? AND cache_version = ?
            ORDER BY molecule_id
            """,
            (split, GEOM_CACHE_VERSION),
        ).fetchall()
    finally:
        con.close()
    return [GEOMRecord(*row) for row in rows]


def load_cached_molecule(path: str | Path) -> dict:
    payload = torch.load(Path(path), map_location="cpu", weights_only=False)
    version = int(payload.get("cache_version", -1))
    if version != GEOM_CACHE_VERSION:
        raise ValueError(
            f"Unsupported GEOM cache version {version} in {path}; "
            f"expected {GEOM_CACHE_VERSION}"
        )
    return payload
