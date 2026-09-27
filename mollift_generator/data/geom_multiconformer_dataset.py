"""GEOM molecule-level caches with epoch-dependent conformer sampling."""

from __future__ import annotations

from bisect import bisect_right
from pathlib import Path
from typing import Literal

import numpy as np
from torch.utils.data import Dataset

from mollift_generator.data.geom_cache import (
    GEOMRecord,
    Split,
    load_cached_molecule,
    read_records,
)
from mollift_generator.data.projection import ProjectionMode, synthetic_projection

ConformerSampling = Literal["uniform", "boltzmann", "all"]


class GEOMMultiConformerDataset(Dataset):
    """Read preprocessed molecule-level GEOM caches.

    ``uniform`` and ``boltzmann`` expose one sample per molecule per epoch. ``all``
    exposes every conformer and is primarily intended for deterministic evaluation.
    Call ``set_epoch`` before each epoch for reproducible resampling.
    """

    def __init__(
        self,
        cache_root: str | Path,
        split: Split = "train",
        *,
        conformer_sampling: ConformerSampling = "uniform",
        projection_mode: ProjectionMode = "synthetic",
        coord_units: Literal["angstrom", "normalized"] = "angstrom",
        seed: int = 42,
        molecule_limit: int | None = None,
    ):
        self.cache_root = Path(cache_root)
        self.records: list[GEOMRecord] = read_records(self.cache_root, split)
        if molecule_limit is not None:
            if molecule_limit <= 0:
                raise ValueError("molecule_limit must be positive")
            self.records = self.records[:molecule_limit]
        if not self.records:
            raise ValueError(f"No GEOM cache records for split={split} at {cache_root}")
        if conformer_sampling not in {"uniform", "boltzmann", "all"}:
            raise ValueError(f"Unknown conformer sampling mode: {conformer_sampling}")
        self.conformer_sampling = conformer_sampling
        self.projection_mode = projection_mode
        self.coord_units = coord_units
        self.seed = int(seed)
        self.epoch = 0
        self._cumulative: list[int] | None = None
        if conformer_sampling == "all":
            cumulative = []
            total = 0
            for record in self.records:
                total += record.n_conformers
                cumulative.append(total)
            self._cumulative = cumulative

    @property
    def num_molecules(self) -> int:
        return len(self.records)

    @property
    def num_conformers(self) -> int:
        return sum(record.n_conformers for record in self.records)

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def __len__(self) -> int:
        if self._cumulative is not None:
            return self._cumulative[-1]
        return len(self.records)

    def _locate(self, index: int) -> tuple[int, int | None]:
        if index < 0:
            index += len(self)
        if index < 0 or index >= len(self):
            raise IndexError(index)
        if self._cumulative is None:
            return index, None
        molecule_index = bisect_right(self._cumulative, index)
        previous = 0 if molecule_index == 0 else self._cumulative[molecule_index - 1]
        return molecule_index, index - previous

    def _rng(self, molecule_index: int, conformer_index: int = 0) -> np.random.Generator:
        sequence = np.random.SeedSequence(
            [self.seed, self.epoch, molecule_index, conformer_index]
        )
        return np.random.default_rng(sequence)

    @staticmethod
    def _sample_boltzmann(weights: np.ndarray, rng: np.random.Generator) -> int:
        weights = np.asarray(weights, dtype=np.float64)
        valid = np.isfinite(weights) & (weights >= 0)
        clean = np.where(valid, weights, 0.0)
        total = float(clean.sum())
        if total <= 0:
            return int(rng.integers(len(weights)))
        return int(rng.choice(len(weights), p=clean / total))

    def get_conformer(
        self,
        molecule_index: int,
        conformer_index: int,
        *,
        _payload: dict | None = None,
    ) -> dict:
        record = self.records[molecule_index]
        payload = _payload if _payload is not None else load_cached_molecule(record.cache_path)
        n_conformers = int(payload["conformer_coords"].shape[0])
        if conformer_index < 0 or conformer_index >= n_conformers:
            raise IndexError(conformer_index)

        rng = self._rng(molecule_index, conformer_index)
        x3d_raw = np.asarray(payload["conformer_coords"][conformer_index], dtype=np.float32)
        # Synthetic projections use coordinates; the cache does not store RDKit Mol objects.
        if self.projection_mode != "synthetic":
            raise NotImplementedError(
                "GEOM cache currently supports synthetic projections only; "
                "reconstruct an RDKit Mol for rdkit_2d if needed later"
            )
        x3d, pos_2d = synthetic_projection(
            x3d_raw,
            rng=rng,
            coord_units=self.coord_units,
        )
        return {
            "smiles": payload["smiles"],
            "canonical_smiles": payload["canonical_smiles"],
            "molecule_id": payload["molecule_id"],
            "atom_types": payload["atom_types"],
            "atom_features": payload["atom_features"],
            "bond_index": payload["bond_index"],
            "bond_features": payload["bond_features"],
            "bond_features_dense": payload["bond_features_dense"],
            "shortest_path": payload["shortest_path"],
            "prior_node_attr": payload["prior_node_attr"],
            "prior_evecs": payload["prior_evecs"],
            "x3d": x3d,
            "pos_2d": pos_2d,
            "n_atoms": int(payload["n_atoms"]),
            "sample_key": molecule_index * 1_000_000 + conformer_index,
            "molecule_index": molecule_index,
            "conformer_index": conformer_index,
            "relative_energy": float(payload["relative_energy"][conformer_index]),
            "total_energy": float(payload["total_energy"][conformer_index]),
            "boltzmann_weight": float(payload["boltzmann_weight"][conformer_index]),
            "degeneracy": float(payload["degeneracy"][conformer_index]),
            "geom_id": payload["geom_ids"][conformer_index],
        }

    def __getitem__(self, index: int) -> dict:
        molecule_index, conformer_index = self._locate(index)
        payload = None
        if conformer_index is None:
            record = self.records[molecule_index]
            rng = self._rng(molecule_index)
            if self.conformer_sampling == "uniform":
                conformer_index = int(rng.integers(record.n_conformers))
            else:
                payload = load_cached_molecule(record.cache_path)
                conformer_index = self._sample_boltzmann(payload["boltzmann_weight"], rng)
        return self.get_conformer(molecule_index, conformer_index, _payload=payload)
