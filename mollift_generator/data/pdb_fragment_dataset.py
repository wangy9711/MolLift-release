"""PDB peptide-fragment package dataset and transfer-training batch sampler."""

from __future__ import annotations

import math
import random
from collections import OrderedDict
from pathlib import Path
from typing import Iterator, Sequence

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, Sampler

from mollift_generator.data.projection import synthetic_projection


class PDBFragmentDataset(Dataset):
    """Read aligned coordinate and graph-feature shards from a peptide package."""

    def __init__(
        self,
        root: str | Path,
        split: str,
        *,
        seed: int = 42,
        coord_units: str = "angstrom",
        limit_per_stratum: int | None = None,
        shard_cache_size: int = 2,
    ) -> None:
        self.root = Path(root)
        manifest = pd.read_parquet(self.root / "manifests" / "fragments.parquet")
        manifest = manifest[manifest.split == split].sort_values("fragment_id")
        if limit_per_stratum is not None:
            if limit_per_stratum <= 0:
                raise ValueError("limit_per_stratum must be positive")
            manifest = manifest.groupby(
                ["length", "secondary_structure_class"], sort=True, group_keys=False
            ).head(limit_per_stratum)
        if manifest.empty:
            raise ValueError(f"no PDB fragments for split={split} in {root}")
        self.manifest = manifest.reset_index(drop=True)
        self.split = split
        self.seed = int(seed)
        self.coord_units = coord_units
        self.epoch = 0
        self.shard_cache_size = int(shard_cache_size)
        self._base_cache: OrderedDict[str, list[dict]] = OrderedDict()
        self._derived_cache: OrderedDict[str, list[dict]] = OrderedDict()

    @property
    def n_atoms(self) -> np.ndarray:
        return self.manifest.n_atoms.to_numpy(dtype=np.int64)

    @property
    def lengths(self) -> np.ndarray:
        return self.manifest.length.to_numpy(dtype=np.int64)

    @property
    def structure_classes(self) -> np.ndarray:
        return self.manifest.secondary_structure_class.to_numpy()

    @property
    def shards(self) -> np.ndarray:
        return self.manifest.shard.to_numpy()

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def __len__(self) -> int:
        return len(self.manifest)

    def _load(self, relative: str, *, derived: bool) -> list[dict]:
        cache = self._derived_cache if derived else self._base_cache
        key = relative
        if key in cache:
            value = cache.pop(key)
            cache[key] = value
            return value
        path = self.root / relative
        if derived:
            path = self.root / "derived_cache" / "graph_v1" / Path(relative).relative_to("fragments")
        value = torch.load(path, map_location="cpu", weights_only=False)
        cache[key] = value
        while len(cache) > self.shard_cache_size:
            cache.popitem(last=False)
        return value

    @staticmethod
    def _dense_bonds(record: dict) -> np.ndarray:
        n = len(record["atomic_numbers"])
        dense = np.zeros((n, n, 5), dtype=np.float32)
        dense[..., 4] = 1.0
        channel = {1.0: 0, 2.0: 1, 3.0: 2, 1.5: 3}
        for (left, right), bond_type in zip(record["bond_index"].T, record["bond_type"]):
            left, right = int(left), int(right)
            dense[left, right] = dense[right, left] = 0.0
            dense[left, right, channel.get(float(bond_type), 4)] = 1.0
            dense[right, left, channel.get(float(bond_type), 4)] = 1.0
        return dense

    def __getitem__(self, index: int) -> dict:
        row = self.manifest.iloc[index]
        record = self._load(row.shard, derived=False)[int(row.shard_index)]
        derived = self._load(row.shard, derived=True)[int(row.shard_index)]
        if record["fragment_id"] != row.fragment_id or derived["fragment_id"] != row.fragment_id:
            raise RuntimeError(f"base/derived/manifest mismatch at dataset index {index}")
        sample_key = int(row.fragment_id[:15], 16)
        rng = np.random.default_rng(np.random.SeedSequence([self.seed, self.epoch, sample_key]))
        x3d, pos_2d = synthetic_projection(
            np.asarray(record["coordinates"], dtype=np.float32),
            rng=rng,
            coord_units=self.coord_units,
        )
        return {
            "smiles": record["smiles"], "fragment_id": row.fragment_id,
            "atom_types": np.asarray(record["atomic_numbers"], dtype=np.int64),
            "atom_features": np.asarray(record["atom_features"], dtype=np.float32),
            "bond_features_dense": self._dense_bonds(record),
            "shortest_path": np.asarray(derived["shortest_path"], dtype=np.int64),
            "prior_node_attr": np.asarray(derived["prior_node_attr"], dtype=np.float32),
            "prior_evecs": np.asarray(derived["prior_evecs"], dtype=np.float32),
            "x3d": x3d, "pos_2d": pos_2d, "n_atoms": int(row.n_atoms),
            "sample_key": sample_key,
            "experimental_atom_mask": np.asarray(record["experimental_atom_mask"], dtype=bool),
            "cap_atom_mask": np.asarray(record["cap_atom_mask"], dtype=bool),
            "fragment_length": int(row.length),
            "secondary_structure_class": row.secondary_structure_class,
            "pdb_id": row.pdb_id,
            "chain_uid": row.chain_uid,
            "sequence": row.sequence,
            "release_date": str(row.release_date),
            "atom_mapping": record["atom_mapping"],
            "phi_degrees": record.get("phi_degrees", []),
            "psi_degrees": record.get("psi_degrees", []),
            "omega_degrees": record.get("omega_degrees", []),
        }


class PDBDrugsReplayDataset(Dataset):
    """Index-compatible union used by the bucketed replay sampler."""

    def __init__(self, pdb: PDBFragmentDataset, drugs: Dataset) -> None:
        self.pdb = pdb
        self.drugs = drugs

    def __len__(self) -> int:
        return len(self.pdb) + len(self.drugs)

    def __getitem__(self, index: int) -> dict:
        if index < len(self.pdb):
            sample = self.pdb[index]
            sample["data_source"] = "pdb"
            return sample
        sample = self.drugs[index - len(self.pdb)]
        sample["data_source"] = "geom_drugs"
        return sample

    def set_epoch(self, epoch: int) -> None:
        self.pdb.set_epoch(epoch)
        if hasattr(self.drugs, "set_epoch"):
            self.drugs.set_epoch(epoch)


class TransferBucketBatchSampler(Sampler[list[int]]):
    """Length/structure-balanced PDB sampling with GEOM-Drugs replay and atom buckets."""

    def __init__(
        self,
        pdb: PDBFragmentDataset,
        drug_n_atoms: Sequence[int],
        *,
        pdb_samples_per_epoch: int,
        pdb_fraction: float,
        bucket_batch_sizes: Sequence[tuple[int, int]],
        seed: int = 42,
        curriculum: Sequence[tuple[int, int]] = (),
    ) -> None:
        if not 0 < pdb_fraction < 1:
            raise ValueError("pdb_fraction must be between zero and one")
        self.pdb = pdb
        self.drug_n_atoms = np.asarray(drug_n_atoms, dtype=np.int64)
        self.pdb_samples_per_epoch = int(pdb_samples_per_epoch)
        self.pdb_fraction = float(pdb_fraction)
        self.bucket_batch_sizes = sorted((int(upper), int(size)) for upper, size in bucket_batch_sizes)
        self.seed = int(seed)
        self.curriculum = sorted((int(epoch), int(length)) for epoch, length in curriculum)
        self.epoch = 0
        self._cached_batches: list[list[int]] | None = None

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)
        self._cached_batches = None

    def _max_length(self) -> int:
        maximum = 50
        for until_epoch, length in self.curriculum:
            if self.epoch < until_epoch:
                return length
        return maximum

    def _draw_pdb(self, rng: random.Random) -> list[int]:
        maximum_length = self._max_length()
        classes = {"helix": 0.20, "sheet": 0.20, "coil": 0.30, "mixed": 0.30}
        lengths = list(range(5, maximum_length + 1))
        groups: dict[tuple[int, str], list[int]] = {}
        for length in lengths:
            for label in classes:
                mask = (self.pdb.lengths == length) & (self.pdb.structure_classes == label)
                groups[(length, label)] = np.flatnonzero(mask).tolist()
        raw = {(length, label): self.pdb_samples_per_epoch * weight / len(lengths)
               for length in lengths for label, weight in classes.items()}
        counts = {key: int(value) for key, value in raw.items()}
        remainder = self.pdb_samples_per_epoch - sum(counts.values())
        for key in sorted(raw, key=lambda item: (-(raw[item] - counts[item]), item))[:remainder]:
            counts[key] += 1
        selected = []
        for key, count in counts.items():
            pool = groups[key]
            if not pool:
                continue
            shuffled = pool.copy(); rng.shuffle(shuffled)
            selected.extend(shuffled[i % len(shuffled)] for i in range(count))
        return selected

    def _batch_size(self, n_atoms: int) -> int:
        for upper, size in self.bucket_batch_sizes:
            if n_atoms <= upper:
                return size
        raise ValueError(f"no atom bucket configured for {n_atoms} atoms")

    def _make_batches(self) -> list[list[int]]:
        rng = random.Random(self.seed + self.epoch)
        pdb_indices = self._draw_pdb(rng)
        n_drugs = round(len(pdb_indices) * (1.0 - self.pdb_fraction) / self.pdb_fraction)
        drug_pool = list(range(len(self.drug_n_atoms)))
        rng.shuffle(drug_pool)
        drug_indices = [drug_pool[i % len(drug_pool)] + len(self.pdb) for i in range(n_drugs)]
        pdb_buckets: dict[int, list[int]] = {upper: [] for upper, _ in self.bucket_batch_sizes}
        drug_buckets: dict[int, list[int]] = {upper: [] for upper, _ in self.bucket_batch_sizes}
        for index in pdb_indices:
            n_atoms = int(self.pdb.n_atoms[index])
            upper = next(upper for upper, _ in self.bucket_batch_sizes if n_atoms <= upper)
            pdb_buckets[upper].append(index)
        for index in drug_indices:
            n_atoms = int(self.drug_n_atoms[index - len(self.pdb)])
            upper = next(upper for upper, _ in self.bucket_batch_sizes if n_atoms <= upper)
            drug_buckets[upper].append(index)
        batches = []
        for upper, _ in self.bucket_batch_sizes:
            size = self._batch_size(upper)
            # Keep PDB records from the same on-disk shard adjacent inside a batch,
            # then randomize batch order. This avoids pathological shard-cache thrash.
            pdb_grouped = sorted(
                pdb_buckets[upper],
                key=lambda index: (self.pdb.shards[index], rng.random()),
            )
            batches.extend(pdb_grouped[offset:offset + size] for offset in range(0, len(pdb_grouped), size))
            drug_group = drug_buckets[upper]; rng.shuffle(drug_group)
            batches.extend(drug_group[offset:offset + size] for offset in range(0, len(drug_group), size))
        rng.shuffle(batches)
        return batches

    def __iter__(self) -> Iterator[list[int]]:
        if self._cached_batches is None:
            self._cached_batches = self._make_batches()
        yield from self._cached_batches

    def __len__(self) -> int:
        if self._cached_batches is None:
            self._cached_batches = self._make_batches()
        return len(self._cached_batches)
