"""Import GeoMol split0 and its fixed 1,000-molecule benchmark test set.

The source files are published in PattanaikL/GeoMol under ``data/{QM9,DRUGS}``.
``smiles_splits/split0.npy`` defines the molecule-level 80/10/10 partition,
while ``test_smiles_corrected.csv`` identifies the fixed 1,000-molecule subset
used for COV/AMR reporting.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import shutil
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from rdkit import Chem, RDLogger
from tqdm import tqdm

GEOMOL_REPOSITORY = "https://github.com/PattanaikL/GeoMol"
GEOMOL_PAPER = "https://proceedings.neurips.cc/paper/2021/hash/725215ed82ab6306919b485b81ff9615-Abstract.html"
EXPECTED_SPLIT_COUNTS = {
    "qm9": (106_586, 13_323, 13_323),
    "drugs": (243_473, 30_433, 30_433),
}

RDLogger.DisableLog("rdApp.error")


def canonical_smiles(smiles: str) -> str:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return f"__RAW__:{smiles}"
    return Chem.MolToSmiles(mol, canonical=True, isomericSmiles=True)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)


def read_sources(source_dir: Path, dataset: str) -> tuple[dict[str, list[dict]], dict]:
    split_path = source_dir / "smiles_splits" / "split0.npy"
    index_split_path = source_dir / "splits" / "split0.npy"
    test_path = source_dir / "test_smiles_corrected.csv"
    for path in (split_path, index_split_path, test_path):
        if not path.is_file():
            raise FileNotFoundError(path)

    smiles_split = np.load(split_path, allow_pickle=True)
    index_split = np.load(index_split_path, allow_pickle=True)
    counts = tuple(len(part) for part in smiles_split)
    if counts != EXPECTED_SPLIT_COUNTS[dataset]:
        raise ValueError(
            f"Unexpected GeoMol {dataset} split0 counts: {counts}; "
            f"expected {EXPECTED_SPLIT_COUNTS[dataset]}"
        )
    if tuple(len(part) for part in index_split) != counts:
        raise ValueError("GeoMol index and SMILES split counts differ")

    splits: dict[str, list[dict]] = {}
    for name, values in zip(("train", "valid", "test_pool_all"), smiles_split):
        splits[name] = [
            {
                "official_index": int(index),
                "source_identifier": str(identifier),
                "candidate_smiles": [str(identifier)],
            }
            for index, identifier in zip(
                index_split[("train", "valid", "test_pool_all").index(name)],
                values,
            )
        ]

    with test_path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if len(rows) != 1_000:
        raise ValueError(f"Expected 1,000 benchmark molecules, found {len(rows)}")
    splits["test"] = []
    for position, row in enumerate(rows):
        candidates = list(
            dict.fromkeys(
                value
                for value in (row.get("corrected_smiles"), row.get("smiles"))
                if value
            )
        )
        splits["test"].append(
            {
                "benchmark_position": position,
                "source_identifier": str(row["smiles"]),
                "candidate_smiles": candidates,
                "n_reference_conformers": int(row["n_conformers"]),
            }
        )

    source_metadata = {
        "repository": GEOMOL_REPOSITORY,
        "paper": GEOMOL_PAPER,
        "files": {
            path.name if path != split_path else "smiles_splits/split0.npy": {
                "path": str(path.resolve()),
                "sha256": sha256_file(path),
            }
            for path in (split_path, index_split_path, test_path)
        },
        "full_partition_counts": {
            "train": counts[0],
            "valid": counts[1],
            "test_pool_all": counts[2],
        },
        "benchmark_test_count": len(splits["test"]),
    }
    return splits, source_metadata


def cache_indexes(
    con: sqlite3.Connection,
) -> tuple[dict[str, list[str]], dict[str, list[str]], dict[str, list[str]]]:
    rows = con.execute(
        "SELECT molecule_id, smiles, source_path FROM molecules"
    ).fetchall()
    raw: dict[str, list[str]] = {}
    canonical: dict[str, list[str]] = {}
    source_stem: dict[str, list[str]] = {}
    for molecule_id, smiles, source_path in tqdm(rows, desc="Index cache SMILES"):
        molecule_id = str(molecule_id)
        smiles = str(smiles)
        raw.setdefault(smiles, []).append(molecule_id)
        canonical.setdefault(canonical_smiles(smiles), []).append(molecule_id)
        source_stem.setdefault(Path(str(source_path)).stem, []).append(molecule_id)
    return raw, canonical, source_stem


def match_records(
    records: list[dict],
    raw: dict[str, list[str]],
    canonical: dict[str, list[str]],
    source_stem: dict[str, list[str]],
) -> tuple[list[str], list[dict], list[dict]]:
    matched: list[str] = []
    missing: list[dict] = []
    ambiguous: list[dict] = []
    for record in records:
        found: list[str] = []
        for candidate in record["candidate_smiles"]:
            found = raw.get(candidate, [])
            if not found:
                found = source_stem.get(candidate, [])
            if not found:
                # GeoMol's raw GEOM files encode path separators in SMILES as
                # underscores while preserving the opposite slash character.
                for escaped in (
                    candidate.replace("/", "_"),
                    candidate.replace("\\", "_"),
                ):
                    found = source_stem.get(escaped, [])
                    if found:
                        break
            if not found:
                found = canonical.get(canonical_smiles(candidate), [])
            if found:
                break
        if len(found) == 1:
            matched.append(found[0])
        elif not found:
            missing.append(record)
        else:
            ambiguous.append({**record, "molecule_ids": found})
    return matched, missing, ambiguous


def apply_dataset(
    source_dir: Path,
    cache_root: Path,
    output_dir: Path,
    dataset: str,
) -> dict:
    splits, source_metadata = read_sources(source_dir, dataset)
    manifest = cache_root / "index.sqlite"
    if not manifest.is_file():
        raise FileNotFoundError(manifest)

    con = sqlite3.connect(manifest)
    try:
        raw, canonical, source_stem = cache_indexes(con)
        matched: dict[str, list[str]] = {}
        missing: dict[str, list[dict]] = {}
        ambiguous: dict[str, list[dict]] = {}
        for name, records in splits.items():
            matched[name], missing[name], ambiguous[name] = match_records(
                records, raw, canonical, source_stem
            )
        if any(ambiguous.values()):
            raise ValueError(
                f"Ambiguous GeoMol mappings: "
                f"{ {name: len(rows) for name, rows in ambiguous.items()} }"
            )

        resolved_test = []
        for record in splits["test"]:
            molecule_ids, _, _ = match_records(
                [record], raw, canonical, source_stem
            )
            if molecule_ids:
                resolved_test.append({**record, "molecule_id": molecule_ids[0]})

        benchmark_manifest = []
        for record in splits["test"]:
            molecule_ids, record_missing, record_ambiguous = match_records(
                [record], raw, canonical, source_stem
            )
            if molecule_ids and not record_missing and not record_ambiguous:
                benchmark_manifest.append(
                    {**record, "molecule_id": molecule_ids[0]}
                )
        benchmark_ids_ordered = [
            row["molecule_id"] for row in benchmark_manifest
        ]
        placeholders = ",".join("?" for _ in benchmark_ids_ordered)
        cached_conformer_counts = dict(
            con.execute(
                f"SELECT molecule_id,n_conformers FROM molecules "
                f"WHERE molecule_id IN ({placeholders})",
                benchmark_ids_ordered,
            ).fetchall()
        )
        for row in benchmark_manifest:
            row["n_cached_conformers"] = int(
                cached_conformer_counts[row["molecule_id"]]
            )
        reference_count_mismatches = [
            row
            for row in benchmark_manifest
            if row["n_cached_conformers"] != row["n_reference_conformers"]
        ]

        duplicate_records = {}
        for name, values in matched.items():
            before = len(values)
            matched[name] = list(dict.fromkeys(values))
            duplicate_records[name] = before - len(matched[name])

        benchmark_ids = set(matched["test"])
        pool_before = len(matched["test_pool_all"])
        matched["test_pool"] = [
            molecule_id
            for molecule_id in matched.pop("test_pool_all")
            if molecule_id not in benchmark_ids
        ]
        benchmark_removed_from_pool = pool_before - len(matched["test_pool"])

        # Held-out identities take precedence if RDKit normalization collapses
        # stereochemical source records onto the same cached molecule.
        heldout = set(matched["test_pool"]) | benchmark_ids
        valid_before = len(matched["valid"])
        matched["valid"] = [x for x in matched["valid"] if x not in heldout]
        removed_valid_overlap = valid_before - len(matched["valid"])
        nontrain = heldout | set(matched["valid"])
        train_before = len(matched["train"])
        matched["train"] = [x for x in matched["train"] if x not in nontrain]
        removed_train_overlap = train_before - len(matched["train"])

        assigned = [item for values in matched.values() for item in values]
        if len(assigned) != len(set(assigned)):
            raise ValueError("A cached molecule maps to multiple GeoMol labels")

        timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        backup = cache_root / f"index.before_geomol_{timestamp}.sqlite"
        con.commit()
        shutil.copy2(manifest, backup)

        con.execute("BEGIN IMMEDIATE")
        con.execute("UPDATE molecules SET split='unused'")
        for name in ("train", "valid", "test_pool", "test"):
            con.executemany(
                "UPDATE molecules SET split=? WHERE molecule_id=?",
                [(name, molecule_id) for molecule_id in matched[name]],
            )
        con.execute(
            "CREATE TABLE IF NOT EXISTS split_metadata "
            "(key TEXT PRIMARY KEY, value TEXT NOT NULL)"
        )
        metadata = {
            "split_protocol": "geomol_split0_random_1000",
            "official_repository": GEOMOL_REPOSITORY,
            "official_paper": GEOMOL_PAPER,
            "applied_utc": datetime.now(timezone.utc).isoformat(),
            "split_files_dir": str(output_dir.resolve()),
        }
        con.execute("DELETE FROM split_metadata")
        con.executemany(
            "INSERT INTO split_metadata(key,value) VALUES (?,?)",
            list(metadata.items()),
        )
        con.commit()
        final_counts = dict(
            con.execute(
                "SELECT split,COUNT(*) FROM molecules GROUP BY split ORDER BY split"
            ).fetchall()
        )
    except Exception:
        con.rollback()
        raise
    finally:
        con.close()

    report = {
        "dataset": dataset,
        "cache_root": str(cache_root.resolve()),
        "backup_manifest": str(backup.resolve()),
        "source": source_metadata,
        "requested": {name: len(rows) for name, rows in splits.items()},
        "matched": {name: len(rows) for name, rows in matched.items()},
        "missing": {name: len(rows) for name, rows in missing.items()},
        "missing_records": missing,
        "collapsed_duplicate_records": duplicate_records,
        "benchmark_removed_from_full_test_pool": benchmark_removed_from_pool,
        "removed_valid_overlap": removed_valid_overlap,
        "removed_train_overlap": removed_train_overlap,
        "final_manifest_counts": final_counts,
        "reference_conformer_count_mismatches": reference_count_mismatches,
        "benchmark_manifest_sha256": hashlib.sha256(
            json.dumps(
                benchmark_manifest, sort_keys=True, separators=(",", ":")
            ).encode("utf-8")
        ).hexdigest(),
    }
    for name in ("train", "valid", "test_pool_all", "test"):
        write_json(output_dir / f"{name}.json", splits[name])
    write_json(output_dir / "test_resolved.json", resolved_test)
    write_json(output_dir / "mapping_report.json", report)
    write_json(output_dir / "test_matched.json", benchmark_manifest)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Import official GeoMol split0")
    parser.add_argument("--geomol-data", type=Path, required=True)
    parser.add_argument("--qm9-cache", type=Path)
    parser.add_argument("--drugs-cache", type=Path)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    if args.qm9_cache is None and args.drugs_cache is None:
        parser.error("Provide --drugs-cache and/or --qm9-cache")

    summary = {"protocol": "geomol_split0_random_1000", "datasets": {}}
    for dataset, source_name, cache_root in (
        ("qm9", "QM9", args.qm9_cache),
        ("drugs", "DRUGS", args.drugs_cache),
    ):
        if cache_root is None:
            continue
        report = apply_dataset(
            args.geomol_data / source_name,
            cache_root,
            args.output_root / dataset,
            dataset,
        )
        summary["datasets"][dataset] = report
    write_json(args.output_root / "metadata.json", summary)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
