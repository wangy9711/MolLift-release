"""Precompute molecule-level caches for GEOM-QM9 or GEOM-Drugs.

Completed molecule files are reused on resume. The cache stores RDKit features,
shortest paths and harmonic-prior eigendecompositions for training.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from pathlib import Path
from typing import Iterable

from tqdm import tqdm

from mollift_generator.data.geom_cache import GEOMRecord, connect_manifest, upsert_records
from mollift_generator.data.geom_pickle import convert_geom_pickle


def assign_hash_split(
    source_path: str | Path,
    *,
    valid_fraction: float,
    test_fraction: float,
    split_seed: int,
) -> str:
    """Stable molecule-level split based on the GEOM SMILES filename."""
    key = f"{split_seed}:{Path(source_path).stem}".encode("utf-8")
    bucket = int.from_bytes(hashlib.sha256(key).digest()[:8], "big") / 2**64
    if bucket < test_fraction:
        return "test"
    if bucket < test_fraction + valid_fraction:
        return "valid"
    return "train"


def _convert_worker(task: tuple) -> tuple[str, GEOMRecord | None, str | None]:
    (
        source_path,
        cache_root,
        split,
        max_atoms,
        min_conformers,
        max_conformers,
        max_hop,
        overwrite,
    ) = task
    try:
        record = convert_geom_pickle(
            source_path,
            cache_root,
            split,
            max_atoms=max_atoms,
            min_conformers=min_conformers,
            max_conformers=max_conformers,
            max_hop=max_hop,
            overwrite=overwrite,
        )
        return source_path, record, None
    except Exception as exc:
        return source_path, None, f"{type(exc).__name__}: {exc}"


def _write_failure_log(path: Path, failures: list[dict]) -> None:
    if not failures:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        for failure in failures:
            handle.write(json.dumps(failure, ensure_ascii=False) + "\n")


def _bounded_worker_results(pool: ProcessPoolExecutor, tasks: Iterable[tuple], limit: int):
    """Yield worker results without creating hundreds of thousands of futures."""
    task_iter = iter(tasks)
    pending = set()
    for _ in range(limit):
        try:
            task = next(task_iter)
        except StopIteration:
            break
        pending.add(pool.submit(_convert_worker, task))

    while pending:
        done, pending = wait(pending, return_when=FIRST_COMPLETED)
        for future in done:
            yield future.result()
            try:
                task = next(task_iter)
            except StopIteration:
                continue
            pending.add(pool.submit(_convert_worker, task))


def main() -> None:
    parser = argparse.ArgumentParser(description="Precompute GEOM molecule caches")
    parser.add_argument("--data-root", type=Path, required=True, help="Directory of *.pickle files")
    parser.add_argument("--cache-root", type=Path, required=True)
    parser.add_argument("--dataset-name", choices=["qm9", "drugs"], required=True)
    parser.add_argument("--workers", type=int, default=max(1, min(8, os.cpu_count() or 1)))
    parser.add_argument("--max-atoms", type=int, default=0, help="0 keeps every molecule")
    parser.add_argument("--min-conformers", type=int, default=1)
    parser.add_argument("--max-conformers", type=int, default=0, help="0 keeps all conformers")
    parser.add_argument("--max-hop", type=int, default=8)
    parser.add_argument("--valid-fraction", type=float, default=0.1)
    parser.add_argument("--test-fraction", type=float, default=0.1)
    parser.add_argument("--split-seed", type=int, default=42)
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--end", type=int, default=None, help="Exclusive file index")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--commit-every", type=int, default=100)
    args = parser.parse_args()

    if not args.data_root.is_dir():
        raise FileNotFoundError(args.data_root)
    if args.valid_fraction < 0 or args.test_fraction < 0:
        raise ValueError("split fractions must be non-negative")
    if args.valid_fraction + args.test_fraction >= 1:
        raise ValueError("valid_fraction + test_fraction must be < 1")
    if args.workers < 1:
        raise ValueError("workers must be >= 1")

    source_paths = sorted(args.data_root.glob("*.pickle"))
    end = len(source_paths) if args.end is None else min(args.end, len(source_paths))
    source_paths = source_paths[args.start:end]
    if not source_paths:
        raise ValueError("No GEOM pickle files selected")

    args.cache_root.mkdir(parents=True, exist_ok=True)
    con = connect_manifest(args.cache_root)
    failure_log = args.cache_root / "failures.jsonl"
    max_atoms = args.max_atoms or None
    max_conformers = args.max_conformers or None
    tasks = (
        (
            str(path),
            str(args.cache_root),
            assign_hash_split(
                path,
                valid_fraction=args.valid_fraction,
                test_fraction=args.test_fraction,
                split_seed=args.split_seed,
            ),
            max_atoms,
            args.min_conformers,
            max_conformers,
            args.max_hop,
            args.overwrite,
        )
        for path in source_paths
    )

    started = time.time()
    converted = 0
    failed = 0
    conformers = 0
    pending_records: list[GEOMRecord] = []
    pending_failures: list[dict] = []
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        results = _bounded_worker_results(pool, tasks, limit=max(args.workers * 4, 1))
        for source_path, record, error in tqdm(
            results,
            total=len(source_paths),
            desc=f"GEOM-{args.dataset_name}",
        ):
            if record is not None:
                pending_records.append(record)
                converted += 1
                conformers += record.n_conformers
            else:
                failed += 1
                pending_failures.append({"source_path": source_path, "error": error})

            if len(pending_records) >= args.commit_every:
                upsert_records(con, pending_records)
                pending_records.clear()
            if len(pending_failures) >= args.commit_every:
                _write_failure_log(failure_log, pending_failures)
                pending_failures.clear()

    if pending_records:
        upsert_records(con, pending_records)
    _write_failure_log(failure_log, pending_failures)

    split_counts = {
        split: con.execute(
            "SELECT COUNT(*), COALESCE(SUM(n_conformers), 0) FROM molecules WHERE split = ?",
            (split,),
        ).fetchone()
        for split in ("train", "valid", "test")
    }
    con.close()

    summary = {
        "dataset_name": args.dataset_name,
        "data_root": str(args.data_root.resolve()),
        "cache_root": str(args.cache_root.resolve()),
        "selected_range": [args.start, end],
        "selected_files": len(source_paths),
        "converted_or_reused": converted,
        "valid_conformers_in_selected_files": conformers,
        "failed_or_filtered": failed,
        "split_seed": args.split_seed,
        "valid_fraction": args.valid_fraction,
        "test_fraction": args.test_fraction,
        "max_atoms": max_atoms,
        "min_conformers": args.min_conformers,
        "max_conformers": max_conformers,
        "max_hop": args.max_hop,
        "workers": args.workers,
        "elapsed_seconds": time.time() - started,
        "manifest_totals": {
            split: {"molecules": int(counts[0]), "conformers": int(counts[1])}
            for split, counts in split_counts.items()
        },
    }
    summary_path = args.cache_root / "precompute_summary.json"
    with summary_path.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)

    print(json.dumps(summary, indent=2))
    print(f"Manifest: {args.cache_root / 'index.sqlite'}")
    print(f"Summary:  {summary_path}")
    if failed:
        print(f"Failures/filters ({failed} this invocation): {failure_log}")


if __name__ == "__main__":
    main()
