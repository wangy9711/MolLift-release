"""Generator / Solver / constrained Refiner execution without reference data."""

from __future__ import annotations

from dataclasses import asdict
from importlib.metadata import version
from pathlib import Path
import time

import numpy as np
import torch

from mollift.io import load_input, load_annotation, sha256_file, write_json, write_sdf
from mollift.selection import select_candidate
from mollift_generator.data.collate import collate_molecules
from mollift_generator.data.featurize import featurize_mol
from mollift_generator.data.proj_mask import apply_proj_mask_to_batch
from mollift_generator.models.checkpoint import load_model_from_checkpoint
from mollift_generator.training.flow_utils import sample_initial_coords
from mollift_generator.training.sampler import euler_sample
from mollift_generator.utils.chirality import chirality_correct


def generate(molecule, model, cfg, *, num_samples=10, steps=50, seed=42,
             batch_size=10, device="cpu"):
    """Candidate i always starts from seed+i, independent of sample chunk size."""
    mol = molecule.to_rdkit_mol()
    n = mol.GetNumAtoms()
    if n > cfg["data"]["max_atoms"]:
        raise ValueError(f"{n} modeled atoms exceed this checkpoint's limit {cfg['data']['max_atoms']}")
    feat = featurize_mol(mol, max_hop=cfg["data"]["max_hop"], dataset="geom")
    sample = {**feat, "smiles": molecule.smiles, "n_atoms": n,
              "x3d": np.zeros((n, 3), dtype=np.float32),
              "pos_2d": molecule.coords_2d.astype(np.float32)}
    results, chirality = [], []
    prior_type = cfg.get("train", {}).get("prior_type", "harmonic")
    for start in range(0, num_samples, batch_size):
        count = min(batch_size, num_samples - start)
        batch = collate_molecules([sample] * count)
        mask = torch.from_numpy(np.repeat(molecule.coords_2d_mask[None, :], count, axis=0))
        batch = apply_proj_mask_to_batch(batch, mask)
        batch = {k: v.to(device) if torch.is_tensor(v) else v for k, v in batch.items()}
        initial = []
        for index in range(start, start + count):
            torch.manual_seed(seed + index)
            initial.append(sample_initial_coords(
                1, n, batch["node_mask"][:1], batch["bond_features_dense"][:1],
                prior_type=prior_type, device=torch.device(device),
            ))
        coords = euler_sample(model, batch, num_steps=steps, prior_type=prior_type,
                              projection_guidance_alpha=cfg.get("sample", {}).get("projection_guidance_alpha", 0.0),
                              x_init=torch.cat(initial))
        for xyz in coords.float().cpu().numpy():
            corrected, info = chirality_correct(xyz, mol)
            results.append(corrected)
            chirality.append(info)
    return np.asarray(results), chirality


def run(args, checkpoint: Path, model_name: str | None):
    if args.samples < 1 or args.steps < 1 or args.batch_size < 1:
        raise ValueError("samples, steps and batch-size must be positive")
    if not (0 <= args.seed < 2**32 - args.samples):
        raise ValueError("seed must be nonnegative and seed+samples < 2**32")
    if args.solver_steps < 43:
        raise ValueError("Solver needs at least 30 full-weight steps; use solver-steps >= 43")
    device = args.device or ("cuda:0" if torch.cuda.is_available() else "cpu")
    if str(device).startswith("cuda") and not torch.cuda.is_available():
        raise ValueError("CUDA requested but unavailable; use --device cpu")
    if "R" in args.pipeline:
        from mollift_refiner import OpenBabelRefiner, OpenBabelRefinerConfig
        from mollift_refiner.refiner import openbabel_version
        try:
            ob_version = openbabel_version()
        except ImportError as exc:
            raise ValueError("Refiner needs the extra: pip install -e '.[refiner]'") from exc
    else:
        ob_version = None
    molecule = (load_input(args.input) if args.input else
                load_annotation(args.annotation, args.pixels_per_angstrom))
    checkpoint = checkpoint.resolve()
    model, cfg = load_model_from_checkpoint(checkpoint, torch.device(device), atom_dim=72, bond_dim=5)
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    report = {"status": "running", "model": model_name, "checkpoint_file": checkpoint.name,
              "checkpoint_sha256": sha256_file(checkpoint), "pipeline": args.pipeline,
              "samples": args.samples, "steps": args.steps, "seed": args.seed,
              "candidate_seeds": list(range(args.seed, args.seed + args.samples)),
              "device": str(device), "solver_device": args.solver_device,
              "versions": {"torch": torch.__version__, "numpy": np.__version__,
                           "rdkit": version("rdkit"), "openbabel": ob_version},
              "input": dict(molecule.metadata), "selection_tolerance_component_angstrom": 0.10,
              "timing_seconds": {}}
    write_json(out / "input.json", {"smiles": molecule.smiles, "units": "angstrom",
               "atom_representation": molecule.metadata["atom_representation"], "centering": "provided",
               "atomic_numbers": molecule.atomic_numbers, "coords_2d": molecule.coords_2d,
               "observed_mask": molecule.coords_2d_mask})
    write_json(out / "model_config.json", cfg)
    arrays = {}
    try:
        stage_start = time.perf_counter()
        print(f"Generating {args.samples} candidates on {device}", flush=True)
        coords, chirality = generate(molecule, model, cfg, num_samples=args.samples,
                steps=args.steps, seed=args.seed, batch_size=args.batch_size, device=device)
        report["timing_seconds"]["G"] = time.perf_counter() - stage_start
        report["chirality"] = chirality
        arrays["G"] = coords
        write_sdf(out / "generator.sdf", molecule, coords)
        del model
        if str(device).startswith("cuda"):
            torch.cuda.empty_cache()
        objectives = np.zeros(args.samples)
        if "S" in args.pipeline:
            from mollift_solver import solve_3d, SolveConfig
            solver_cfg = SolveConfig(steps=args.solver_steps, device=args.solver_device)
            report["solver_config"] = asdict(solver_cfg)
            stage_start = time.perf_counter()
            print("Running Solver", flush=True)
            solved = solve_3d(molecule.smiles, molecule.coords_2d,
                    coords_3d_init=list(coords), coords_2d_mask=molecule.coords_2d_mask,
                    config=solver_cfg, mol=molecule.to_rdkit_mol())
            coords, objectives = solved.candidate_coords_3d, solved.candidate_final_losses
            report["timing_seconds"]["S"] = time.perf_counter() - stage_start
            report["solver"] = "MolLift"
            report["solver_objectives"] = objectives
            arrays["GS"] = coords
            write_json(out / "solver_diagnostics.json", solved.diagnostics)
            write_sdf(out / "solver.sdf", molecule, coords)
        rows = []
        if "R" in args.pipeline:
            from mollift_solver.projection import Plane2DFrame
            refiner_cfg = OpenBabelRefinerConfig(refine_top_m=args.samples)
            report["refiner_config"] = asdict(refiner_cfg)
            stage_start = time.perf_counter()
            print("Refining all candidates", flush=True)
            refined = OpenBabelRefiner(refiner_cfg).refine_all_candidates(
                molecule, coords, objectives, diagnostics={"frame": Plane2DFrame.default_xy_plane()},
            )
            coords = np.asarray([r.coords_3d if r.accepted else np.full_like(coords[0], np.nan)
                                 for r in refined])
            rows = [{"accepted": r.accepted, "status": r.status, "failure_reason": r.failure_reason,
                     "energy": r.energy, "force_field": r.metadata.get("force_field"),
                     "projection_rmse": r.metadata.get("projection_rmse_after"),
                     "metadata": r.metadata} for r in refined]
            report["refiner"] = rows
            report["timing_seconds"]["R"] = time.perf_counter() - stage_start
            arrays[args.pipeline] = coords
            write_sdf(out / "refined.sdf", molecule, coords)
        index = select_candidate(args.pipeline, coords, objectives, rows)
        report["selected_candidate_index"] = index
        report["candidate_projection_component_rmse"] = np.sqrt(np.mean(
            (coords[:, molecule.coords_2d_mask, :2] - molecule.coords_2d[molecule.coords_2d_mask])**2,
            axis=(1, 2)))
        if index is None:
            raise RuntimeError("No eligible candidate; see diagnostics. No fallback structure was selected.")
        write_sdf(out / "selected.sdf", molecule, coords, [index])
        report["status"] = "success"
        print(f"Selected candidate {index}; wrote {out / 'selected.sdf'}", flush=True)
    except Exception as exc:
        report.update(status="failed", error=f"{type(exc).__name__}: {exc}")
        raise
    finally:
        if arrays:
            np.savez_compressed(out / "candidates.npz", **arrays,
                  coords_2d=molecule.coords_2d, observed_mask=molecule.coords_2d_mask,
                  atomic_numbers=np.asarray(molecule.atomic_numbers))
        report["timing_seconds"]["total"] = time.perf_counter() - started
        write_json(out / "run.json", report)
