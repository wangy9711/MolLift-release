"""Public training entry tested against an entirely synthetic two-molecule cache."""

import copy
import pickle
from pathlib import Path

import pytest
import torch
import yaml
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem

from mollift.cli import main
from mollift.io import sha256_file
from mollift_generator.data.geom_cache import connect_manifest, upsert_records
from mollift_generator.data.geom_pickle import convert_geom_pickle


def make_cache(tmp_path):
    cache = tmp_path / "cache"
    rows = []
    for smiles, split in [("CCO", "train"), ("CCCO", "valid")]:
        mol = Chem.AddHs(Chem.MolFromSmiles(smiles))
        assert AllChem.EmbedMolecule(mol, randomSeed=42) == 0
        source = tmp_path / f"{split}.pickle"
        source.write_bytes(pickle.dumps({"smiles": smiles, "conformers": [{"rd_mol": mol}]}))
        rows.append(convert_geom_pickle(source, cache, split, max_atoms=181))
    con = connect_manifest(cache)
    upsert_records(con, rows)
    con.commit()
    con.close()
    return cache


def test_public_trainer_cache_validation_and_resume(tmp_path, monkeypatch):
    torch.set_num_threads(1)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    cache = make_cache(tmp_path)
    root = Path(__file__).resolve().parents[1]
    cfg = yaml.safe_load((root / "configs/drugs_mixed.yaml").read_text())
    cfg["run_name"] = "synthetic_smoke"
    cfg["model"].update(hidden_dim=32, num_layers=1, num_heads=4)
    cfg["data"]["cache_root"] = str(cache)
    cfg["train"].update(epochs=1, batch_size=1, num_workers=0, use_amp=False,
                        checkpoint_dir=str(tmp_path / "checkpoints"), log_dir=str(tmp_path / "logs"))
    path = tmp_path / "smoke.yaml"
    path.write_text(yaml.safe_dump(cfg))
    # An unimported cache is rejected rather than silently training with a hash split.
    with pytest.raises(SystemExit) as error:
        main(["train", "--config", str(path)])
    assert error.value.code == 1
    # A toy test explicitly opts out; this is not the GeoMol benchmark.
    cfg["data"]["require_geomol_split"] = False
    path.write_text(yaml.safe_dump(cfg))
    main(["train", "--config", str(path)])
    latest = tmp_path / "checkpoints/synthetic_smoke_latest.pt"
    best = tmp_path / "checkpoints/synthetic_smoke_best.pt"
    assert latest.exists() and best.exists()
    state = torch.load(latest, weights_only=False)
    assert state["epoch"] == 1 and state["global_step"] == 1
    assert state["best_val"] == state["val_loss"]
    assert "time_embed.freqs" in state["ema_state_dict"]
    assert "ratios=0.3:" in (tmp_path / "logs/synthetic_smoke.log").read_text()
    digest = sha256_file(latest)
    main(["train", "--config", str(path)])
    assert sha256_file(latest) == digest  # completed epoch is not trained again


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Peptide trainer requires CUDA")
def test_peptide_transfer_and_retention_baseline_resume(tmp_path):
    # These are generated fixtures, not records from the private PDB package.
    import pandas as pd
    from mollift_generator.data.featurize import featurize_mol
    from mollift_generator.data.geom_pickle import _prior_arrays
    from mollift_generator.models.factory import build_model
    from mollift_generator.training.ema import EMAModel
    torch.set_num_threads(1)
    cache = make_cache(tmp_path)
    root = Path(__file__).resolve().parents[1]
    input_data = __import__("json").loads((root / "examples/peptide.json").read_text())
    full = Chem.AddHs(Chem.MolFromSmiles(input_data["smiles"]))
    assert AllChem.EmbedMolecule(full, randomSeed=42) == 0
    mol = Chem.RemoveHs(full)
    features = featurize_mol(mol, dataset="geom")
    node_attr, evecs = _prior_arrays(features["bond_features_dense"], mol.GetNumAtoms())
    base, derived, manifest = [], [], []
    for i, (split, label) in enumerate([("train", x) for x in ["helix", "sheet", "coil", "mixed"]] + [("valid", "coil")]):
        fid = f"{i+1:016x}"
        base.append({"fragment_id": fid, "smiles": input_data["smiles"],
                     "coordinates": mol.GetConformer().GetPositions(),
                     "atomic_numbers": features["atom_types"], "atom_features": features["atom_features"],
                     "bond_index": features["bond_index"],
                     "bond_type": np.array([mol.GetBondBetweenAtoms(int(x), int(y)).GetBondTypeAsDouble()
                                            for x, y in features["bond_index"].T]),
                     "experimental_atom_mask": np.ones(mol.GetNumAtoms(), dtype=bool),
                     "cap_atom_mask": np.zeros(mol.GetNumAtoms(), dtype=bool), "atom_mapping": []})
        derived.append({"fragment_id": fid, "shortest_path": features["shortest_path"],
                        "prior_node_attr": node_attr, "prior_evecs": evecs})
        manifest.append({"fragment_id": fid, "split": split, "shard": "fragments/toy.pt",
                         "shard_index": i, "n_atoms": mol.GetNumAtoms(), "length": 5,
                         "secondary_structure_class": label, "pdb_id": "synthetic", "chain_uid": str(i),
                         "sequence": "GAGAG", "release_date": "2020-01-01"})
    package = tmp_path / "pdb"
    for directory in ["manifests", "fragments", "derived_cache/graph_v1"]:
        (package / directory).mkdir(parents=True)
    pd.DataFrame(manifest).to_parquet(package / "manifests/fragments.parquet")
    torch.save(base, package / "fragments/toy.pt")
    torch.save(derived, package / "derived_cache/graph_v1/toy.pt")
    cfg = yaml.safe_load((root / "configs/peptide_mixed.yaml").read_text())
    cfg["run_name"] = "peptide_smoke"
    cfg["model"].update(hidden_dim=32, num_layers=1, num_heads=4)
    cfg["data"].update(pdb_root=str(package), drugs_cache_root=str(cache), require_geomol_split=False)
    init = tmp_path / "init.pt"
    model = build_model(cfg, 72, 5)
    torch.save({"ema_state_dict": EMAModel(model).state_dict()}, init)
    cfg["train"].update(epochs=1, num_workers=0, use_amp=False, init_checkpoint=str(init),
                        pdb_samples_per_epoch=3, target_graphs_per_step=4,
                        curriculum=[{"until_epoch": 99, "max_length": 5}],
                        atom_buckets=[{"max_atoms": 500, "batch_size": 2}],
                        checkpoint_dir=str(tmp_path / "peptide_checkpoints"), log_dir=str(tmp_path / "peptide_logs"))
    path = tmp_path / "peptide.yaml"
    path.write_text(yaml.safe_dump(cfg))
    main(["train", "--config", str(path)])
    latest = tmp_path / "peptide_checkpoints/peptide_smoke_latest.pt"
    saved = torch.load(latest, weights_only=False)
    assert saved["global_step"] >= 1
    assert set(saved["transfer_baseline"]["drugs_ratios"]) == {.3, .5, .8, 1.}
    digest = sha256_file(latest)
    main(["train", "--config", str(path)])
    assert sha256_file(latest) == digest
