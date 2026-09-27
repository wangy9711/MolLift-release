"""One optimizer update and EMA reload on synthetic input; no private data needed."""

import numpy as np
import torch
from rdkit import Chem
from rdkit.Chem import AllChem

from mollift_generator.data.collate import collate_molecules
from mollift_generator.data.featurize import featurize_mol
from mollift_generator.models.factory import build_model
from mollift_generator.models.checkpoint import load_model_from_checkpoint
from mollift_generator.training.config_utils import loss_config_from_cfg
from mollift_generator.training.losses import compute_losses
from mollift_generator.training.ema import EMAModel
from mollift_generator.training.train_step import optimizer_step


def test_backward_ema_and_checkpoint_reload(tmp_path):
    torch.set_num_threads(1)
    mol = Chem.AddHs(Chem.MolFromSmiles("CCO"))
    assert AllChem.EmbedMolecule(mol, randomSeed=42) == 0
    xyz = np.asarray(mol.GetConformer().GetPositions(), dtype=np.float32)
    xyz -= xyz.mean(0)
    sample = {**featurize_mol(mol, dataset="geom"), "smiles": "CCO",
              "n_atoms": len(xyz), "x3d": xyz, "pos_2d": xyz[:, :2]}
    batch = collate_molecules([sample])
    cfg = {"data": {"max_hop": 8, "max_atoms": 181, "ratio_proj": 1.,
                    "ratio_proj_choices": [.3, .5, .8, 1.], "projection_mode": "synthetic"},
           "model": {"model_type": "ape", "hidden_dim": 32, "num_layers": 1,
                     "num_heads": 4, "num_atom_types": 100},
           "train": {"prior_type": "harmonic", "align_x0": False,
                     "augmentation_bool": True, "sigma": .08, "lambda_proj_xy": 1., "lambda_bond": 1.}}
    model = build_model(cfg, 72, 5)
    ema = EMAModel(model)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)
    scaler = torch.amp.GradScaler("cuda", enabled=False)
    before = {k: v.clone() for k, v in model.state_dict().items()}
    losses = compute_losses(model, batch, loss_config_from_cfg(cfg))
    assert optimizer_step(model, optimizer, scaler, losses["loss_total"], 1.)
    assert any(not torch.equal(before[k], v) for k, v in model.state_dict().items())
    ema.update(model)
    path = tmp_path / "model.pt"
    torch.save({"config": cfg, "ema_state_dict": ema.state_dict()}, path)
    loaded, _ = load_model_from_checkpoint(path, torch.device("cpu"))
    assert all(torch.equal(v, loaded.state_dict()[k]) for k, v in ema.state_dict().items())
