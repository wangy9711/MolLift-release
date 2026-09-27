"""Frozen APE architecture. Parameter names match the research checkpoints."""

from mollift_generator.models.dit2d import Mol2D3DFlow


def build_model(cfg: dict, atom_feature_dim: int, bond_feature_dim: int):
    m = cfg["model"]
    if m.get("model_type", "ape") != "ape" or m.get("use_chirality", False):
        raise ValueError("This release supports the frozen APE / post-hoc chirality model only")
    hops = m.get("embed_shortest_hops", True)
    distances = m.get("embed_distances", False)
    return Mol2D3DFlow(
        hidden_dim=m["hidden_dim"], num_layers=m["num_layers"],
        num_heads=m["num_heads"], num_atom_types=m["num_atom_types"],
        atom_feature_dim=atom_feature_dim, bond_feature_dim=bond_feature_dim,
        max_hop=cfg["data"]["max_hop"], use_2d_condition=m.get("use_2d_condition", True),
        use_absolute_pos_embed=m.get("use_absolute_pos_embed", True),
        embed_distances=distances, embed_shortest_hops=hops,
        radial_basis_bool=m.get("radial_basis_bool", False),
        num_radial_basis=m.get("num_radial_basis", 8),
        use_pair_modulation=distances or hops,
        act_dense_correct=m.get("act_dense_correct", True), use_chirality=False,
    )
