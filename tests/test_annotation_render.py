import pytest
from rdkit import Chem

from mollift_annotator.render import (
    atom_at_point,
    compose_projection_image,
    compute_display_size,
    compute_mol_render_size,
    display_to_original,
    original_to_display,
    render_molecule,
)


def test_render_molecule_has_atom_coords():
    mol = Chem.MolFromSmiles("CCO")
    img, coords = render_molecule(mol, width=400, height=400)
    assert img.size == (400, 400)
    assert len(coords) == 3


def test_atom_at_point_hit():
    mol = Chem.MolFromSmiles("CCO")
    _, coords = render_molecule(mol, width=400, height=400)
    idx, (cx, cy) = next(iter(coords.items()))
    assert atom_at_point(cx, cy, coords) == idx
    assert atom_at_point(-100.0, -100.0, coords) is None


def test_compose_projection_image_markers():
    from PIL import Image

    base = Image.new("RGBA", (100, 100), (240, 240, 240, 255))
    out = compose_projection_image(base, {0: (50.0, 50.0)}, None, scale=1.0)
    assert out.size == (100, 100)
    assert out.getpixel((50, 50)) != base.getpixel((50, 50))


def test_coordinate_scale_roundtrip():
    dw, dh, scale = compute_display_size(800, 600, max_width=400, max_height=400)
    assert dw <= 400
    assert dh <= 400
    ox, oy = display_to_original(100.0, 75.0, scale)
    dx, dy = original_to_display(ox, oy, scale)
    assert dx == pytest.approx(100.0, abs=0.5)
    assert dy == pytest.approx(75.0, abs=0.5)


def test_compute_mol_render_size_scales_up():
    rw, rh, scale = compute_mol_render_size(num_atoms=3)
    assert rw >= 420 * 4
    assert rh >= 420 * 4
    assert scale >= 4.0

    rw_large, _, scale_large = compute_mol_render_size(num_atoms=100)
    assert rw_large > rw
    assert scale_large > scale
