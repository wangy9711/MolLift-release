"""Pure rendering helpers for molecule and projection images."""

from __future__ import annotations

import io
from typing import Dict, Optional, Set, Tuple

from PIL import Image, ImageDraw
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem.Draw import rdMolDraw2D

DEFAULT_MOL_WIDTH = 420
DEFAULT_MOL_HEIGHT = 420
DEFAULT_HIT_RADIUS = 25.0
# Render molecules at higher pixel density so zoom stays sharp.
MOL_RENDER_SCALE = 4.0
MOL_RENDER_SCALE_MAX = 8.0


def compute_mol_render_size(
    num_atoms: int,
    view_width: int = DEFAULT_MOL_WIDTH,
    view_height: int = DEFAULT_MOL_HEIGHT,
    base_scale: float = MOL_RENDER_SCALE,
) -> Tuple[int, int, float]:
    """
    Pick Cairo render size for a molecule view.

    Returns (render_width, render_height, scale) where scale maps view coords
    to render coords (render = view * scale).
    """
    # Larger structures benefit from extra pixels when zoomed in.
    atom_boost = 1.0 + min(max(num_atoms - 1, 0), 150) / 50.0
    scale = min(base_scale * atom_boost, MOL_RENDER_SCALE_MAX)
    return int(view_width * scale), int(view_height * scale), scale


def atom_at_point(
    x: float,
    y: float,
    atom_coords: Dict[int, Tuple[float, float]],
    hit_radius: float = DEFAULT_HIT_RADIUS,
) -> Optional[int]:
    best_idx: Optional[int] = None
    best_dist = hit_radius
    for idx, (cx, cy) in atom_coords.items():
        dist = ((x - cx) ** 2 + (y - cy) ** 2) ** 0.5
        if dist <= best_dist:
            best_dist = dist
            best_idx = idx
    return best_idx


def render_molecule(
    mol: Chem.Mol,
    width: int = DEFAULT_MOL_WIDTH,
    height: int = DEFAULT_MOL_HEIGHT,
    annotated: Optional[Set[int]] = None,
    skipped: Optional[Set[int]] = None,
    selected: Optional[int] = None,
) -> Tuple[Image.Image, Dict[int, Tuple[float, float]]]:
    """Render 2D molecule depiction; return PIL image and atom draw coordinates."""
    annotated = annotated or set()
    skipped = skipped or set()

    mol_copy = Chem.Mol(mol)
    AllChem.Compute2DCoords(mol_copy)

    highlight_atoms: list[int] = []
    highlight_colors: Dict[int, Tuple[float, float, float]] = {}
    for idx in range(mol_copy.GetNumAtoms()):
        if idx in annotated:
            highlight_atoms.append(idx)
            highlight_colors[idx] = (0.2, 0.75, 0.3)
        elif idx in skipped:
            highlight_atoms.append(idx)
            highlight_colors[idx] = (0.7, 0.7, 0.7)

    if selected is not None:
        if selected not in highlight_atoms:
            highlight_atoms.append(selected)
        highlight_colors[selected] = (1.0, 0.55, 0.1)

    drawer = rdMolDraw2D.MolDraw2DCairo(width, height)
    opts = drawer.drawOptions()
    opts.addAtomIndices = True
    size_factor = max(width, height) / float(DEFAULT_MOL_WIDTH)
    # High-res canvas: scale bond strokes for crisp zoom, but keep labels small.
    opts.bondLineWidth = max(1.5, 2.0 * min(size_factor, 3.0))
    opts.padding = 0.05
    opts.fixedFontSize = 12
    opts.minFontSize = 8
    opts.maxFontSize = 12
    opts.annotationFontScale = 0.7
    if highlight_atoms:
        drawer.DrawMolecule(
            mol_copy,
            highlightAtoms=highlight_atoms,
            highlightAtomColors=highlight_colors,
        )
    else:
        drawer.DrawMolecule(mol_copy)
    drawer.FinishDrawing()

    atom_coords: Dict[int, Tuple[float, float]] = {}
    for idx in range(mol_copy.GetNumAtoms()):
        pt = drawer.GetDrawCoords(idx)
        atom_coords[idx] = (float(pt.x), float(pt.y))

    img = Image.open(io.BytesIO(drawer.GetDrawingText())).convert("RGBA")

    if selected is not None and selected in atom_coords:
        cx, cy = atom_coords[selected]
        draw = ImageDraw.Draw(img)
        size_factor = max(width, height) / float(DEFAULT_MOL_WIDTH)
        r = max(10, int(10 * min(size_factor, 2.0)))
        line_w = max(2, int(2 * min(size_factor, 2.0)))
        draw.ellipse((cx - r, cy - r, cx + r, cy + r), outline="#FF8800", width=line_w)

    return img, atom_coords


def load_projection_image(image_path: str) -> Image.Image:
    return Image.open(image_path).convert("RGBA")


def compute_display_size(
    orig_width: int,
    orig_height: int,
    max_width: int,
    max_height: int,
) -> Tuple[int, int, float]:
    """Return display (w, h) and scale factor (display / original)."""
    if orig_width <= 0 or orig_height <= 0:
        return 1, 1, 1.0
    scale = min(max_width / orig_width, max_height / orig_height, 1.0)
    return (
        max(1, int(orig_width * scale)),
        max(1, int(orig_height * scale)),
        scale,
    )


def display_to_original(x: float, y: float, scale: float) -> Tuple[float, float]:
    if scale <= 0:
        return float(x), float(y)
    return x / scale, y / scale


def original_to_display(x: float, y: float, scale: float) -> Tuple[float, float]:
    return x * scale, y * scale


def compose_projection_image(
    base_image: Image.Image,
    bindings: Dict[int, Tuple[float, float]],
    pending: Optional[Tuple[float, float]],
    scale: float,
) -> Image.Image:
    """Draw annotation markers on a display-scaled copy of the projection image."""
    img = base_image.copy()
    draw = ImageDraw.Draw(img)
    for atom_idx, (ox, oy) in bindings.items():
        dx, dy = original_to_display(ox, oy, scale)
        _draw_marker(draw, dx, dy, str(atom_idx), fill="#22AA44")
    if pending is not None:
        ox, oy = pending
        dx, dy = original_to_display(ox, oy, scale)
        _draw_marker(draw, dx, dy, "?", fill="#FF8800")
    return img


def _draw_marker(
    draw: ImageDraw.ImageDraw,
    x: float,
    y: float,
    label: str,
    fill: str,
) -> None:
    r = 6
    draw.ellipse((x - r, y - r, x + r, y + r), fill=fill, outline="#FFFFFF", width=2)
    draw.text((x + 8, y - 14), label, fill="#FFFFFF")
