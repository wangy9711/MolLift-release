"""CLI entry point."""

from __future__ import annotations

import argparse


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="MolLift 2D atom annotation tool (PySide6 GUI)",
    )
    parser.add_argument("--smiles", required=True, help="SMILES string")
    parser.add_argument("--image", required=True, help="Path to projection image")
    parser.add_argument(
        "--output",
        default="annotation.json",
        help="Default output JSON path (default: annotation.json)",
    )
    return parser.parse_args(argv)


def main() -> None:
    args = parse_args()
    try:
        from mollift_annotator.gui.app import run_gui
    except ImportError as exc:
        raise SystemExit("Install the desktop extra: pip install -e '.[annotator]'") from exc
    run_gui(args.smiles, args.image, args.output)


if __name__ == "__main__":
    main()
