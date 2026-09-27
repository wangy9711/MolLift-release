"""Application entry for PySide6 GUI."""

from __future__ import annotations

import sys
from typing import Optional

from PySide6.QtWidgets import QApplication

from mollift_annotator.gui.main_window import AnnotatorWindow


def run_gui(
    smiles: str,
    image_path: str,
    output_path: Optional[str] = None,
) -> None:
    """Launch the annotation GUI and block until the window is closed."""
    app = QApplication.instance()
    if app is None:
        app = QApplication(sys.argv)

    win = AnnotatorWindow(
        smiles=smiles,
        image_path=image_path,
        output_path=output_path,
    )
    win.show()
    app.exec()


def main() -> None:
    from mollift_annotator.cli import parse_args

    args = parse_args()
    run_gui(args.smiles, args.image, args.output)
