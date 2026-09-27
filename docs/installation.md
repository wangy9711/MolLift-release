# Installation

Run commands from the repository root, in your existing Python environment.
PyTorch must be installed with a build supporting your GPU/driver. MolLift does
not install or reconfigure CUDA drivers.

```bash
python -m pip install -e '.[refiner,train]'
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
python -m mollift --help
```

For G or GS only, `pip install -e .` is sufficient. `.[refiner]` installs
`openbabel-wheel==3.1.1.22`; this is the backend used by the paper's Refiner.

The release was checked using Python 3.10, PyTorch 2.11.0, NumPy 2.2.6,
RDKit 2025.9.6, Open Babel wheel 3.1.1.22 and PyArrow 24.0.0. Version ranges in
`pyproject.toml` are installation constraints, not claims that every combination
has been tested. RDKit upgrades can affect embeddings, atom perception and
force-field preparation; record the versions when comparing results.

The optional GUI requires `pip install -e '.[annotator]'` and a desktop display.
JSON annotation reading does not require Qt. Headless servers can perform all
inference using annotations prepared on another computer.

