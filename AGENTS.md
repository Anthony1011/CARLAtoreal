# Repository Guidelines

## Project Structure & Module Organization

This repository converts CARLA semantic recordings into photorealistic driving video using a modified pix2pixHD model.

- `pix2pixHD/models/`, `data/`, `options/`, and `util/` contain networks, dataset loaders, CLI options, and utilities; `train.py` and `test.py` run training and inference.
- `carla2real/` groups Python tools by function; `scripts/` contains workflow entry points and `experiments/` preserves version-specific recipes.
- `carla2real/config.py` and `configs/config.sh` define shared paths. `docs/` contains state, handover notes, and experiment history; `datasets/` contains corpus documentation.
- Recordings, checkpoints, results, and videos are external or Git-ignored assets. A clone contains no trained weights.

## Build, Test, and Development Commands

Use a Linux/Bash environment matching the documented Python 3.10 and CUDA setup. There is no separate build step.

```bash
pip install -r requirements.txt
source configs/config.sh
python3 -m carla2real.recording.record_town_auto --town Town05 --weather sunny --outname Town05_sunny_inst
bash scripts/inference/render_model.sh sunny carla2real_semantic_v50_graft review01 Town05
python3 -m carla2real.evaluation.eval_model MODEL PARENT PHASE --out results.json
```

These install dependencies, export paths, record a drive, render an existing checkpoint, and compare model outputs. Recording requires a matching CARLA server; rendering requires prepared channels and weights. The render example uses a historical checkpoint, not the current baseline. Consult `README.md` for baseline recipes and verify referenced scripts exist before launching; some delivery helpers are absent. Run one GPU job at a time.

## Coding Style & Naming Conventions

Use four-space Python indentation, `snake_case` functions and modules, and uppercase configuration constants. Match surrounding code; no formatter or linter is configured. Import paths from `carla2real/config.py` and source `configs/config.sh` in shell drivers. Keep their environment-variable semantics aligned. Use distinct experiment names and output tags to preserve comparisons.

## Testing Guidelines

Run `python scripts/check_layout.py` for dependency-free syntax and reference checks (Bash checks require Bash). It does not execute pipeline modules, many of which run at import time. No coverage threshold is configured; `pix2pixHD/test.py` performs inference. Validate changed stages on a prepared recording, check frame/channel alignment, and reject checkpoint logs containing `not initialized`. Compare frames visually alongside metrics; use `carla2real/evaluation/true_instability.py` for stability across differing sharpness.

## Commit & Pull Request Guidelines

History uses descriptive subject lines, sometimes prefixed with `README:`; Conventional Commits are not required. Keep commits focused. PRs should explain the change, identify affected models/channels, include commands and validation results, and attach before/after crops for visual changes. Link relevant issues and record significant findings in `docs/EXPERIMENTS.md`.

## Configuration & Asset Handling

Set `CARLA2REAL_DATA` and `CARLA2REAL_OUT` for local storage. Keep bulk assets and machine-specific configuration out of commits. Consult `THIRD_PARTY_NOTICES.md` before redistributing code, datasets, or weights.
