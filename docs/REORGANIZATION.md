# Repository reorganization — 2026-09-14

## Entry points

Run Python tools from the repository root using `python3 -m carla2real.<group>.<module>`.
For example, `python3 -m carla2real.recording.record_town_auto --town Town05 --weather sunny`.
Run shell workflows with `bash scripts/inference/render_model.sh ...` or the corresponding path
below. Old root-level entry points are removed; no compatibility wrappers are installed.

Shell workflows source `configs/config.sh` relative to their own location. This sets the original
repository root and adds it to `PYTHONPATH`, allowing Python modules to resolve after `cd` into
pix2pixHD. For manual Python execution outside the repository, set `PYTHONPATH` to its absolute
root first. `scripts/init_asset_dirs.py` and `scripts/check_layout.py` discover their root from their
own location and may be invoked by absolute path from another directory.

## Scope and limits

- `carla2real/`: recording, preprocessing, postprocessing, evaluation, validation, visualization,
  and common utilities. The old 19-class label converter is under `preprocessing/legacy/`.
- `scripts/`: shared workflow entry points; `experiments/`: version-specific recipes, retained
  even when historical. `make_v33.py` stays in postprocessing because render still calls it.
- The pix2pixHD model core, algorithms, CLI argument names, dependency versions, and asset paths
  are unchanged. No datasets or checkpoints were downloaded.
- Many Python modules still execute at import time. Use their CLI; moving them into a package
  does not turn them into safe importable APIs. Package `__init__.py` files perform no work.
- `datasets/`, `pix2pixHD/checkpoints/`, `pix2pixHD/results/`, and `output/` retain their existing
  locations. The render driver still hardcodes its data directory and conda environment.
- Training/delivery workflows still contain their original cleanup and overwrite operations;
  syntax checks do not run these workflows.

## Known unresolved dependencies

The shell drivers already referenced these absent helpers before the move. They remain unresolved
at their original root locations rather than being assigned speculative new paths:

`build_v51_corpus.py`, `gen_chroma_maps.py`, `gen_depth_moge.py`, `gen_edges_denoised.py`,
`gen_light_maps.py`, `gen_mapillary_labels_paths.py`, `make_v50_init.py`.

The README's `make_v50r.sh`, v75 texture integration, DVP checkout, and actual image/weight assets
also remain unavailable. See [the inference inventory](INFERENCE_FLOW.md) for Q1–Q8. Historical
experiment and handover documents retain their original command names; use the table below to
translate those names. A historical recipe is not a verified current baseline.

## Verification

Completed on 2026-09-14:

- All 60 mapped files are present at their new locations, with no old root entry points.
- 78 Python files pass AST parsing; 19 shell files pass Git Bash syntax checks.
- 40 moved Python tools retain identical executable ASTs after normalizing import/bootstrap
  changes; 18 shell drivers differ only by the planned path/import substitutions.
- All model-core file hashes match the pre-move inventory.
- Python configuration and directory initialization work from both root and docs working
  directories; the data environment override and Bash root discovery pass.
- 13 representative code/asset ignore cases and local Markdown links pass inspection.
- GPU inference, training, real data processing, DVP and external perception were not run.

Run `python scripts/check_layout.py` for migration coverage, Python syntax, local imports, module
calls, sourced shell paths, and Bash syntax when Bash is installed. It reports known missing
helpers separately and does not import pipeline modules. Full inference still requires assets and
the appropriate GPU environment. The migration manifest records the pre-move model file hashes
for auditing this reorganization; it is not a restriction on future model development.

## File mapping

| Before | After |
|---|---|
| `config.py` | `carla2real/config.py` |
| `config.sh` | `configs/config.sh` |
| `record_town_auto.py` | `carla2real/recording/record_town_auto.py` |
| `flatten_pandaset.py` | `carla2real/preprocessing/flatten_pandaset.py` |
| `flatten_zod.py` | `carla2real/preprocessing/flatten_zod.py` |
| `build_pandaset_night.py` | `carla2real/preprocessing/build_pandaset_night.py` |
| `build_temporal_corpus.py` | `carla2real/preprocessing/build_temporal_corpus.py` |
| `gen_instance_maps.py` | `carla2real/preprocessing/gen_instance_maps.py` |
| `gen_texture_energy.py` | `carla2real/preprocessing/gen_texture_energy.py` |
| `prepare_gt_test_label.py` | `carla2real/preprocessing/legacy/prepare_gt_test_label.py` |
| `protect_billboards.py` | `carla2real/postprocessing/protect_billboards.py` |
| `protect_buildings.py` | `carla2real/postprocessing/protect_buildings.py` |
| `protect_lane_markings.py` | `carla2real/postprocessing/protect_lane_markings.py` |
| `protect_light_pools.py` | `carla2real/postprocessing/protect_light_pools.py` |
| `protect_traffic_lights_carla.py` | `carla2real/postprocessing/protect_traffic_lights_carla.py` |
| `protect_vehicle_colour.py` | `carla2real/postprocessing/protect_vehicle_colour.py` |
| `stabilize_frames_v2.py` | `carla2real/postprocessing/stabilize_frames_v2.py` |
| `photoreal_post.py` | `carla2real/postprocessing/photoreal_post.py` |
| `temporal_deshimmer.py` | `carla2real/postprocessing/temporal_deshimmer.py` |
| `class_deshimmer.py` | `carla2real/postprocessing/class_deshimmer.py` |
| `despeckle_night.py` | `carla2real/postprocessing/despeckle_night.py` |
| `fuse_colour.py` | `carla2real/postprocessing/fuse_colour.py` |
| `make_v33.py` | `carla2real/postprocessing/make_v33.py` |
| `eval_model.py` | `carla2real/evaluation/eval_model.py` |
| `score_vp.py` | `carla2real/evaluation/score_vp.py` |
| `compare_scores.py` | `carla2real/evaluation/compare_scores.py` |
| `epoch_sweep.py` | `carla2real/evaluation/epoch_sweep.py` |
| `flicker_report.py` | `carla2real/evaluation/flicker_report.py` |
| `true_instability.py` | `carla2real/evaluation/true_instability.py` |
| `veg_report.py` | `carla2real/evaluation/veg_report.py` |
| `road_texture.py` | `carla2real/evaluation/road_texture.py` |
| `road_sky_ceiling.py` | `carla2real/evaluation/road_sky_ceiling.py` |
| `speckle_report.py` | `carla2real/evaluation/speckle_report.py` |
| `speck_where.py` | `carla2real/evaluation/speck_where.py` |
| `tail_check.py` | `carla2real/evaluation/tail_check.py` |
| `deshim_compare.py` | `carla2real/evaluation/deshim_compare.py` |
| `make_compare.py` | `carla2real/visualization/make_compare.py` |
| `make_compare3.py` | `carla2real/visualization/make_compare3.py` |
| `check_reference.py` | `carla2real/validation/check_reference.py` |
| `check_teleport.py` | `carla2real/validation/check_teleport.py` |
| `find_truncated.py` | `carla2real/validation/find_truncated.py` |
| `vidcodec.py` | `carla2real/common/vidcodec.py` |
| `stage_pandaset_channels.sh` | `scripts/preprocessing/stage_pandaset_channels.sh` |
| `stage_pandaset_night.sh` | `scripts/preprocessing/stage_pandaset_night.sh` |
| `stage_zod_channels.sh` | `scripts/preprocessing/stage_zod_channels.sh` |
| `train_generic.sh` | `scripts/training/train_generic.sh` |
| `render_model.sh` | `scripts/inference/render_model.sh` |
| `refresh_new.sh` | `scripts/delivery/refresh_new.sh` |
| `organise_calibrated.sh` | `scripts/delivery/organise_calibrated.sh` |
| `gpu_wait.sh` | `scripts/common/gpu_wait.sh` |
| `train_v50.sh` | `experiments/training/train_v50.sh` |
| `train_v51_night.sh` | `experiments/training/train_v51_night.sh` |
| `train_v63_veg.sh` | `experiments/training/train_v63_veg.sh` |
| `train_v64_veg.sh` | `experiments/training/train_v64_veg.sh` |
| `make_v50d.sh` | `experiments/delivery/make_v50d.sh` |
| `make_v50i.sh` | `experiments/delivery/make_v50i.sh` |
| `make_v50j.sh` | `experiments/delivery/make_v50j.sh` |
| `make_v50kl.sh` | `experiments/delivery/make_v50kl.sh` |
| `make_v50m.sh` | `experiments/delivery/make_v50m.sh` |
| `make_v51d.sh` | `experiments/delivery/make_v51d.sh` |
