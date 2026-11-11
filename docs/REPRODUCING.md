# Reproduction guide

## 1. Verify the released components

Run the CPU quick start from the repository root. `requirements-eval.txt` records versions installed in the validated evaluation environment. It is not a GPU requirements file. Numerical tests are data-free and do not import the native model stacks.

```bash
python scripts/validate_release.py
python -m unittest discover -s evaluation -p 'test_*.py' -v
```

## 2. Resolve data and deployment roots

Obtain the external data and provenance listed in [DATA.md](DATA.md). The snapshot uses relative paths internally and configurable deployment roots for former machine-specific literals; see `configs/paths.example.json`.

```bash
export RIGID_WORKSPACE="$PWD/research_snapshot"
export RIGID_DATA_ROOT=/your/data/root
export RIGID_CONDA_ROOT=/your/conda
export RIGID_PROJECT_ROOT=/your/project/root
export RIGID_ASSETBENCH=/your/assetbench
export RIGID_DRIVESTUDIO=/your/drivestudio
```

These variables do not rewrite JSON manifests automatically. Rebuild local manifests with real paths and record their hashes. Do not point this checkout at a live experiment's writable state directories. `RIGID_HOME` and `RIGID_EXTERNAL_ROOT` are legacy dependency roots only when needed by a referenced helper.

## 3. Prepare separate native model environments

| Method | Recorded setup | Required local components |
|---|---|---|
| TRELLIS.2 | Python 3.10; torch 2.6.0 / torchvision 0.21.0, CUDA 12.4 wheels | Pinned TRELLIS.2 source, native CUDA extensions, main weights, DINOv3, RMBG and shader/export dependencies |
| Hunyuan3D 2.1 | Python 3.10; torch 2.5.1 / torchvision 0.20.1, CUDA 12.4 wheels | Shape/paint weights, DINOv2, RealESRGAN, renderer extensions and texture configuration |
| ReconViaGen v0.2 | Python 3.10; torch 2.4.0 / torchvision 0.19.0, CUDA 12.1 wheels | Native source, weights and renderer extensions; recorded NumPy 1.26.4 and plyfile 1.1.2 constraints |
| Original TRELLIS | Legacy Gaussian pipeline environment | Native TRELLIS modules, Gaussian weights, legacy loader and native renderer |
| AH | Isolated original AH environment | Native source/weights, C-RADIO/VAE dependencies and recorded local isolation patch |
| Scene methods | Separate native scene environments | Upstream source/patches, scene conversion, cameras, tracks, supervision and checkpoints |

These are recorded experiment settings, not a universal cross-platform installation guarantee. See the preparation scripts in `research_snapshot/model_preparation/20261109_2100_CST/scripts/`. They expect local plans and may have operational side effects; inspect them before executing. Do not run all model stacks inside the evaluation venv.

Source pins recorded by this experiment include:

- TRELLIS.2: `75fbf0183001ed9876c8dbb35de6b68552ee08bd`
- Hunyuan3D 2.1: `82920d643c0dc2f7bfd7255f45f62d386edfe60c`
- ReconViaGen v0.2: `f6720923260927333e22761fa722144d214733d9`
- Asset Harvester: `4e87f859f373cd468c24aca78ebfdacb45d99f33`
- Historical standalone Street Gaussians checkout: `0924b0c4e22e35c2c16673c48c14e50897a10610`. This is **not** the implementation used for the frozen nuScenes100 StreetGS baseline. That baseline runs DriveStudio `configs/streetgs.yaml`; its checked config SHA-256 is `194c6b2890b7514a04fa01192d66f58c4f5fbc353041d6ddb6a0003130557cd5`. A complete immutable DriveStudio source/patch inventory is still required for external replay.

A source revision does not capture uncommitted local patches. The release does not include complete upstream working-tree patches for every scene method, so their historical replay remains incomplete. Original TRELLIS's legacy loader is also a separate local dependency. Do not claim stock upstream reproduces these results without checking that dependency.

## 4. Stage inputs and run native adapters

Read the relevant scripts before running them. The snapshot keeps historical names (`vehicle432`, `rigid111`) to preserve lineage; these names do not redefine the final 500-object membership.

| Stage | Representative script in the snapshot |
|---|---|
| Single real RGBA staging | `scripts/stage_vehicle432_image_inputs.py`, `scripts/stage_rigid111_real_inputs.py` |
| Real multi-view staging | `scripts/prepare_recon02_real_multiview.py`, `scripts/stage_recon02_rigid111_real_views.py` |
| Frozen StreetGS baseline | `scripts/run_streetgs_drivestudio_nusc_queue.py` (existing 72 scenes), then `scripts/run_nuscenes100_expansion_queue.py` (additional 28 scenes) |
| Scene conversion / training | `scripts/prepare_nuscenes100_drivestudio.py`, `scripts/run_nuscenes100_expansion_queue.py` |
| Original TRELLIS | `model_preparation/20261109_2100_CST/scripts/run_trellis_original_object.py` |
| TRELLIS.2 | `model_preparation/20261109_2100_CST/scripts/run_trellis2_object.py` |
| Hunyuan3D | `model_preparation/20261109_2100_CST/scripts/run_hunyuan21_frozen_object.py` |
| ReconViaGen | `model_preparation/20261109_2100_CST/scripts/run_recon02_frozen_object.py` |
| AH | `model_preparation/20261109_2100_CST/scripts/run_ah_rigid111_object.py` |
| Scene actor rendering | `scripts/extract_omnire_native_actor.py`, `scripts/extract_streetgs_native_actor.py`, `scripts/render_splatad_native_actor.py` |

The StreetGS nuScenes controller explicitly overrides the upstream config with `dataset=nuscenes/6cams`, `data.pixel_source.load_smpl=false` and `data.pixel_source.test_image_stride=0`. It trains 30,000 iterations after the 100-step native pilot. The upstream YAML default dataset name must not be mistaken for the actual overridden nuScenes input. Historical `run_streetgs_stage_queue.py` and `extract_streetgs_native_actor.py` refer to the older standalone path; do not use them as the frozen DriveStudio baseline. Its common-camera rendering uses `render_drivestudio_common_camera_vehicle.py`.

For example, **only after** supplying the expected source directory, weights, manifest, environment and output root:

```bash
P="$RIGID_WORKSPACE/model_preparation/20261109_2100_CST"
# Replace SAMPLE_ID with an existing object in the staged manifest.
CUDA_VISIBLE_DEVICES=0 python "$P/scripts/run_trellis2_object.py" \
  --sample-id SAMPLE_ID --run-dir "$P/runs/trellis2_vehicle432_v1/SAMPLE_ID"
CUDA_VISIBLE_DEVICES=0 python "$P/scripts/run_trellis2_object.py" \
  --sample-id SAMPLE_ID --run-dir "$P/runs/trellis2_vehicle432_v1/SAMPLE_ID" --reload-only
```

This example is deliberately not the quick start: missing licensed weights or private provenance must fail rather than be silently substituted. Script assertions and output directory restrictions protect against overwrites and protocol drift. A source subset with absent identity support cannot be exported as a qualified actor merely because a scene checkpoint exists.

## 5. Independent reload and scheduling

Generation, serialization, fresh-process native reload, and exported-GLB reload are distinct stages. Wait for successful process exit and verify recorded hashes. For Gaussian serialization, `recon02_serialization.py` preserves native opacity logits instead of a potentially saturating sigmoid/logit round trip. Native reload previews are integrity checks, not common-camera metrics.

GPU workers must claim exclusive per-device leases, respect other processes and retain a 10 GiB free-disk launch floor. Eight GPUs are a capacity target, not a requirement for CPU evaluation. Long mesh UV/export phases may be CPU-bound. Do not treat low instantaneous GPU utilization as proof that an active job is stuck. The eight-GPU controller is a historical six-supplement-plus-two-legacy handoff implementation, not a standalone new eight-worker launcher. Its tests mock the GPU/process boundaries. Do not launch historical queue controllers against existing state or blindly rerun failed objects with changed seeds/settings.

## 6. Qualify references, then evaluate

Follow [EVALUATION.md](EVALUATION.md). Keep formal metric entries disabled until the associated reference, representation, camera and encoder gates are actually satisfied. Generation completion alone does not enable geometry, held-out, or source-camera metrics.

## Failure evidence audit

`research_snapshot/scripts/audit_trellis2_failures.py` reads the frozen-cohort queue snapshots, generation state, input hashes, logs and native/reload previews, and writes a separate report. It distinguishes empty sparse-coordinate generation from exact serialization with insufficient rendered alpha support. It never retries a job, edits queue status, changes a model parameter or relaxes an acceptance gate. A visibility failure is not automatically a corrupted checkpoint; preserve both native-artifact availability and the failed support gate in coverage reporting. The script requires local experiment artifacts and is not part of the data-free quick start.
