# Incremental evaluation preparation

This CPU-only stage prepares existing outputs while generation continues. It preserves **500 objects × 9 methods = 4,500 records** and references native artifacts in place. No images, assets, or checkpoints are duplicated. It does not claim a complete final benchmark.

## Entry points

After providing the local experiment manifests and artifacts described in DATA.md:

```bash
python research_snapshot/scripts/prepare_rigid500_evaluation.py \
  --workspace "$RIGID_WORKSPACE" --hash-budget-mib 1024

# A single persistent worker; exclusive file lock prevents duplicate workers.
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  nice -n 15 ionice -c 3 \
  python -u research_snapshot/scripts/prepare_rigid500_evaluation.py \
  --workspace "$RIGID_WORKSPACE" --watch --interval 120 --hash-budget-mib 4096
```

The source workspace version of the command is `scripts/prepare_rigid500_evaluation.py`. The public snapshot can import the released CPU components from `evaluation/`. The default output is `evaluation_nuscenes_v1/preparation_rigid500_v1` inside the specified workspace. Do not launch two workers for the same output directory.

This is **not** a data-free quick start. It requires the existing 500-object freeze, original input and generation manifests, completed source-camera batches, reviewed reference evidence, and any accepted heading decisions. Reconstructing private source mappings from IDs alone is not supported.

## What each pass does

1. Read coherent snapshots of the frozen membership and current generation queues, including reconciled TRELLIS.2 progress.
2. Preserve pending, failed and unavailable objects. Reconsider them automatically in the next pass if authoritative generation state changes.
3. Verify actual input records: single real RGBA, real multi-view train-only records for ReconViaGen, or AH image/mask provenance. Existing pipeline input regimes remain separately disclosed.
4. Check available native PLY/GLB content against recorded SHA-256 values under a bounded I/O budget. An unprocessed file is queued rather than marked validated. A single oversized file may exceed a pass budget to avoid permanent starvation.
5. Export a small semantic coordinate contract only when an existing accepted heading decision matches the exact asset hash. Do not invent a front direction for non-vehicles. No geometry, opacity, appearance or pose is altered.
6. For native scene renders, independently check qualified reference identity, time, camera, validation hashes, image dimensions and current render hashes. Preserve technically valid degenerate outputs instead of filtering them for poor appearance.
7. Write an atomic manifest, summary, worker status and content-check cache. All formal metric flags remain false.

## Output states

| State | Meaning / next action |
|---|---|
| `waiting_generation` | Native job has not completed; re-read next pass |
| `native_output_unavailable` | Failed or blocked native job; retain reason and denominator |
| `queued_integrity_check` | I/O budget exhausted; resume content checks next pass |
| `evidence_error` | Missing file, changed hash, identity mismatch or another intake failure; never promote silently |
| `waiting_semantic_heading` | Native artifact verified; an accepted heading is still missing |
| `waiting_nonvehicle_axes_and_symmetry` | Non-vehicle category/object frame policy is not yet qualified |
| `waiting_physical_registration` | Accepted semantic frame exported; source-camera physical placement remains unqualified |
| `waiting_reference_qualification` | Reference-side conditions not satisfied or not yet reviewed |
| `waiting_native_scene_output` | A qualified reference exists but its native scene render is unavailable |
| `source_camera_intake_ready` | Native scene identity/camera/reference intake passes; scoring-stage verification still required |

`artifact_integrity_verified` does not imply good reconstruction quality. `source_camera_intake_ready` does not imply all nine methods are comparable, an independent held-out view exists, a complete geometry reference exists, or formal metrics are enabled.

## Reference and provenance limits

Six existing reference-side reviews cover the 432 vehicles. Their merged acceptance file is `configs/source_reference_vehicle432_v1/acceptance_all432_20261111.json`; the recorded review outcome is 177 accepted visible references. This is reference-side audit evidence, not 177 physically registered nine-method assets, and does not cover the additional 68 non-vehicles. The merger verifies the bound review records, decision sources, contracts and review-image hashes. Asset membership is unchanged.

Some historical AH source manifests bind the image digest but do not record a mask digest. Those masks are hashed at intake and explicitly labeled `real_image_verified_historical_mask_hash_missing`; the tool does not invent a historical hash or claim original-mask verification.

Native source preview hashes may likewise be first bound at preparation intake because the original batch binds only the validation record. That origin is recorded. The final scoring stage must verify its selected files against the prepared hashes again.

## Cache and resume behavior

Cache keys include resolved paths, device/inode, size, modification and change timestamps. The expected digest is checked even on cache hits. Recent writes are rehashed to avoid coarse-timestamp reuse. Changed or missing files invalidate successful intake; symlink retargeting is detected. This is an efficiency cache for immutable experiment artifacts, not a replacement for final scoring-time content verification.

The six cache regression tests cover same-size mutation, changed expected hashes, resume without rereading settled files, I/O deferral, oversized-file progress, disappearance and symlink retargeting:

```bash
python -m unittest discover -s evaluation -p 'test_preparation_cache.py' -v
```

The worker imports no model stacks, uses no CUDA context and claims no GPU lease. It does not change native generation settings, shared job states, the accepted cohort, or the cancelled non-vehicle extraction-repair branch.
