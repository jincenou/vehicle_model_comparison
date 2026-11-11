# Data and input provenance

## Frozen membership

`configs/cohort_rigid500.json` is a public **projection** of the reference-only local manifest. It retains stable IDs, historical categories, cohort origin, relevant hashes, and the existing pipeline's input-mode labels. The full source manifest SHA-256 is recorded separately. The public file has different bytes and must not be presented as having the source manifest's hash.

`configs/scenes_nuscenes100.json` similarly projects the original scene selection. Some scenes may contain no selected final rigid object; all 100 scene memberships remain preserved. Assets are restricted to those scenes.

The total is 432 previously frozen vehicle assets plus 68 accepted non-vehicle assets after removing duplicate/redundant extension entries. Membership is not recomputed from model success. No raw images, masks, videos, LiDAR, assets, pretrained weights, or scene checkpoints are distributed.

## What must be supplied locally

1. Obtain nuScenes independently under its terms.
2. Supply the original pipeline's object-to-source provenance, retained real crops/masks, raw camera and timestamp records, and the existing train/reserved-view split manifests.
3. Resolve IDs against those records and verify file hashes before model execution.
4. Store references to existing files or local symlinks; do not duplicate the dataset to implement the freeze.
5. Keep generated inputs and actual real inputs separately identified in provenance.

The public cohort IDs and hashes are enough to inspect membership, **not enough to reconstruct missing masks, source-to-object mappings, or the original private asset library**. The historical split hashes alone are also not complete train/test split contents. Recreating those inputs independently constitutes a new protocol version unless equivalence is demonstrated.

## Input modes

- AH, original TRELLIS, TRELLIS.2 and Hunyuan use the retained real source observations according to their adapters.
- ReconViaGen consumes eligible, already-used real views from the frozen training split. Missing views are not replaced with FLUX or held-out observations. Preserve available-view counts per object.
- SplatAD, OmniRe and StreetGS use scene reconstruction inputs. Record cameras, temporal context, masks, tracks, LiDAR and other supervision actually consumed; do not describe these as single-image baselines.
- Existing pipeline outputs include 364 `flux_plus_real` and 136 `flux_only` records. These labels describe the historical pipeline, not the external baselines' current inputs.

## Bias and unavailable references

Selection depended on asset-library availability and manual appearance acceptance. It is not random, output-unbiased, or a complete taxonomy benchmark. Historical non-vehicle category labels may require semantic verification before category-specific conclusions. Failed or visually weak baseline predictions remain in coverage reports.

The earlier vehicle input audit found no qualified independent real held-out observations. Never relabel a training frame as held-out. New eligibility requires explicit provenance evidence.
