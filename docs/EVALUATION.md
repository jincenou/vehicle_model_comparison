# Evaluation protocol and eligibility gates

This is a specification for controlled evaluation, not a claim that all metrics have been run. `configs/metric_plan.json` records the current disabled metric entries. Changing a definition, threshold, feature extractor, camera, or eligible subset requires a new protocol identifier.

## Required gates

1. Freeze object identity, source provenance, input split and content hashes.
2. Verify native output serialization in an independent process; check finite geometry/attributes and nonempty render support. Successful export alone is insufficient.
3. Use an accepted semantic object frame (+X front, +Y left, +Z up). For symmetric objects, declare the symmetry rule before scoring. Unreviewed PCA or VLM headings are suggestions, not accepted frames.
4. For pixel comparisons, validate physical scale, translation, rotation, camera intrinsics/extrinsics, image size and source timestamp. Turntable display normalization does not perform this registration.
5. Validate reference identity, masks, occlusion handling and eligibility. Keep missing/invalid references separate from valid but poor model predictions.
6. Aggregate a shared eligible object intersection and also report each method's own coverage out of the fixed 500 requested assets. Never compare unrelated available-subset means without disclosure.

## Source-image metrics

| Metric | Definition and handling |
|---|---|
| IoU | Binary mask intersection / union, byte alpha >127. Empty prediction with nonempty valid reference scores 0. Empty reference is unavailable. Native raw-source canvas is retained for source IoU. |
| PSNR | `10 log10(1 / MSE)` for RGB in [0,1], float64 arithmetic, display RGB. Exact equality returns explicit positive-infinity status, not an invalid JSON number. |
| SSIM | RGB channel-wise implementation, data range 1, Gaussian window 11, sigma 1.5, population covariance, K1=.01 and K2=.03. |
| LPIPS | Planned AlexNet v0.1; fixed package/weights and RGB preprocessing must be recorded. Convert [0,1] to [-1,1] exactly once. No network switch or per-prediction crop. The CPU core does not bundle a LPIPS encoder. |
| ED-R | Foreground DINO feature cosine distance, `1 - cosine`. Encoder, patch/mask correspondence and pooling must be fixed and validated. This is not an edge-distance metric. Only the cosine numerical primitive is released here. |

For source appearance metrics, the reference mask alone determines a square crop with side `ceil(1.10 * max(reference_bbox_width, reference_bbox_height))`. Apply the **same** crop to reference and prediction. Resize to 512×512 using bicubic RGB and nearest-neighbour masks; pad RGB white outside the source image. Follow the fixed protocol's foreground compositing. Prediction-dependent recentering, zooming, mask cleanup or image registration is prohibited.

Retained crops must map to raw pixels using the exact pixel-centre crop/resize transform. An upsampled mask is diagnostic unless native-pixel equivalence has been qualified. Source reconstruction scores are not held-out generalization scores.

## Fixed-orbit feature metrics

I16 averages the similarity between the source feature and **all 16 fixed rendered-view features**. Do not select a best view. The reference rig uses 22.5° azimuth increments, 12° elevation, 40° field of view and 512×512 resolution; distance is determined from the reference object's bounds. Reference physical dimensions and pose must be qualified. CLIP/DINO feature extractor revisions, layers, resizing and normalization require separate fixed manifests.

A semantically aligned preview grid is useful for inspection but does not certify that a model is registered in this physical rig.

## Distribution metrics

The released Fréchet numerical core computes

`||mu_r - mu_p||² + tr(C_r) + tr(C_p) - 2 tr(sqrt(C_r C_p))`

using float64 statistics, unbiased sample covariance (n−1), and centred-factor singular values for the cross term. No feature L2 normalization is applied implicitly. The exact feature extractor defines which metric this represents; arbitrary embeddings do not become Inception FID by calling this function.

FID and related feature-distribution distances are **group metrics**, not per-object scores. Publish extractor revision, sample counts, subsets and preprocessing. Small sample estimates are unstable; the implementation does not certify statistical reliability. Do not average per-object FID or substitute unvalidated feature encoders.

## Geometry metrics: currently ineligible

For qualified surface samples P and Q, define `d(p,Q)` as nearest-neighbour **Euclidean** distance in metres:

- `CD = 0.5 * (mean_p d(p,Q) + mean_q d(q,P))`.
- `Precision@tau = mean_p [d(p,Q) < tau]`.
- `Recall@tau = mean_q [d(q,P) < tau]`.
- `F1@tau = 2PR/(P+R)`, with 0 if the denominator is 0.

The distance is not squared. The common threshold tau has not been frozen. Candidate surface sampling settings (100,000 points and seed 42) are proposals, not qualified final settings.

nuScenes boxes and sparse LiDAR are not complete watertight surface ground truth. Gaussian centres are not surface samples. Until reference surfaces and native-representation surface adapters are independently validated, report these metrics as unavailable. Do not silently replace the geometry protocol with box overlap or Gaussian-centre distance.

## Legacy size diagnostics

Width/height errors and Box extent IoU are retained separately from physical reconstruction metrics. With `s = L_ref / L_pred`, width error is `100 * abs(s * W_pred / W_ref - 1)`; height is analogous. Box extent IoU uses the intersection `prod(min(s * pred_extent, ref_extent))` and the usual volume union, assuming identical centres and semantic orientation. These are length-normalized proportion diagnostics, not absolute scale recovery or surface IoU. Legacy Gaussian-centre extents used the .005/.995 quantiles; cross-representation comparison remains disabled until a shared qualified extent adapter exists. They are specified in the metric plan but not implemented in the released numerical core.

## Reporting

Report requested, generated, reload-qualified, reference-qualified, and metric-qualified counts separately, with explicit failure reasons. Preserve degenerate valid output scores when the metric is defined. Do not replace missing values with zero or discard low-quality predictions after looking at scores. Report the paired subset and per-method coverage alongside aggregate numbers. AOR is outside the present metric scope.

Numerical regression tests cover primitives and gates; they do not replace full renderer, encoder, dataset, registration or end-to-end qualification.
