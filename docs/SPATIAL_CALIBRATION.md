# Spatial calibration before comparison

A common metric formula does not make differently scaled or positioned assets comparable. Before rendering or scoring, establish the object's reference frame, coordinate units, centre, accepted heading, and reference cameras. Do not compare each model's default screenshot or interpret its native coordinate unit as a metre.

## Shared transform

The released numerical component uses

`x_reference = s R (x_input - c_input) + c_reference`, with `s > 0`.

`R` must be a proper rotation. Only one scalar `s` is allowed. Nonuniform axis scaling, reflections and shear are rejected. Width/height proportion errors survive length conditioning; they are not stretched to match the reference.

For a **known-length-conditioned shape comparison**, use `s = reference_length / measured_input_length`. This consumes a reference dimension and therefore does not evaluate recovery of absolute physical scale. A scale-recovery experiment must preserve the unconditioned output and be reported separately.

Use the same reference camera intrinsics, extrinsics, canvas and clipping conditions for every model. When a native Gaussian field is retained, factor `s` out of the camera transform and convert clipping planes into native units. This preserves projected pixels and avoids changing Gaussian covariance or SH direction conventions. If transforming geometry explicitly, transform the full covariance as `s² R covariance Rᵀ`, not only its centres.

## What is validated

`evaluation/spatial_calibration.py` implements these transformations. Six CPU tests check unit/centre invariance, preservation of width/height errors, forbidden affine transforms, native-camera projection/depth/clipping invariance, squared covariance scaling and invalid inputs.

The same component was checked against an existing six-method development object (`scene-0064_t000637`). Its prior native display-length measurements ranged from approximately 0.973 to 1.979, while the annotation reference length was 4.673 m. The shared transform reproduces the recorded conditioned transforms and fixed cameras. This is a numerical consistency check, not a new quality result or proof of accurate physical registration.

## What remains unresolved

A mathematically valid transform can still be built from the wrong measured centre or size. The previous display pipeline measured Gaussian-centre quantiles for Gaussian assets and vertex bounds/quantiles for meshes. Those are representation-dependent measurements and are not accepted as a common physical extent for formal metrics. The development object's source reference is also truncated.

A candidate for validation is to measure native foreground alpha support under fixed orthographic views, a common threshold and resolution-convergence checks. The installed gsplat implementation exposes an orthographic camera model, as documented in the [native rasterization API](https://docs.gsplat.studio/versions/1.5.3/apis/rasterization.html). API availability alone does not validate an extent-measurement rule, opaque/transparent material handling, clipping, renderer agreement or reference-centre interpretation. This candidate has **not** been approved or applied as a formal metric protocol.

`configs/spatial_calibration_protocol_draft_v1.json` records that distinction. The incremental preparation worker continues to report `waiting_physical_registration` after accepted semantic orientation. It does not promote the old display heuristic to metric qualification.


## Native alpha measurement pilot

The candidate implementation is now available in `evaluation/alpha_extent_measurement.py` and `research_snapshot/scripts/render_calibration_alpha_pilot.py`. It uses six cardinal orthographic directions, pixel-edge coordinates (pixel centres at `i + 0.5`), and a fixed `uint8 alpha > 127` threshold. Each semantic axis receives four boundary observations; their union defines an extent. Empty support and clipped silhouettes are rejected. Native mesh bounds and Gaussian six-sigma bounds are used only to frame the measurement, never as the measured size. This is an alpha-support convention, not a complete physical surface estimator.

Real pilot renders use 512 and 1024 pixels. The predeclared boundary-stability tolerance is four coarse pixels in native units. Constant Gaussian colors are permitted only to obtain alpha: no appearance scores are produced. Covariances and opacity remain unchanged; the Gaussian pilot uses orthographic gsplat, classic rasterization, `eps2d=0.3`, and zero radius clipping. The mesh pilot uses native materials in Cycles CPU with 16 samples and seed 42. Renderer settings and transparency still require cross-backend qualification.

A 2-by-1-by-0.5 analytic mesh fixture exposed a Blender dependency-graph issue: reading the world matrix before updating newly assigned scale overwrote the intended dimensions. That attempt passed resolution convergence yet failed the known dimensions. The adapter now updates and captures world matrices before applying the accepted semantic rotation, including parent-first assignment. The corrected fixture passed its two-pixel extent tolerance. This demonstrates why convergence alone is insufficient.

One TRELLIS.2 and one Hunyuan mesh for the same development object have each completed twelve native-alpha renders and passed the resolution check. Their native extents were approximately `[1.0091, 0.5789, 0.4382]` and `[1.9937, 0.9103, 0.7114]`; these are native units, not quality scores. Evidence metadata is in `configs/calibration_alpha_pilot_evidence.json`. Raw assets and rendered images are not part of this source release.

The Gaussian fixture and four Gaussian asset pilots are queued locally behind existing generation jobs. The low-priority dispatcher respects shared device locks, active-process reservations, and the 10 GiB disk floor. It does not preempt a busy card, and it stops before real Gaussian pilots if its analytic fixture fails. Historical runtime state and private input paths are required; this dispatcher is not a portable turnkey launcher.

Neither these pilot results nor the 50 passing numerical tests approve a formal protocol. Gaussian verification, transparency/threshold sensitivity, reference-centre interpretation, and broader object/category coverage remain unresolved. All formal scoring gates stay disabled.
