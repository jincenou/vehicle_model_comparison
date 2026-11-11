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
