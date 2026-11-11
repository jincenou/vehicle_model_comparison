"""Shared similarity calibration math; measurement eligibility is separate.

Column-vector convention. A single positive scale preserves shape proportions.
These functions do not infer object centres, headings, physical dimensions or
visibility from a model. In particular, algebraic validity is not a certificate
that Gaussian-centre and mesh-vertex extent measurements are comparable.
"""
import numpy as np
from asset_orientation_contract import checked_rotation, checked_rigid


def vector3(value, name):
    array = np.asarray(value, dtype=np.float64)
    if array.shape != (3,) or not np.isfinite(array).all():
        raise ValueError(name + ' must be a finite three-vector')
    return array


def positive(value, name):
    value = float(value)
    if not np.isfinite(value) or value <= 0:
        raise ValueError(name + ' must be finite and positive')
    return value


def similarity(rotation, input_center, reference_center, scale):
    """Construct x_reference = scale * R * (x_input - c_input) + c_reference."""
    rotation = checked_rotation(rotation)
    source = vector3(input_center, 'input_center')
    target = vector3(reference_center, 'reference_center')
    scale = positive(scale, 'scale')
    transform = np.eye(4)
    transform[:3, :3] = scale * rotation
    transform[:3, 3] = target - scale * (rotation @ source)
    validate_similarity(transform)
    return transform


def validate_similarity(value):
    transform = np.asarray(value, dtype=np.float64)
    if (transform.shape != (4, 4) or not np.isfinite(transform).all()
            or not np.allclose(transform[3], [0, 0, 0, 1], rtol=0, atol=1e-12)):
        raise ValueError('Invalid homogeneous similarity transform')
    # No axis-by-axis stretching, reflections, shear or arbitrary projective fit.
    lengths = np.linalg.norm(transform[:3, :3], axis=0)
    scale = positive(lengths.mean(), 'scale')
    if not np.allclose(lengths, scale, rtol=1e-8, atol=1e-12):
        raise ValueError('Nonuniform scale is forbidden')
    checked_rotation(transform[:3, :3] / scale)
    return scale, transform[:3, :3] / scale


def length_conditioned(rotation, input_center, input_length, reference_length,
                       reference_center=(0, 0, 0)):
    """Known-length shape-conditioning transform, not absolute-scale recovery."""
    source = positive(input_length, 'input_length')
    target = positive(reference_length, 'reference_length')
    transform = similarity(rotation, input_center, reference_center, target/source)
    return dict(T_reference_from_input=transform.tolist(),
                scale_reference_units_per_input_unit=target/source,
                calibration_mode='known_length_conditioned_shape',
                absolute_scale_recovery_evaluated=False,
                shape_proportions_preserved=True,
                measurement_qualification_required=True,
                formal_metrics_qualified=False)


def transform_points(points, transform):
    validate_similarity(transform)
    transform = np.asarray(transform, dtype=np.float64)
    points = np.asarray(points, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 3 or not np.isfinite(points).all():
        raise ValueError('Expected finite Nx3 points')
    return points @ transform[:3, :3].T + transform[:3, 3]


def native_camera(camera_from_reference, reference_from_input, near, far):
    """Keep a native Gaussian field intact; move the camera and clipping planes.

Factoring scale out of all three camera rows preserves perspective pixels. This
avoids rotating positions without covariance, or rotating SH coefficients in an
unverified way. Metric reference-depth values equal native depth times scale.
"""
    camera = checked_rigid(camera_from_reference)
    scale, _ = validate_similarity(reference_from_input)
    near, far = positive(near, 'near'), positive(far, 'far')
    if far <= near:
        raise ValueError('far must exceed near')
    view = camera @ np.asarray(reference_from_input, dtype=np.float64)
    view[:3, :] /= scale
    checked_rigid(view)
    return dict(T_camera_from_input=view.tolist(), near_input_units=near/scale,
                far_input_units=far/scale, reference_depth_per_input_depth=scale)


def transform_covariances(covariances, transform):
    """Full covariance transformation if geometry is explicitly transformed."""
    scale, rotation = validate_similarity(transform)
    cov = np.asarray(covariances, dtype=np.float64)
    if (cov.ndim < 2 or cov.shape[-2:] != (3, 3) or not np.isfinite(cov).all()
            or not np.allclose(cov, cov.swapaxes(-1, -2), rtol=0, atol=1e-10)
            or np.any(np.linalg.eigvalsh(cov) < -1e-10)):
        raise ValueError('Covariance must be finite symmetric positive semidefinite')
    return scale**2 * (rotation @ cov @ rotation.T)
