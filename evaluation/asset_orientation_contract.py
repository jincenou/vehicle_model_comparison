"""Shared semantic orientation and camera-coordinate contract, without metric fitting.

All matrices act on column vectors. Semantic rotation is rigid and unit-preserving.
Display centering/scaling is explicitly separate from physical registration.
"""
import hashlib
import json
from pathlib import Path

import numpy as np

ACCEPTED = {'accepted_semantic_heading', 'published_semantic_frame_reused',
            'example_heading_visually_reviewed'}


def checked_rotation(value):
    r = np.asarray(value, dtype=np.float64)
    if r.shape != (3, 3) or not np.isfinite(r).all():
        raise ValueError('rotation must be finite 3x3')
    if not np.allclose(r @ r.T, np.eye(3), atol=1e-6, rtol=0) or not np.isclose(np.linalg.det(r), 1, atol=1e-6):
        raise ValueError('rotation must be proper; reflections are forbidden')
    return r


def checked_rigid(value):
    t = np.asarray(value, dtype=np.float64)
    if t.shape != (4, 4) or not np.isfinite(t).all() or not np.allclose(t[3], [0, 0, 0, 1]):
        raise ValueError('expected finite homogeneous 4x4 transform')
    checked_rotation(t[:3, :3])
    return t


def make_contract(row, decision, verify_asset=True):
    if decision.get('status') not in ACCEPTED:
        raise ValueError('heading is not accepted; a VLM suggestion or PCA axis is insufficient')
    for key in ('sample_id', 'method', 'asset_sha256'):
        if row[key] != decision[key]:
            raise ValueError('heading identity/hash mismatch: ' + key)
    if verify_asset and hashlib.sha256(Path(row['asset']).read_bytes()).hexdigest() != row['asset_sha256']:
        raise ValueError('asset content changed since heading review')
    rotation = checked_rotation(decision['native_to_semantic'])
    transform = np.eye(4)
    transform[:3, :3] = rotation
    # Mesh review is performed after Blender imports glTF. Do not apply this
    # matrix directly to raw glTF Y-up vertices or an unflattened scene graph.
    frame = {'gaussian_ply': 'ply_native_xyz',
             'glb_mesh': 'blender_gltf_import_world_z_up'}[row['representation']]
    return dict(schema='asset_orientation_contract_v1', sample_id=row['sample_id'],
                method=row['method'], asset=row['asset'], asset_sha256=row['asset_sha256'],
                input_coordinate_frame=frame, semantic_axes=dict(front='+X', left='+Y', up='+Z'),
                T_semantic_from_input=transform.tolist(), preserves_source_units=True,
                semantic_origin='unchanged native origin rotated; no recentering',
                heading_evidence=decision, heading_qualified=True,
                physical_registration_qualified=False, display_normalization_is_metric_registration=False,
                restrictions=['Camera transforms must use the same input coordinate frame.',
                              'Source/held-out pixel metrics require independent physical registration and camera validation.',
                              'Surface geometry metrics still require qualified reference surface and fixed scale protocol.'])


def transform_points(points, transform):
    t = checked_rigid(transform)
    p = np.asarray(points, dtype=np.float64)
    if p.ndim != 2 or p.shape[1] != 3 or not np.isfinite(p).all():
        raise ValueError('expected finite Nx3 points')
    return p @ t[:3, :3].T + t[:3, 3]


def camera_from_semantic(contract, camera_from_input, input_coordinate_frame):
    """Change coordinates without changing any physical ray or projected pixel."""
    if input_coordinate_frame != contract['input_coordinate_frame']:
        raise ValueError('camera/asset coordinate frame mismatch')
    camera = checked_rigid(camera_from_input)
    canonical = checked_rigid(contract['T_semantic_from_input'])
    return camera @ np.linalg.inv(canonical)


def rotate_gaussian_covariances(covariances, rotation):
    """Rotate full Gaussian shape, not only center positions; SH remains native."""
    r = checked_rotation(rotation)
    cov = np.asarray(covariances, dtype=np.float64)
    if cov.ndim < 2 or cov.shape[-2:] != (3, 3) or not np.isfinite(cov).all():
        raise ValueError('expected finite (...,3,3) covariances')
    if not np.allclose(cov, np.swapaxes(cov, -1, -2), atol=1e-8) or np.any(np.linalg.eigvalsh(cov) < -1e-10):
        raise ValueError('covariance must be symmetric positive semidefinite')
    return r @ cov @ r.T


def load_contract(path):
    result = json.loads(Path(path).read_text())
    if result.get('schema') != 'asset_orientation_contract_v1':
        raise ValueError('unsupported orientation contract')
    checked_rigid(result['T_semantic_from_input'])
    return result
