"""Projection checks for the raw-source camera; no fitted image alignment."""
import hashlib
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
CONTRACT = ROOT / 'configs/common_camera_pilot_v1/camera.json'


def load_contract(sample_id):
    data = json.loads(CONTRACT.read_text())
    assert data['sample_id'] == sample_id
    assert hashlib.sha256(Path(data['source_rgb']).read_bytes()).hexdigest() == data['source_rgb_sha256']
    return data


def projection_audit(contract, model_from_world, model_from_camera_opencv, intrinsics):
    """Check 27 world points at three depths, including image corners/centre."""
    transform = np.asarray(model_from_world, dtype=np.float64)
    camera = np.asarray(model_from_camera_opencv, dtype=np.float64)
    k = np.asarray(intrinsics, dtype=np.float64)
    raw_camera = np.asarray(contract['T_world_from_camera_opencv'])
    raw_k = np.asarray(contract['K_raw'])
    width, height = contract['render_wh']
    pixels = np.array([[x, y, 1.] for x in (0., width/2, width-1.)
                       for y in (0., height/2, height-1.)])
    points = np.concatenate([(np.linalg.inv(raw_k) @ pixels.T).T * z
                             for z in (2., 10., 50.)])
    world = (raw_camera @ np.c_[points, np.ones(len(points))].T).T
    modeled = (transform @ world.T).T
    reprojected = (np.linalg.inv(camera) @ modeled.T).T[:, :3] @ k.T
    uv = reprojected[:, :2] / reprojected[:, 2:3]
    error = float(np.max(np.abs(uv - np.tile(pixels[:, :2], (3, 1)))))
    pose_error = float(np.max(np.abs(camera - transform @ raw_camera)))
    assert error < 0.05, ('raw-source reprojection mismatch in pixels', error)
    assert pose_error < 1e-3, ('raw-source pose mismatch', pose_error)
    return {'contract': str(CONTRACT), 'contract_sha256': hashlib.sha256(CONTRACT.read_bytes()).hexdigest(),
            'projection_probe_count': len(points), 'projection_max_error_px': error,
            'pose_matrix_max_error': pose_error, 'T_model_from_world': transform.tolist(),
            'T_world_from_camera_opencv': raw_camera.tolist(), 'K_raw': raw_k.tolist(),
            'render_wh': [width, height], 'camera_pose_refinement': False,
            'render_time_shift_seconds': 0, 'rolling_shutter_seconds': 0,
            'asset_registration_certified': False}
