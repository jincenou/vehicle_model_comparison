"""Reference-only metric-space cameras. Does not register a predicted asset."""
import itertools
import numpy as np


def check_pose(T):
    T = np.asarray(T, dtype=np.float64)
    if T.shape != (4, 4) or not np.isfinite(T).all():
        raise ValueError('invalid pose shape or nonfinite pose')
    if not np.allclose(T[3], [0, 0, 0, 1], atol=1e-9):
        raise ValueError('invalid homogeneous row')
    R = T[:3, :3]
    if not np.allclose(R.T @ R, np.eye(3), atol=1e-7) or not np.isclose(np.linalg.det(R), 1, atol=1e-7):
        raise ValueError('pose must be rigid and right handed')
    return T


def corners(length_width_height):
    dims = np.asarray(length_width_height, dtype=np.float64)
    if dims.shape != (3,) or not np.isfinite(dims).all() or (dims <= 0).any():
        raise ValueError('positive finite length width height required')
    return np.asarray(list(itertools.product([-1, 1], repeat=3))) * dims / 2


def project(points, camera_from_object, K):
    T = check_pose(camera_from_object)
    xyz = np.asarray(points) @ T[:3, :3].T + T[:3, 3]
    uvh = xyz @ np.asarray(K).T
    return uvh[:, :2] / uvh[:, 2:3], xyz[:, 2]


def make_rig(length_width_height, world_from_object):
    """Object +X front, +Y left, +Z up; camera +X right,+Y down,+Z forward."""
    Two = check_pose(world_from_object)
    pts = corners(length_width_height)
    radius = float(np.linalg.norm(np.asarray(length_width_height) / 2))
    distance = 1.10 * radius / np.sin(np.deg2rad(20))
    focal = 256 / np.tan(np.deg2rad(20))
    K = np.array([[focal, 0, 255.5], [0, focal, 255.5], [0, 0, 1.]])
    views = []
    for i in range(16):
        az, el = np.deg2rad([i * 22.5, 12])
        direction = np.array([np.cos(az)*np.cos(el), np.sin(az)*np.cos(el), np.sin(el)])
        eye = distance * direction
        forward = -direction
        right = np.cross(forward, [0, 0, 1.])
        right /= np.linalg.norm(right)
        down = np.cross(forward, right)
        R = np.stack([right, down, forward])
        Tco = np.eye(4); Tco[:3, :3] = R; Tco[:3, 3] = -R @ eye
        check_pose(Tco)
        uv, depth = project(pts, Tco, K)
        near, far = max(1e-4, distance-1.1*radius), distance+1.1*radius
        if not ((depth > near).all() and (depth < far).all() and (uv >= -.5).all() and (uv < 511.5).all()):
            raise ValueError('reference box leaves common camera frustum')
        views.append(dict(index=i, azimuth_degrees=i*22.5, elevation_degrees=12,
                          T_camera_from_object=Tco.tolist(),
                          T_world_from_camera_opencv=(Two @ np.linalg.inv(Tco)).tolist(),
                          near_m=near, far_m=far,
                          reference_corner_uv_min=uv.min(0).tolist(),
                          reference_corner_uv_max=uv.max(0).tolist()))
    return dict(schema='reference_box_orbit16_v1', render_wh=[512,512],
                pixel_convention='integer pixel centres; principal point (255.5,255.5)',
                K=K.tolist(), vertical_fov_degrees=40, reference_radius_m=radius,
                distance_m=distance, views=views,
                predicted_asset_bounds_used=False, asset_registration_certified=False,
                reference_is_complete_surface_gt=False)
