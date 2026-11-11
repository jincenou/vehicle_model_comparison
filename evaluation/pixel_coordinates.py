"""Explicit pixel-origin conversion; never fit or resample an output image.

integer_centres: pixel (i,j) has coordinate (j,i).
pixel_edges: pixel (i,j) has coordinate (j+.5,i+.5).
gsplat samples pixel_edges; reference orbit contracts use integer_centres.
"""
import numpy as np

OFFSETS = {'integer_centres': 0., 'pixel_edges': .5}


def intrinsics_for_convention(intrinsics, source, destination):
    if source not in OFFSETS or destination not in OFFSETS:
        raise ValueError('Pixel conventions must be explicit and recognized')
    k = np.asarray(intrinsics, dtype=np.float64)
    if (k.shape != (3, 3) or not np.isfinite(k).all()
            or not np.allclose(k[2], [0, 0, 1], rtol=0, atol=1e-12)
            or k[0, 0] <= 0 or k[1, 1] <= 0 or abs(np.linalg.det(k)) < 1e-12):
        raise ValueError('Invalid camera intrinsics')
    shift = np.eye(3)
    shift[:2, 2] = OFFSETS[destination] - OFFSETS[source]
    return shift @ k
