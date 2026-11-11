import unittest
import numpy as np
from asset_orientation_contract import (make_contract, camera_from_semantic,
                                        transform_points, rotate_gaussian_covariances)


class OrientationContractTests(unittest.TestCase):
    def setUp(self):
        self.row = dict(sample_id='s', method='m', asset_sha256='hash', asset='unused', representation='gaussian_ply')
        self.rotation = np.array([[0., -1, 0], [1, 0, 0], [0, 0, 1]])
        self.decision = dict(sample_id='s', method='m', asset_sha256='hash',
                             status='accepted_semantic_heading', native_to_semantic=self.rotation.tolist())
        self.contract = make_contract(self.row, self.decision, verify_asset=False)

    def test_changing_frame_preserves_projected_pixels_and_depth(self):
        # Nonzero translation and anisotropic intrinsics expose a missing inverse.
        points = np.array([[.3, -.2, 1], [1., .8, 2], [-.4, .9, 3]])
        camera = np.eye(4)
        camera[:3, 3] = [1.2, -.5, 6.]
        k = np.array([[1200., 0, 800], [0, 900., 450], [0, 0, 1.]])
        before = transform_points(points, camera)
        semantic = transform_points(points, self.contract['T_semantic_from_input'])
        after = transform_points(semantic, camera_from_semantic(self.contract, camera, 'ply_native_xyz'))
        np.testing.assert_allclose(before, after, atol=1e-12)
        a, b = before @ k.T, after @ k.T
        np.testing.assert_allclose(a[:, :2]/a[:, 2:], b[:, :2]/b[:, 2:], atol=1e-10)

    def test_full_gaussian_covariance_keeps_projected_shape(self):
        cov = np.array([[[4., .3, 0], [.3, 1., .2], [0, .2, .5]]])
        rotated = rotate_gaussian_covariances(cov, self.rotation)
        camera = camera_from_semantic(self.contract, np.eye(4), 'ply_native_xyz')[:3, :3]
        np.testing.assert_allclose(camera @ rotated @ camera.T, cov, atol=1e-12)

    def test_rejects_guess_reflection_identity_mismatch_and_wrong_mesh_frame(self):
        for change in [dict(status='consistent_heading_proposal'), dict(asset_sha256='other'),
                       dict(native_to_semantic=np.diag([-1, 1, 1]).tolist())]:
            with self.assertRaises(ValueError):
                make_contract(self.row, dict(self.decision, **change), verify_asset=False)
        with self.assertRaises(ValueError):
            camera_from_semantic(self.contract, np.eye(4), 'raw_gltf_y_up')
        with self.assertRaises(ValueError):
            rotate_gaussian_covariances(np.diag([-1., 1, 1]), self.rotation)


if __name__ == '__main__': unittest.main()
