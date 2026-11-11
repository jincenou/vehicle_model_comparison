import unittest
import numpy as np
from pixel_coordinates import intrinsics_for_convention


class PixelCoordinatesTests(unittest.TestCase):
    def test_symmetric_camera_and_roundtrip(self):
        k = np.array([[700., 0, 255.5], [0, 800., 255.5], [0, 0, 1]])
        edge = intrinsics_for_convention(k, 'integer_centres', 'pixel_edges')
        np.testing.assert_array_equal(edge[:2, 2], [256, 256])
        np.testing.assert_array_equal(intrinsics_for_convention(edge, 'pixel_edges', 'integer_centres'), k)
        np.testing.assert_array_equal(k[:2, 2], [255.5, 255.5])

    def test_sampling_rays_match_with_skew_and_offcentre_principal_point(self):
        k = np.array([[700., 5., 307.2], [0, 810., 241.7], [0, 0, 1]])
        edge = intrinsics_for_convention(k, 'integer_centres', 'pixel_edges')
        pixels = np.array([[0., 0., 1.], [203., 117., 1.], [639., 479., 1.]])
        sampled = pixels + np.array([.5, .5, 0])
        np.testing.assert_allclose(pixels @ np.linalg.inv(k).T, sampled @ np.linalg.inv(edge).T, atol=1e-14)
        points = np.array([[0., 0., 1.], [1., 2., 3.]])
        reference = points @ k.T; native = points @ edge.T
        np.testing.assert_allclose(native[:, :2]/native[:, 2:] - .5,
                                   reference[:, :2]/reference[:, 2:], atol=1e-12)

    def test_unconverted_raster_has_half_pixel_shift(self):
        k = np.array([[700., 0, 255.5], [0, 700., 255.5], [0, 0, 1]])
        # At the optical axis, gsplat's array-index location is cx-.5.
        old_array_index = k[:2, 2] - .5
        np.testing.assert_array_equal(old_array_index - k[:2, 2], [-.5, -.5])
        edge = intrinsics_for_convention(k, 'integer_centres', 'pixel_edges')
        np.testing.assert_array_equal(edge[:2, 2] - .5, k[:2, 2])

    def test_invalid_intrinsics_and_unknown_convention_rejected(self):
        for k in [np.zeros((3, 3)), np.eye(4), np.full((3, 3), np.nan)]:
            with self.assertRaises(ValueError):
                intrinsics_for_convention(k, 'integer_centres', 'pixel_edges')
        with self.assertRaises(ValueError):
            intrinsics_for_convention(np.eye(3), 'unspecified', 'pixel_edges')


if __name__ == '__main__':
    unittest.main()
