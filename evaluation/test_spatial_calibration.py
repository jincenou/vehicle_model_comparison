import unittest
import numpy as np
from spatial_calibration import (similarity, validate_similarity, length_conditioned,
                                 transform_points, native_camera, transform_covariances)


class SpatialCalibrationTests(unittest.TestCase):
    def test_different_units_and_centres_share_one_reference_shape(self):
        points = np.array([[-2,-1,-.5], [2,1,.5], [.7,-.1,.3]])
        rotation = np.array([[0,-1,0],[1,0,0],[0,0,1.]])
        center = np.array([11.,-4,9])
        native = (points @ rotation)*7 + center
        a = length_conditioned(np.eye(3), [0,0,0], 4, 4)
        b = length_conditioned(rotation, center, 28, 4)
        np.testing.assert_allclose(transform_points(points,a['T_reference_from_input']),
                                   transform_points(native,b['T_reference_from_input']),atol=1e-12)
        self.assertFalse(a['absolute_scale_recovery_evaluated'])
        self.assertFalse(a['formal_metrics_qualified'])

    def test_width_height_errors_are_not_removed_by_length_conditioning(self):
        native = np.array([[-1,-.5,-.25],[1,.5,.25]])
        contract = length_conditioned(np.eye(3), [0,0,0], 2, 4)
        transformed = transform_points(native, contract['T_reference_from_input'])
        np.testing.assert_allclose(np.ptp(transformed,axis=0), [4,2,1])
        # A desired reference width/height of 3/2 is deliberately not forced.
        self.assertNotEqual(float(np.ptp(transformed,axis=0)[1]),3)

    def test_nonuniform_scale_reflection_shear_and_nonfinite_rejected(self):
        bad=[]
        t=np.eye(4);t[0,0]=2;bad.append(t)
        t=np.eye(4);t[0,0]=-1;bad.append(t)
        t=np.eye(4);t[0,1]=.2;bad.append(t)
        t=np.eye(4);t[0,3]=np.nan;bad.append(t)
        for t in bad:
            with self.assertRaises(ValueError):validate_similarity(t)

    def test_native_camera_projection_depth_and_clipping_invariant(self):
        rotation=np.array([[0,-1,0],[1,0,0],[0,0,1.]])
        a=similarity(rotation,[3,4,5],[0,0,0],2.5)
        camera=np.eye(4);camera[2,3]=30
        native=np.array([[3.,4,5],[4,5,6],[2,3,4]])
        k=np.array([[900,0,800],[0,870,450],[0,0,1.]])
        result=native_camera(camera,a,.1,100)
        ref=transform_points(native,a)@camera[:3,:3].T+camera[:3,3]
        v=np.asarray(result['T_camera_from_input']);xyz=native@v[:3,:3].T+v[:3,3]
        p=ref@k.T;q=xyz@k.T
        np.testing.assert_allclose(p[:,:2]/p[:,2:],q[:,:2]/q[:,2:],atol=1e-10)
        np.testing.assert_allclose(ref[:,2],xyz[:,2]*2.5)
        self.assertAlmostEqual(result['near_input_units'],.04)
        self.assertAlmostEqual(result['far_input_units'],40)

    def test_covariance_uses_squared_scale_and_full_rotation(self):
        rotation=np.array([[0,-1,0],[1,0,0],[0,0,1.]])
        a=similarity(rotation,[0,0,0],[0,0,0],3)
        actual=transform_covariances(np.diag([1,4,9]),a)
        np.testing.assert_allclose(actual,np.diag([36,9,81]))

    def test_invalid_lengths_and_clipping_rejected(self):
        for length in [0,-1,np.nan,np.inf]:
            with self.assertRaises(ValueError):length_conditioned(np.eye(3),[0,0,0],length,4)
        with self.assertRaises(ValueError):native_camera(np.eye(4),np.eye(4),10,1)


if __name__=='__main__':unittest.main()
