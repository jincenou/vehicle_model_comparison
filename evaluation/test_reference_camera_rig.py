import unittest
import numpy as np
from reference_camera_rig import make_rig, project, corners


class ReferenceRigTests(unittest.TestCase):
    def test_reference_in_frustum_and_front_convention(self):
        for dims in [[4.5,1.8,1.6],[16,2.6,4],[.01,.01,.01]]:
            rig=make_rig(dims,np.eye(4))
            self.assertEqual(len(rig['views']),16)
            for v in rig['views']:
                uv,z=project(corners(dims),v['T_camera_from_object'],rig['K'])
                self.assertTrue((uv>=-.5).all()and(uv<511.5).all())
                self.assertTrue((z>v['near_m']).all()and(z<v['far_m']).all())
            # Front view looks from +X; vehicle left projects toward image right.
            uv,z=project([[dims[0]/2,0,0],[0,0,0],[0,dims[1]/2,0]],rig['views'][0]['T_camera_from_object'],rig['K'])
            self.assertLess(z[0],z[1]);self.assertGreater(uv[2,0],uv[1,0])

    def test_world_and_object_cameras_identical_projection(self):
        T=np.eye(4);a=.71;T[:3,:3]=[[np.cos(a),-np.sin(a),0],[np.sin(a),np.cos(a),0],[0,0,1]];T[:3,3]=[1900,750,1.3]
        p=corners([4,2,1.5]);world=p@T[:3,:3].T+T[:3,3]
        rig=make_rig([4,2,1.5],T)
        for v in rig['views']:
            a,za=project(p,v['T_camera_from_object'],rig['K']);b,zb=project(world,np.linalg.inv(v['T_world_from_camera_opencv']),rig['K'])
            np.testing.assert_allclose(a,b,atol=1e-8);np.testing.assert_allclose(za,zb,atol=1e-10)

    def test_bad_reference_rejected(self):
        for dims in [[0,1,1],[-1,2,3],[np.nan,2,3]]:
            with self.assertRaises(ValueError):make_rig(dims,np.eye(4))
        T=np.eye(4);T[0,0]=-1
        with self.assertRaises(ValueError):make_rig([4,2,1.5],T)

if __name__=='__main__':unittest.main()
