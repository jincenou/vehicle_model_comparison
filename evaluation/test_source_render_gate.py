import copy
import unittest
import numpy as np
from source_render_gate import validate_native_source

class RenderGateTests(unittest.TestCase):
    def fixture(self):
        T=np.eye(4).tolist();K=np.eye(3).tolist()
        c=dict(sample_id='s',native_instance_token='i',timestamp_us=123,source_rgb='/raw.jpg',render_wh=[1600,900],K_raw=K,T_world_from_camera_opencv=T)
        ref=dict(sample_id='s',native_instance_token='i',timestamp_us=123,raw_source='/raw.jpg',identity_status='recorded_identity_supported_at_source',native_reference_pixels_verified=True)
        a=dict(contract_sha256='h',camera_pose_refinement=False,render_time_shift_seconds=0,rolling_shutter_seconds=0,projection_max_error_px=.001,pose_matrix_max_error=1e-6,render_wh=[1600,900],K_raw=K,T_world_from_camera_opencv=T)
        v=dict(status='passed',sample_id='s',actor_uuid='i',frozen_raw_timestamp_us=123,source_image='/raw.jpg',render_wh=[1600,900],gaussians=32,common_camera_audit=a)
        return v,c,ref

    def test_valid_and_empty_quality_not_filtered(self):
        v,c,r=self.fixture();validate_native_source(v,c,'h',r,'splatad')
        v.update(status='rendered_degenerate_output',technical_finite_render=True,quality_gate_passed=False)
        self.assertTrue(validate_native_source(v,c,'h',r,'splatad')['degenerate_render'])
        v['technical_finite_render']=False
        with self.assertRaises(ValueError):validate_native_source(v,c,'h',r,'splatad')

    def test_missing_or_active_time_corrections_rejected(self):
        v,c,r=self.fixture()
        for k in ['camera_pose_refinement','render_time_shift_seconds','rolling_shutter_seconds']:
            bad=copy.deepcopy(v);bad['common_camera_audit'].pop(k)
            with self.assertRaises(ValueError):validate_native_source(bad,c,'h',r,'splatad')
            bad=copy.deepcopy(v);bad['common_camera_audit'][k]=True if k=='camera_pose_refinement'else.01
            with self.assertRaises(ValueError):validate_native_source(bad,c,'h',r,'splatad')

    def test_identity_time_camera_and_invalid_numbers(self):
        v,c,r=self.fixture()
        for key,value in [('actor_uuid','other'),('frozen_raw_timestamp_us',124),('source_image','/other.jpg')]:
            bad=copy.deepcopy(v);bad[key]=value
            with self.assertRaises(ValueError):validate_native_source(bad,c,'h',r,'splatad')
        for key,value in [('projection_max_error_px',float('nan')),('pose_matrix_max_error',.002),('contract_sha256','other')]:
            bad=copy.deepcopy(v);bad['common_camera_audit'][key]=value
            with self.assertRaises(ValueError):validate_native_source(bad,c,'h',r,'splatad')
        bad=copy.deepcopy(v);bad['common_camera_audit']['K_raw'][0][0]=2
        # The fixture shares K deliberately; copy the unchanged camera for this test.
        _,camera,_=self.fixture()
        with self.assertRaises(ValueError):validate_native_source(bad,camera,'h',r,'splatad')

if __name__=='__main__':unittest.main()
