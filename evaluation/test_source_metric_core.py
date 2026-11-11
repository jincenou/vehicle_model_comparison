"""Numerical and fairness regressions; CPU only, no datasets or model downloads."""
import copy
import json
import math
import unittest
import numpy as np
from source_metric_core import iou, psnr, ssim, reference_roi, roi_canvas, paired_summary, validate_camera


class Metrics(unittest.TestCase):
    def test_identity_and_json(self):
        x = np.arange(16*16*3, dtype=np.uint8).reshape(16,16,3)
        p = psnr(x,x)
        self.assertEqual(p['status'],'positive_infinity')
        self.assertTrue(p['perfect_match'])
        json.dumps(p,allow_nan=False)
        self.assertAlmostEqual(ssim(x,x)['value'],1,places=12)

    def test_known_psnr(self):
        x=np.zeros((16,16,3)); y=np.full_like(x,.1)
        self.assertAlmostEqual(psnr(x,y)['value'],20,places=12)
        y=np.zeros_like(x); y[:,:,0]=.3
        self.assertAlmostEqual(psnr(x,y)['mse'],.03,places=12)

    def test_reject_wrong_ranges_shapes_nonfinite(self):
        x=np.zeros((16,16,3))
        for y in (np.full_like(x,255),np.full_like(x,np.nan),np.zeros((1,16,3)),x.astype(np.int16)):
            with self.assertRaises(ValueError): psnr(x,y)

    def test_mask_empty_and_threshold(self):
        a=np.array([[0,127,128,255]],dtype=np.uint8)
        self.assertEqual(iou(a,a)['value'],1)
        self.assertEqual(iou(a,np.zeros_like(a))['value'],0)
        self.assertEqual(iou(np.zeros_like(a),a)['reason'],'empty_reference_mask')
        self.assertEqual(iou(a,a)['reference_pixels'],2)
        with self.assertRaises(ValueError): iou(a,a/255)

    def test_known_translated_iou(self):
        a=np.zeros((20,20),bool);b=a.copy()
        a[5:15,5:15]=True;b[5:15,10:20]=True
        self.assertAlmostEqual(iou(a,b)['value'],1/3)

    def test_fixed_roi_keeps_prediction_translation(self):
        a=np.zeros((100,100),bool);a[30:70,30:70]=True
        b=np.zeros_like(a);b[30:70,40:80]=True
        box=reference_roi(a)
        self.assertEqual(box,[28,28,72,72])
        self.assertLess(iou(roi_canvas(a,box,True),roi_canvas(b,box,True))['value'],.9)

    def test_outside_padding_white_not_red(self):
        x=np.zeros((10,10,3),np.uint8)
        y=roi_canvas(x,[-10,-10,10,10])
        np.testing.assert_array_equal(y[0,0],[255,255,255])
        np.testing.assert_array_equal(y[-1,-1],[0,0,0])

    def test_ssim_fixed_window(self):
        with self.assertRaises(ValueError): ssim(np.zeros((10,10,3)),np.zeros((10,10,3)))

    def rows(self):
        return [dict(method=m,sample_id=s,protocol_id='v1',scores={'PSNR':{'value':v,'status':'ok'}})
                for m,s,v in [('a','x',10),('a','y',30),('b','x',20)]]

    def test_pairwise_missing_not_independent_means(self):
        r=paired_summary(self.rows(),['a','b'],['x','y','z'],'PSNR','v1')
        self.assertEqual(r['requested'],3)
        self.assertEqual(r['paired_ids'],['x'])
        self.assertEqual(r['paired']['a']['mean']['value'],10)
        self.assertEqual(r['own_valid']['a']['mean']['value'],20)

    def test_perfect_not_filtered(self):
        rows=self.rows();rows[0]['scores']['PSNR']=psnr(np.zeros((11,11,3)),np.zeros((11,11,3)))
        r=paired_summary(rows,['a','b'],['x','y'],'PSNR','v1')
        self.assertEqual(r['paired_ids'],['x'])
        self.assertEqual(r['paired']['a']['mean']['status'],'positive_infinity')
        json.dumps(r,allow_nan=False)

    def test_duplicate_and_mixed_protocol_rejected(self):
        rows=self.rows()
        with self.assertRaises(ValueError): paired_summary(rows+[rows[0]],['a','b'],['x','y'],'PSNR','v1')
        rows[0]['protocol_id']='legacy'
        with self.assertRaises(ValueError): paired_summary(rows,['a','b'],['x','y'],'PSNR','v1')

    def test_empty_common_subset(self):
        r=paired_summary(self.rows()[:2],['a','b'],['x','y'],'PSNR','v1')
        self.assertEqual(r['paired_ids'],[])
        self.assertEqual(r['paired']['a']['mean']['status'],'unavailable')

    def test_camera_reject_mismatch(self):
        c=dict(sample_id='s',render_wh=[1600,900],timestamp_us=123)
        a=dict(contract_sha256='sha',projection_max_error_px=.001,render_wh=[1600,900],camera_pose_refinement=False,render_time_shift_seconds=0)
        v=dict(status='passed',sample_id='s',common_camera_audit=a,frozen_raw_timestamp_us=123)
        self.assertTrue(validate_camera(v,c,'sha'))
        for k,x in [('contract_sha256','bad'),('projection_max_error_px',math.nan),('projection_max_error_px',.05),('camera_pose_refinement',True),('render_time_shift_seconds',.1),('render_wh',[512,512])]:
            bad=copy.deepcopy(v);bad['common_camera_audit'][k]=x
            with self.assertRaises(ValueError): validate_camera(bad,c,'sha')
        bad=copy.deepcopy(v);bad['frozen_raw_timestamp_us']=124
        with self.assertRaises(ValueError): validate_camera(bad,c,'sha')


if __name__=='__main__': unittest.main()
