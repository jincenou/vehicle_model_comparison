import unittest
import numpy as np
from source_reference_geometry import place_reference

class SourceReferenceTests(unittest.TestCase):
    def setup_data(self):
        raw=np.arange(20*30*3,dtype=np.uint8).reshape(20,30,3)
        rgba=np.concatenate([raw[4:14,3:15],np.full((10,12,1),255,np.uint8)],axis=2)
        return raw,rgba,[3,4,15,14],[[1,0,-3],[0,1,-4],[0,0,1]]

    def test_native_location_and_roi(self):
        mask,rgb,m,meta=place_reference(*self.setup_data())
        self.assertEqual(int(mask.sum()),120);self.assertTrue(mask[4,3]);self.assertFalse(mask[3,3])
        self.assertEqual(meta['foreground_rgb_max_error'],0);self.assertFalse(meta['raw_frame_edge_contact'])
        self.assertTrue(meta['stored_crop_edge_contact']);self.assertEqual(rgb.shape,(512,512,3))
        self.assertFalse(meta['reference_qualified'])

    def test_resized_mask_not_claimed_native(self):
        raw,rgba,box,_=self.setup_data();small=rgba[::2,::2]
        mask,_,_,meta=place_reference(raw,small,box,[[.5,0,-1.75],[0,.5,-2.25],[0,0,1]])
        self.assertEqual(int(mask.sum()),120);self.assertFalse(meta['native_reference_pixels_verified'])
        self.assertIsNone(meta['foreground_rgb_max_error'])

    def test_incorrect_source_or_transform_rejected(self):
        raw,rgba,box,H=self.setup_data();bad=rgba.copy();bad[0,0,0]^=1
        with self.assertRaises(ValueError):place_reference(raw,bad,box,H)
        H[0][2]+=.5
        with self.assertRaises(ValueError):place_reference(raw,rgba,box,H)

    def test_frame_border_distinct_from_crop_border(self):
        raw,rgba,_,_=self.setup_data();rgba[:,:,:3]=raw[0:10,0:12]
        mask,_,_,meta=place_reference(raw,rgba,[0,0,12,10],np.eye(3))
        self.assertTrue(meta['raw_frame_edge_contact']);self.assertEqual(meta['raw_frame_edge_pixels']['top'],12)

if __name__=='__main__':unittest.main()
