import unittest
import numpy as np
from alpha_extent_measurement import rig,measure_view,combine,convergence


def box_measure(center,extent,span,resolution):
    center=np.asarray(center);extent=np.asarray(extent);result=[]
    for view in rig(center,span,resolution):
        rotation=np.asarray(view['T_camera_from_semantic'])[:3,:3]
        u=np.argmax(np.abs(rotation[0]));v=np.argmax(np.abs(rotation[1]))
        coords=(np.arange(resolution)+.5-resolution/2)*span/resolution
        mask=(np.abs(coords)[None,:]<extent[u]/2)&(np.abs(coords)[:,None]<extent[v]/2)
        result.append(measure_view(mask.astype('uint8')*255,view))
    return combine(result)


class AlphaExtentTests(unittest.TestCase):
    def test_translated_box_recovers_centres_and_extents(self):
        result=box_measure([7,-4,2],[2,1,.5],4,256)
        np.testing.assert_allclose(result['center'],[7,-4,2],atol=1e-12)
        np.testing.assert_allclose(result['extent'],[2,1,.5],atol=1e-12)

    def test_native_unit_change_scales_measurement_uniformly(self):
        a=box_measure([0,0,0],[2,1,.5],4,256)
        b=box_measure([0,0,0],[20,10,5],40,256)
        np.testing.assert_allclose(np.array(a['extent'])*10,b['extent'])

    def test_empty_weak_alpha_clipped_and_missing_views_rejected(self):
        view=rig([0,0,0],4,32)[0]
        for image in [np.zeros((32,32),np.uint8),np.full((32,32),127,np.uint8),np.full((32,32),255,np.uint8)]:
            with self.assertRaises(ValueError):measure_view(image,view)
        with self.assertRaises(ValueError):combine([])

    def test_resolution_convergence_is_not_a_formal_qualification(self):
        a=box_measure([0,0,0],[1.9,.9,.4],4,128)
        b=box_measure([0,0,0],[1.9,.9,.4],4,256)
        result=convergence(a,b,4,128);self.assertTrue(result['passed']);self.assertFalse(result['formal_metrics_qualified'])
        b['upper'][0]+=1;self.assertFalse(convergence(a,b,4,128)['passed'])


if __name__=='__main__':unittest.main()
