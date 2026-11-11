import unittest
import numpy as np
from scipy.linalg import sqrtm
from feature_metric_core import frechet_features,orbit_i16,foreground_cosine_distance

class FeatureMetricTests(unittest.TestCase):
    def test_one_dimensional_analytic(self):
        a=np.array([[0.],[2.]])
        b=np.array([[1.],[5.],[9.]])
        expected=(1-5)**2+(np.sqrt(2)-4)**2
        self.assertAlmostEqual(frechet_features(a,b)['value'],expected,places=12)

    def test_agrees_with_covariance_sqrtm_full_rank(self):
        rng=np.random.default_rng(42);a=rng.normal(size=(31,5));b=rng.normal(size=(19,5))*1.4+.2
        ca=np.cov(a,rowvar=False);cb=np.cov(b,rowvar=False);delta=a.mean(0)-b.mean(0)
        expected=delta@delta+np.trace(ca)+np.trace(cb)-2*np.trace(sqrtm(ca@cb)).real
        self.assertAlmostEqual(frechet_features(a,b)['value'],expected,places=11)

    def test_rank_deficient_identity_and_permutation(self):
        rng=np.random.default_rng(8);a=rng.normal(size=(5,2048))
        self.assertLess(frechet_features(a,a[::-1])['value'],1e-9)

    def test_translation_and_symmetry(self):
        rng=np.random.default_rng(3);a=rng.normal(size=(7,23));shift=np.arange(23)/10
        self.assertAlmostEqual(frechet_features(a,a+shift)['value'],float(shift@shift),places=10)
        b=rng.normal(size=(9,23))
        self.assertAlmostEqual(frechet_features(a,b)['value'],frechet_features(b,a)['value'],places=11)

    def test_constant_features_and_no_silent_normalization(self):
        a=np.tile([1.,0.],(3,1));b=np.tile([2.,0.],(4,1))
        self.assertEqual(frechet_features(a,b)['value'],1.)
        self.assertEqual(frechet_features(a,a)['value'],0.)

    def test_invalid_feature_batches(self):
        valid=np.zeros((2,3),dtype=float)
        for bad in [np.zeros((1,3)),np.zeros((2,4)),np.full((2,3),np.nan),np.ones((2,3),int),np.ones((2,3),complex)]:
            with self.assertRaises(ValueError):frechet_features(valid,bad)

    def test_orbit_averages_all_sixteen(self):
        source=np.array([[1.,0.]]);renders=np.tile([-1.,0.],(16,1));renders[0]=source[0]
        result=orbit_i16(source,renders)
        self.assertEqual(result['value'],-.875)
        with self.assertRaises(ValueError):orbit_i16(source,renders[:15])
        with self.assertRaises(ValueError):orbit_i16(source,np.zeros((16,2)))

    def test_edr_bounds_and_numerically_scaled_embeddings(self):
        self.assertEqual(foreground_cosine_distance(np.array([[1e300,0.]]),np.array([[1e-300,0.]])),0.)
        self.assertEqual(foreground_cosine_distance(np.array([[1.,0.]]),np.array([[-1.,0.]])),2.)

if __name__=='__main__':unittest.main()
