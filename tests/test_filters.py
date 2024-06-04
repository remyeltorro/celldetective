import unittest
import numpy as np
from celldetective.filters import gauss_filter, abs_filter


class TestFilters(unittest.TestCase):

	@classmethod
	def setUpClass(self):
		self.img = np.ones((256,256),dtype=int)
		self.img[100:110,100:110] = 0
		self.gauss_sigma = 1.6

	def test_gauss_filter_is_float(self):
		self.assertIsInstance(gauss_filter(self.img, self.gauss_sigma)[0,0], float)
	
	def test_gauss_filter_has_same_shape(self):
		self.assertEqual(gauss_filter(self.img, self.gauss_sigma).shape, self.img.shape)

	def test_abs_filter_is_positive(self):
		self.assertTrue(np.all(abs_filter(self.img) >= 0.))

if __name__=="__main__":
	unittest.main()