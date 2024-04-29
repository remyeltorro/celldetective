import unittest
import matplotlib.pyplot as plt
from celldetective.utils import create_patch_mask, remove_redundant_features

class TestPatchMask(unittest.TestCase):

	@classmethod
	def setUpClass(self):
		self.radius = 3

	def test_correct_shape(self):
		self.patch = create_patch_mask(self.radius, self.radius)
		self.assertEqual(self.patch.shape,(3,3))

	def test_correct_ring(self):
		self.patch = create_patch_mask(5, 5,radius=[1,2])
		self.assertFalse(self.patch[2,2])

class TestRemoveRedundantFeatures(unittest.TestCase):

	@classmethod
	def setUpClass(self):
		self.list_a = ['feat1','feat2','feat3','feat4','intensity_mean']
		self.list_b = ['feat5','feat2','feat1','feat6','test_channel_mean']
		self.expected = ['feat3','feat4']

	def test_remove_red_features(self):
		self.assertEqual(remove_redundant_features(self.list_a, self.list_b, channel_names=['test_channel']), self.expected)


if __name__=="__main__":
	unittest.main()