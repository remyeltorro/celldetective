import unittest
import matplotlib.pyplot as plt
import numpy as np
import os

# class TestPatchMask(unittest.TestCase):

# 	@classmethod
# 	def setUpClass(self):
# 		self.radius = 3

# 	def test_correct_shape(self):
# 		self.patch = create_patch_mask(self.radius, self.radius)
# 		self.assertEqual(self.patch.shape,(3,3))

# 	def test_correct_ring(self):
# 		self.patch = create_patch_mask(5, 5,radius=[1,2])
# 		self.assertFalse(self.patch[2,2])

# class TestRemoveRedundantFeatures(unittest.TestCase):

# 	@classmethod
# 	def setUpClass(self):
# 		self.list_a = ['feat1','feat2','feat3','feat4','intensity_mean']
# 		self.list_b = ['feat5','feat2','feat1','feat6','test_channel_mean']
# 		self.expected = ['feat3','feat4']

# 	def test_remove_red_features(self):
# 		self.assertEqual(remove_redundant_features(self.list_a, self.list_b, channel_names=['test_channel']), self.expected)


# class TestExtractChannelIndices(unittest.TestCase):

# 	@classmethod
# 	def setUpClass(self):
# 		self.channels = ['ch1','ch2','ch3','ch4']
# 		self.required_channels = ['ch4','ch2']
# 		self.expected_indices = [3,1]

# 	def test_extracted_channels_are_correct(self):
# 		self.assertEqual(list(_extract_channel_indices(self.channels, self.required_channels)), self.expected_indices)


# class TestImgIndexPerChannel(unittest.TestCase):

# 	@classmethod
# 	def setUpClass(self):
# 		self.channels_indices = [1]
# 		self.len_movie = 5
# 		self.nbr_channels = 3
# 		self.expected_indices = [1,4,7,10,13]

# 	def test_index_sequence_is_correct(self):
# 		self.assertEqual(list(_get_img_num_per_channel(self.channels_indices, self.len_movie, self.nbr_channels)[0]), self.expected_indices)


# class TestSplitArrayByRatio(unittest.TestCase):

# 	@classmethod
# 	def setUpClass(self):
# 		self.array_length = 100
# 		self.array = np.ones(self.array_length)

# 	def test_ratio_split_is_correct(self):
# 		split_array = split_by_ratio(self.array,0.5,0.25,0.1)
# 		self.assertTrue(np.all([len(split_array[0])==50, len(split_array[1])==25, len(split_array[2])==10]))


# if __name__=="__main__":
# 	unittest.main()