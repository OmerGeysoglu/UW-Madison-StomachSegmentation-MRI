import unittest
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../utils/')))
from data_utils import MadisonStomach

class TestMadisonStomach(unittest.TestCase):

    def setUp(self):
        self.data_path = '../madison-stomach-data'
        self.train_mode = 'train'
        self.test_mode = 'test'

    def test_init_train_mode(self):
        dataset = MadisonStomach(data_path=self.data_path, mode=self.train_mode)
        self.assertEqual(len(dataset.image_paths), len(dataset.mask_paths))
        self.assertTrue(dataset.augment)
        self.assertIsNotNone(dataset.transform)
        self.assertIsNotNone(dataset.mask_transform)
        self.assertIsNotNone(dataset.augmentation_transforms)

    def test_init_test_mode(self):
        dataset = MadisonStomach(data_path=self.data_path, mode=self.test_mode)
        self.assertEqual(len(dataset.image_paths), len(dataset.mask_paths))
        self.assertFalse(dataset.augment)
        self.assertIsNotNone(dataset.transform)
        self.assertIsNotNone(dataset.mask_transform)
        self.assertIsNotNone(dataset.augmentation_transforms)

if __name__ == '__main__':
    unittest.main()