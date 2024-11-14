import unittest
import torch
import torch.nn as nn
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../losses/')))
from dice_loss import DiceLoss

class TestDiceLoss(unittest.TestCase):
    def setUp(self):
        self.dice_loss_with_logits = DiceLoss()

    def test_perfect_prediction_large(self):
        # Large tensor where prediction is perfect
        logits = torch.rand(5, 1, 10, 10) * 20  # Large positive values to push sigmoid to 1
        true_labels = torch.ones(5, 1, 10, 10)
        loss = self.dice_loss_with_logits(logits, true_labels)
        self.assertLess(loss, 0.1, msg="Loss should be close to 0 when prediction is perfect")
        
    def test_no_intersection_large(self):
        # Large tensor with no intersection
        logits = torch.rand(5, 1, 10, 10) * 10  # Large positive values to push sigmoid to 1
        true_labels = torch.zeros(5, 1, 10, 10)
        loss = self.dice_loss_with_logits(logits, true_labels)
        self.assertAlmostEqual(loss, 1.0, places=4, msg="Loss should be 1 when there is no intersection")

    def test_mixed_values(self):
        # Mixed values for intersection testing
        logits = torch.tensor([[[[5.0, -5.0], [0.0, 0.0]]]])
        true_labels = torch.tensor([[[[1.0, 0.0], [1.0, 0.0]]]])
        loss = self.dice_loss_with_logits(logits, true_labels)
        # Expected that the loss is neither 0 nor 1 since there's partial overlap
        self.assertTrue(0.0 < loss < 1.0, msg="Loss should be between 0 and 1 for partial overlap")


if __name__ == '__main__':
    unittest.main()
