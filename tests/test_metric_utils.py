import unittest
import torch
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../utils/')))
from metric_utils import DiceLoss

class TestDiceLoss(unittest.TestCase):
    def setUp(self):
        self.dice_loss = DiceLoss()

    def test_dice_loss_perfect_match(self):
        preds = torch.tensor([1, 1, 1, 1], dtype=torch.float32)
        targets = torch.tensor([1, 1, 1, 1], dtype=torch.float32)
        loss = self.dice_loss(preds, targets)
        self.assertAlmostEqual(loss.item(), 0.0, places=6)

    def test_dice_loss_no_match(self):
        preds = torch.tensor([1, 1, 1, 1], dtype=torch.float32)
        targets = torch.tensor([0, 0, 0, 0], dtype=torch.float32)
        loss = self.dice_loss(preds, targets)
        self.assertAlmostEqual(loss.item(), 1.0, places=6)

    def test_dice_loss_partial_match(self):
        preds = torch.tensor([1, 0, 1, 0], dtype=torch.float32)
        targets = torch.tensor([1, 1, 0, 0], dtype=torch.float32)
        loss = self.dice_loss(preds, targets)
        expected_loss = 1 - (2 * 1.0 / (4.0 + 1e-6))
        self.assertAlmostEqual(loss.item(), expected_loss, places=6)

    def test_dice_loss_smooth_factor(self):
        preds = torch.tensor([0, 0, 0, 0], dtype=torch.float32)
        targets = torch.tensor([0, 0, 0, 0], dtype=torch.float32)
        loss = self.dice_loss(preds, targets)
        expected_loss = 1 - (2 * 0.0 + 1e-6) / (0.0 + 0.0 + 1e-6)
        self.assertAlmostEqual(loss.item(), expected_loss, places=6)

if __name__ == '__main__':
    unittest.main()