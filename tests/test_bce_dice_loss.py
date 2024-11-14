import unittest
import torch
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../losses/')))
from bce_dice_loss import BCEDiceLoss

class TestBCEDiceLoss(unittest.TestCase):
    def test_init_default_weights(self):
        loss = BCEDiceLoss()
        self.assertEqual(loss.bce_weight, 0.5)
        self.assertEqual(loss.dice_weight, 0.5)
        self.assertIsInstance(loss.bce, torch.nn.BCEWithLogitsLoss)

    def test_init_custom_weights(self):
        loss = BCEDiceLoss(bce_weight=0.7, dice_weight=0.3)
        self.assertEqual(loss.bce_weight, 0.7)
        self.assertEqual(loss.dice_weight, 0.3)
        self.assertIsInstance(loss.bce, torch.nn.BCEWithLogitsLoss)

    def test_bce_only(self):
        loss = BCEDiceLoss(bce_weight=1.0, dice_weight=0.0)
        input_tensor = torch.randn(3, 1, 256, 256)
        target_tensor = torch.randint(0, 2, (3, 1, 256, 256)).float()
        bce_loss = torch.nn.BCEWithLogitsLoss()(input_tensor, target_tensor)
        combined_loss = loss(input_tensor, target_tensor)
        self.assertAlmostEqual(combined_loss.item(), bce_loss.item(), places=5)

    def test_dice_only(self):
        loss = BCEDiceLoss(bce_weight=0.0, dice_weight=1.0)
        input_tensor = torch.randn(3, 1, 256, 256)
        target_tensor = torch.randint(0, 2, (3, 1, 256, 256)).float()
        dice_loss = loss.dice_loss(input_tensor, target_tensor)
        combined_loss = loss(input_tensor, target_tensor)
        self.assertAlmostEqual(combined_loss.item(), dice_loss.item(), places=5)
        
if __name__ == '__main__':
    unittest.main()