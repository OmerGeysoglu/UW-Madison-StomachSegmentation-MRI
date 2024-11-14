import unittest
import torch
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../losses/')))
from focal_loss import FocalLoss

class TestFocalLoss(unittest.TestCase):
    def test_default_parameters(self):
        loss = FocalLoss()
        self.assertEqual(loss.alpha, 1)
        self.assertEqual(loss.gamma, 2)
        self.assertEqual(loss.reduction, 'mean')

    def test_custom_parameters(self):
        loss = FocalLoss(alpha=0.5, gamma=3, reduction='sum')
        self.assertEqual(loss.alpha, 0.5)
        self.assertEqual(loss.gamma, 3)
        self.assertEqual(loss.reduction, 'sum')

    def test_forward_mean_reduction(self):
        loss = FocalLoss()
        pred = torch.tensor([0.9, 0.1, 0.8, 0.4], requires_grad=True)
        target = torch.tensor([1, 0, 1, 0], dtype=torch.float32)
        output = loss(pred, target)
        self.assertTrue(torch.is_tensor(output))
        self.assertEqual(output.shape, torch.Size([]))

    def test_forward_sum_reduction(self):
        loss = FocalLoss(reduction='sum')
        pred = torch.tensor([0.9, 0.1, 0.8, 0.4], requires_grad=True)
        target = torch.tensor([1, 0, 1, 0], dtype=torch.float32)
        output = loss(pred, target)
        self.assertTrue(torch.is_tensor(output))
        self.assertEqual(output.shape, torch.Size([]))

    def test_forward_no_reduction(self):
        loss = FocalLoss(reduction='none')
        pred = torch.tensor([0.9, 0.1, 0.8, 0.4], requires_grad=True)
        target = torch.tensor([1, 0, 1, 0], dtype=torch.float32)
        output = loss(pred, target)
        self.assertTrue(torch.is_tensor(output))
        self.assertEqual(output.shape, torch.Size([4]))

if __name__ == '__main__':
    unittest.main()