import torch
import torch.nn as nn
import torch.nn.functional as F

class FocalLoss(nn.Module):
    def __init__(self, alpha=1, gamma=2, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, pred, target):
        '''
        Computes the Focal Loss for binary classification.
        
        Args:
        - pred (torch.Tensor): Predicted probabilities of shape (batch_size, 1, height, width).
        - target (torch.Tensor): Ground truth binary labels of shape (batch_size, 1, height, width).
        
        Returns:
        - torch.Tensor: Computed Focal Loss.
        '''
        # Apply sigmoid to predictions
        pred = torch.sigmoid(pred)
        
        # make sure the tensors are contiguous
        pred = pred.contiguous()
        target = target.contiguous()

        # Flatten the prediction and target tensors
        pred = pred.view(-1)
        target = target.view(-1)
        
        # Compute the binary cross entropy loss
        bce_loss = F.binary_cross_entropy(pred, target, reduction='none')
        
        # Compute the modulating factor
        modulating_factor = (1 - pred).pow(self.gamma)
        
        # Compute the focal loss
        focal_loss = self.alpha * modulating_factor * bce_loss
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss
