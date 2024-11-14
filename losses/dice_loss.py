
import torch
import torch.nn as nn

class DiceLoss(nn.Module):
    def __init__(self, smooth=1e-6):
        '''
        Initializes the Dice Loss, a measure of similarity between two sets.
        
        Args:
        - smooth (float): Smoothing factor to avoid division by zero.
        
        Formula:
        - Dice Loss = 1 - Dice Score
        
        References:
        - https://oecd.ai/en/catalogue/metrics/dice-score
        - https://en.wikipedia.org/wiki/Dice-Sørensen_coefficient
        '''
        super(DiceLoss, self).__init__()
        self.smooth = smooth
    
    def forward(self, preds, targets):
        '''
        Computes the Dice Loss, a measure of similarity between two sets.
        
        Args:
        - preds (torch.Tensor): Predicted segmentation mask (binary or probabilistic tensor).
        - targets (torch.Tensor): Ground truth segmentation mask (binary tensor).
        
        Formula:
        - Dice Loss = 1 - Dice Score
        
        Returns:
        - torch.Tensor: Dice Loss value.
        '''
        # Apply sigmoid to predictions
        preds = torch.sigmoid(preds)

        # make sure the tensors are contiguous
        # This is necessary for the view operation to work
        # https://pytorch.org/docs/stable/tensors.html#torch.Tensor.view
        preds = preds.contiguous()
        targets = targets.contiguous()
  
        # Flatten preds and targets
        preds = preds.view(-1)
        targets = targets.view(-1)

        # Compute the intersection between predictions and targets
        intersection = (preds * targets).sum()
        
        # Compute the union (sum of all values in both predictions and targets)
        union = preds.sum() + targets.sum()
        
        # Calculate the Dice Score using the formula with a smoothing factor to prevent division by zero
        dice = (2. * intersection + self.smooth) / (union + self.smooth)
        
        # Calculate the Dice Loss as 1 - Dice Score
        dice_loss = 1. - dice
        
        # Return the Dice Loss as a tensor
        return dice_loss
