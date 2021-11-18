import torch
from torch import nn

class RMSELoss(nn.Module):
    def __init__(self, reduction='mean'):
        super(RMSELoss, self).__init__()
        self.reduction = reduction 

    def forward(self,input,target):
        return torch.sqrt(F.mse_loss(input, target, reduction=self.reduction))

METRICS = {
    'rmse': RMSELoss
}