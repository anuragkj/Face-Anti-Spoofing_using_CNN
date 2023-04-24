import torch
from torch import nn
import torch.nn.functional as F

#Ground truth mask is the actual mask, like what is expected from our mask, the difference bw the two give pixel wise loss
# pixel loss is used to optimize the model to produce more accurate masks, FOR MASKS 
# while binary loss is used to optimize the model to produce more accurate binary labels. FOR LABELS

class PixWiseBCELoss(nn.Module):
    def __init__(self, beta=0.5):
        super().__init__()
        self.criterion = nn.BCELoss()
        self.beta = beta

    def forward(self, net_mask, net_label, target_mask, target_label):
        pixel_loss = self.criterion(net_mask, target_mask)
        binary_loss = self.criterion(net_label, target_label)
        loss = pixel_loss * self.beta + binary_loss * (1 - self.beta)
        return loss
