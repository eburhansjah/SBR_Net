# Author: Anastasios Nikolas Angelopoulos, angelopoulos@berkeley.edu
# https://github.com/aangelopoulos/im2im-uq/blob/main/core/models/losses/pinball.py
# pinball loss class
import torch
import torch.nn as nn
import logging
from torch.nn.functional import relu

logger = logging.getLogger(__name__)


# pinball loss class
class PinballLoss:
    def __init__(self, quantile=0.10, reduction="mean"):
        self.quantile = quantile
        assert 0 < self.quantile
        assert self.quantile < 1
        self.reduction = reduction
        logger.info(f"Initialized PinballLoss with quantile {self.quantile}")

    def __call__(self, output, target):
        assert output.shape == target.shape
        loss = torch.zeros_like(target, dtype=torch.float)
        error = output - target
        smaller_index = error < 0
        bigger_index = 0 < error
        loss[smaller_index] = self.quantile * (abs(error)[smaller_index])
        loss[bigger_index] = (1 - self.quantile) * (abs(error)[bigger_index])

        if self.reduction == "sum":
            loss = loss.sum()
        if self.reduction == "mean":
            loss = loss.mean()

        return loss
    
# usage:
# criterion = PinballLoss(quantile = 0.1)
# loss = criterion(out, gt) Order matters. If (gt, out) = 90th quantile