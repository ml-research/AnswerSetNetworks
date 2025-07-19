import torch
import torch.nn as nn


class PromisMockNet(nn.Module):
    """TODO"""

    def __init__(self, type="house"):
        """TODO"""
        super(PromisMockNet, self).__init__()
        self.param = nn.Parameter(torch.tensor([0.0]))

    def forward(self, x):
        return x
