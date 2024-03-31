import torch.nn as nn
from torch import Tensor
import torch
import torch.nn.functional as F


class DiffusionNoiseLoss(nn.Module):
    def __init__(self, loss_type):
        super().__init__()
        self.loss_type = loss_type
        if self.loss_type == 'l1':
            self.loss = nn.L1Loss(reduction='none')
        elif self.loss_type == 'l2':
            self.loss = nn.MSELoss(reduction='none')
        else:
            raise NotImplementedError()

    @staticmethod
    def _mask_nonpadding(x_recon, noise, nonpadding=None):
        if nonpadding is not None:
            nonpadding = nonpadding.transpose(1, 2).unsqueeze(1)
            return x_recon * nonpadding, noise * nonpadding
        else:
            return x_recon, noise

    def _forward(self, x_r, v_pred, t):
        weights = 0.398942 / t / (1 - t) * torch.exp(-0.5 * torch.log(t / (1 - t)) ** 2)
        loss = torch.mean(weights[:, None, None, None] * F.mse_loss(x_recon, v_pred, reduction='none'))
        return loss

    def forward(self, x_recon: Tensor, noise: Tensor, t, nonpadding: Tensor = None) -> Tensor:
        """
        :param x_recon: [B, 1, M, T]
        :param noise: [B, 1, M, T]
        :param nonpadding: [B, T, M]
        """
        x_recon, noise = self._mask_nonpadding(x_recon, noise, nonpadding)
        return self._forward(x_recon, noise, t).mean()
