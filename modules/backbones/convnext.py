from typing import Optional

import torch
import torch.nn as nn

from modules.backbones.wavenet import SinusoidalPosEmb
from utils import hparams


class ConvNeXtBlock(nn.Module):
    """ConvNeXt Block adapted from https://github.com/facebookresearch/ConvNeXt to 1D audio signal.

    Args:
        dim (int): Number of input channels.
        intermediate_dim (int): Dimensionality of the intermediate layer.
        layer_scale_init_value (float, optional): Initial value for the layer scale. None means no scaling.
            Defaults to None.
    """

    def __init__(
            self,
            dim: int,
            intermediate_dim: int, cond_dim, time_embed_dim,
            layer_scale_init_value: Optional[float] = None, drop_out: float = 0.0

    ):
        super().__init__()
        self.dwconv = nn.Conv1d(dim, dim, kernel_size=7, padding=3, groups=dim)  # depthwise conv
        # self.dwconv = nn.Conv1d(dim, dim, kernel_size=3, padding=1,)  # depthwise conv
        self.condconv = nn.Conv1d(cond_dim, dim, kernel_size=1, padding=0)
        self.timeconv = nn.Conv1d(time_embed_dim, dim, kernel_size=1, padding=0)
        self.norm = nn.LayerNorm(dim, eps=1e-6)
        self.pwconv1 = nn.Linear(dim, intermediate_dim)  # pointwise/1x1 convs, implemented with linear layers
        self.act = nn.GELU()
        self.pwconv2 = nn.Linear(intermediate_dim, dim)
        self.gamma = (
            nn.Parameter(layer_scale_init_value * torch.ones(dim), requires_grad=True)
            if layer_scale_init_value > 0
            else None
        )
        # self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.drop_path = nn.Identity()
        self.dropout = nn.Dropout(drop_out) if drop_out > 0. else nn.Identity()

    def forward(self, x: torch.Tensor, t, cond) -> torch.Tensor:
        residual = x

        x = x + self.timeconv(t.unsqueeze(-1) )+ self.condconv(cond)
        x = self.dwconv(x)
        x = x.transpose(1, 2)  # (B, C, T) -> (B, T, C)

        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        if self.gamma is not None:
            x = self.gamma * x
        x = x.transpose(1, 2)  # (B, T, C) -> (B, C, T)
        x = self.dropout(x)

        x = residual + self.drop_path(x)
        return x


class ConvNeXtModule(nn.Module):
    def __init__(self, dim, lays, time_embed_dim, cond_dim, intermediate_dim=None, layer_scale_init_value=1e-6,
                 drop_out=0.):
        super().__init__()
        if intermediate_dim is None:
            intermediate_dim = dim * 2
        else:
            intermediate_dim = int(dim * intermediate_dim)

        self.layers = nn.ModuleList(
            [ConvNeXtBlock(dim=dim,
                           intermediate_dim=intermediate_dim, cond_dim=cond_dim, time_embed_dim=time_embed_dim,
                           layer_scale_init_value=layer_scale_init_value, drop_out=drop_out) for _ in range(lays)])

    def forward(self, x, t, cond):

        for layer in self.layers:
            x = layer(x, t, cond)

        return x


class ConvNeXtModel(nn.Module):
    def __init__(self, in_dim, time_embed_dim, cond_dim, dims=None, lays=None, intermediate_dim=None,
                 layer_scale_init_value=1e-6, drop_out=0.):
        super().__init__()
        if dims is None:
            dims = [64, 128, 256, 512]
        if lays is None:
            lays = [3, 6, 2, 1]
        dims = [in_dim] + dims
        self.num = len(lays)
        self.encoder = nn.ModuleList()
        self.dim_proj = nn.ModuleList()
        for i in range(len(lays)):
            self.dim_proj.append(nn.Conv1d(dims[i], dims[i + 1], 1))
            self.encoder.append(
                ConvNeXtModule(dim=dims[i + 1], lays=lays[i], time_embed_dim=time_embed_dim, cond_dim=cond_dim,
                               intermediate_dim=intermediate_dim, layer_scale_init_value=layer_scale_init_value,
                               drop_out=drop_out))
        self.output_projection = nn.Conv1d(dims[-1], in_dim, 1)

    def forward(self, x, t, cond, mask=None):
        for i in range(self.num):
            x = self.dim_proj[i](x)
            if mask is not None:
                x = x * mask
            x = self.encoder[i](x, t, cond)
            if mask is not None:
                x = x * mask
        x = self.output_projection(x)
        if mask is not None:
            x = x * mask

        return x


class ConvNeXt(nn.Module):
    def __init__(self, in_dims, n_feats, *, n_layers=[20], time_embed_dim=512, n_chans=[256], intermediate_dim=4):
        super().__init__()
        self.in_dims = in_dims
        self.n_feats = n_feats

        self.diffusion_embedding = SinusoidalPosEmb(time_embed_dim)
        self.mlp = nn.Sequential(
            nn.Linear(time_embed_dim, time_embed_dim * 4),
            nn.Mish(),
            nn.Linear(time_embed_dim * 4, time_embed_dim)
        )

        self.model = ConvNeXtModel(in_dim=in_dims * n_feats, dims=n_chans, lays=n_layers,
                                   intermediate_dim=intermediate_dim,
                                   layer_scale_init_value=1e-6, drop_out=0., time_embed_dim=time_embed_dim,
                                   cond_dim=hparams['hidden_size'])

    def forward(self, spec, diffusion_step, cond):
        """
        :param spec: [B, F, M, T]
        :param diffusion_step: [B, 1]
        :param cond: [B, H, T]
        :return:
        """
        if self.n_feats == 1:
            x = spec.squeeze(1)  # [B, M, T]
        else:
            x = spec.flatten(start_dim=1, end_dim=2)  # [B, F x M, T]
        diffusion_step = self.diffusion_embedding(diffusion_step)
        diffusion_step = self.mlp(diffusion_step)
        # x=x.transpose(1,2)
        x = self.model(x, diffusion_step, cond)
        # x = x.transpose(1, 2)
        if self.n_feats == 1:
            x = x[:, None, :, :]
        else:
            # This is the temporary solution since PyTorch 1.13
            # does not support exporting aten::unflatten to ONNX
            # x = x.unflatten(dim=1, sizes=(self.n_feats, self.in_dims))
            x = x.reshape(-1, self.n_feats, self.in_dims, x.shape[2])
        return x


if __name__ == '__main__':
    m = VAE(1, 32)
    x1 = torch.randn(1, 1, 128)
    x2 = m(x1)
    print(x2[0].shape)
