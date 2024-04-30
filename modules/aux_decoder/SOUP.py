import torch
import torch.nn as nn


# 参考 https://github.com/DiffBeautifier/svbcode/blob/main/net.py
class PitchNet(nn.Module):
    def __init__(
            self, in_dims, out_dims, /, *,
            num_channels=512, num_layers=2, kernel_size=5, dropout_rate=0.1, strides=None
    ):
        super().__init__()
        in_dim = in_dims
        out_dim = out_dims
        n_layers = num_layers
        kernel = kernel_size
        
        padding = kernel // 2
        self.layers = []
        self.strides = strides if strides is not None else [1] * n_layers
        for l in range(n_layers):
            self.layers.append(
                nn.Sequential(
                    nn.Conv1d(
                        in_dim,
                        num_channels,
                        kernel_size=kernel,
                        padding=padding,
                        stride=self.strides[l],
                    ),
                    nn.ReLU(),
                    nn.BatchNorm1d(num_channels),
                )
            )
            in_dim = num_channels
        self.layers = nn.ModuleList(self.layers)

        self.mlp = nn.Sequential(
            nn.Linear(num_channels, num_channels // 2),
            nn.Mish(),
            nn.Linear(num_channels // 2, out_dim)
        )
        


    def forward(self, x, infer=False):
        """
        input:[B, T, 256]
        output:[B, T, F x C = 128]
        """
        x = x.transpose(1, 2)
        for _, l in enumerate(self.layers):
            x = l(x)
        x = x.transpose(1, 2)
        x = self.mlp(x)    
        
        return x
