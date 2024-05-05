import torch
import torch.nn as nn


class ResidualConvBlock(nn.Module):
    # 参考 https://github.com/CNChTu/Diffusion-SVC
    def __init__(self, dim, expansion_factor=2, kernel_size=31, dropout=0., use_norm=False):

        super().__init__()

        inner_dim = dim * expansion_factor
        padding = (kernel_size - 1) // 2
        self.norm = nn.LayerNorm(dim) if use_norm else nn.Identity()
        self.conv1x1_expansion = nn.Conv1d(dim, inner_dim * 2, 1)
        self.glu = nn.GLU(dim=1)
        self.dwconv = nn.Conv1d(inner_dim, inner_dim, kernel_size, groups=inner_dim, padding=padding)
        self.prelu = nn.PReLU(inner_dim)
        self.conv1x1_reduction = nn.Conv1d(inner_dim, dim, 1)
        self.dropout = nn.Dropout(dropout) if float(dropout) > 0. else nn.Identity()
        self.shortcut = nn.Identity() if dim == inner_dim else nn.Conv1d(dim, dim, 1)


    def forward(self, x):

        residual = x

        x = self.norm(x)
        x = self.conv1x1_expansion(x)
        x = self.glu(x)
        x = self.dwconv(x)
        x = self.prelu(x)
        x = self.conv1x1_reduction(x)
        x = self.dropout(x)
        
        x = x + self.shortcut(residual)

        return x


class PitchNet(nn.Module):
    # 参考 https://github.com/DiffBeautifier/svbcode/blob/main/net.py
    def __init__(
            self, in_dims, out_dims, /, *, num_channels=256, num_layers=3, expansion_factor=1, kernel_size=31, dropout_rate=0.1, use_norm=False
    ):
        super().__init__()
        self.mlp_input = nn.Sequential(
            nn.Linear(in_dims, in_dims * 2),
            nn.Mish(),
            nn.Linear(in_dims * 2, num_channels)
        )

        self.residual_blocks = nn.ModuleList([
            ResidualConvBlock(num_channels, expansion_factor, kernel_size, dropout_rate, use_norm)
            for _ in range(num_layers)
        ])
        
        self.mlp_output = nn.Sequential(
            nn.Linear(num_channels, num_channels // 2),
            nn.Mish(),
            nn.Linear(num_channels // 2, out_dims)
        )


    def forward(self, x, infer=False):
        """
        input:[B, T, 256]
        output:[B, T, F x C = 128]
        """
        x = self.mlp_input(x)
        x = x.transpose(1, 2)
        for block in self.residual_blocks:
            x = block(x)
        x = x.transpose(1, 2)
        x = self.mlp_output(x)
        
        return x



# 参考 https://github.com/DiffBeautifier/svbcode/blob/main/net.py
class PitchNetOld(nn.Module):
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
