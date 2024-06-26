import torch
from torch import nn
import torch.nn.functional as F

# From https://github.com/CNChTu/Diffusion-SVC/ by CNChTu
# License: MIT


class SwiGLU(nn.Module):
    ## Swish-Applies the gated linear unit function.
    def __init__(self, dim=-1):
        super().__init__()
        self.dim = dim
    def forward(self, x):
        out, gate = x.chunk(2, dim=self.dim)
        return out * F.silu(gate)


class ConformerNaiveEncoder(nn.Module):
    """
    Conformer Naive Encoder

    Args:
        dim_model (int): Dimension of model
        num_layers (int): Number of layers
        num_heads (int): Number of heads
        use_norm (bool): Whether to use norm for FastAttention, only True can use bf16/fp16, default False
        conv_only (bool): Whether to use only conv module without attention, default False
        conv_dropout (float): Dropout rate of conv module, default 0.
        atten_dropout (float): Dropout rate of attention module, default 0.
    """

    def __init__(self,
                 num_layers: int,
                 num_heads: int,
                 dim_model: int,
                 use_norm: bool = False,
                 conv_only: bool = False,
                 conv_dropout: float = 0.,
                 atten_dropout: float = 0.,
                 conv_model_type='mode1',
                 conv_model_activation='SiLU'
                 ):
        super().__init__()
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.dim_model = dim_model
        self.use_norm = use_norm
        self.residual_dropout = 0.1  # 废弃代码,仅做兼容性保留
        self.attention_dropout = 0.1  # 废弃代码,仅做兼容性保留

        self.encoder_layers = nn.ModuleList(
            [
                CFNEncoderLayer(
                    dim_model=dim_model,
                    expansion_factor=expansion_factor,
                    kernel_size=kernel_size,
                    num_heads=num_heads,
                    use_norm=use_norm,
                    conv_only=conv_only,
                    conv_dropout=conv_dropout,
                    atten_dropout=atten_dropout,
                    conv_model_type=conv_model_type,
                    conv_model_activation=conv_model_activation
                )
                for _ in range(num_layers)
            ]
        )

    def forward(self, x, mask=None) -> torch.Tensor:
        """
        Args:
            x (torch.Tensor): Input tensor (#batch, length, dim_model)
            mask (torch.Tensor): Mask tensor, default None
        return:
            torch.Tensor: Output tensor (#batch, length, dim_model)
        """

        for (i, layer) in enumerate(self.encoder_layers):
            x = layer(x, mask)
        return x  # (#batch, length, dim_model)


class CFNEncoderLayer(nn.Module):
    """
    Conformer Naive Encoder Layer

    Args:
        dim_model (int): Dimension of model
        expansion_factor (int): Expansion factor of conv module, default 2
        kernel_size (int): Kernel size of conv module, default 31
        num_heads (int): Number of heads
        use_norm (bool): Whether to use norm
        conv_only (bool): Whether to use only conv module without attention, default False
        conv_dropout (float): Dropout rate of conv module, default 0.1
        atten_dropout (float): Dropout rate of attention module, default 0.1
    """

    def __init__(self,
                 dim_model: int,
                 expansion_factor: int = 2,
                 kernel_size: int = 31,
                 num_heads: int = 8,
                 use_norm: bool = False,
                 conv_only: bool = True,
                 conv_dropout: float = 0.,
                 atten_dropout: float = 0.1,
                 conv_model_type='mode1',
                 conv_model_activation='SiLU'
                 ):
        super().__init__()

        self.conformer = ConformerConvModule(
            dim_model,
            expansion_factor=expansion_factor,
            kernel_size=kernel_size,
            use_norm=use_norm,
            dropout=conv_dropout,
            conv_model_type=conv_model_type,
            activation=conv_model_activation
        )

        self.norm = nn.LayerNorm(dim_model)

        self.dropout = nn.Dropout(0.1)  # 废弃代码,仅做兼容性保留

        # selfatt -> fastatt: performer!
        if not conv_only:
            self.attn = nn.TransformerEncoderLayer(
                d_model=dim_model,
                nhead=num_heads,
                dim_feedforward=dim_model * 4,
                dropout=atten_dropout,
                activation='gelu'
            )
        else:
            self.attn = None

    def forward(self, x, mask=None) -> torch.Tensor:
        """
        Args:
            x (torch.Tensor): Input tensor (#batch, length, dim_model)
            mask (torch.Tensor): Mask tensor, default None
        return:
            torch.Tensor: Output tensor (#batch, length, dim_model)
        """
        if self.attn is not None:
            x = x + (self.attn(self.norm(x), src_mask=mask))

        x = x + (self.conformer(x))

        return x  # (#batch, length, dim_model)


class ConformerConvModule(nn.Module):
    def __init__(
            self,
            dim,
            expansion_factor=2,
            kernel_size=31,
            dropout=0.,
            use_norm=False,
            activation='SiLU',
            # 炼丹魅力时刻之激活函数带音染，Swish会让声音变尖一些，DoubleSwish更尖，ReLU稍弱，不同数据表现不一样，建议自行测试
            conv_model_type='mode1',
            GLU_type='GLU',
    ):
        super().__init__()
        inner_dim = dim * expansion_factor
        padding = calc_same_padding(kernel_size)
        activation = activation if activation is not None else 'SiLU'
        if activation == 'SiLU':
            _activation = nn.SiLU()
        elif activation == 'ReLU':
            _activation = nn.ReLU()
        elif activation == 'PReLU':
            _activation = nn.PReLU(inner_dim)
        else:
            raise ValueError(f'{activation} is not a valid activation')

        if GLU_type=='GLU':
            _GLU = nn.GLU(dim=1)
        elif GLU_type=='SwiGLU':
            _GLU = SwiGLU(dim=1)
        else:
            raise ValueError(f'{GLU_type} is not a valid GLU type')

        if use_norm:
            _norm = nn.LayerNorm(dim)
        else:
            _norm = nn.Identity()
        if float(dropout) > 0.:
            _dropout = nn.Dropout(dropout)
        else:
            _dropout = nn.Identity()
        if conv_model_type == 'mode1':
            self.net = nn.Sequential(
                _norm,
                Transpose((1, 2)),
                nn.Conv1d(dim, inner_dim * 2, 1),
                _GLU,
                nn.Conv1d(inner_dim, inner_dim, kernel_size=kernel_size, padding=padding[0], groups=inner_dim),
                _activation,
                nn.Conv1d(inner_dim, dim, 1),
                Transpose((1, 2)),
                _dropout
            )
        else:
            raise ValueError(f'{conv_model_type} is not a valid conv_model_type')

    def forward(self, x):
        return self.net(x)


def calc_same_padding(kernel_size):
    pad = kernel_size // 2
    return (pad, pad - (kernel_size + 1) % 2)


class Transpose(nn.Module):
    def __init__(self, dims):
        super().__init__()
        assert len(dims) == 2, 'dims must be a tuple of two dimensions'
        self.dims = dims

    def forward(self, x):
        return x.transpose(*self.dims)