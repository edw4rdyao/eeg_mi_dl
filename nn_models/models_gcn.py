import numpy as np
import torch
from torch import nn
from utils import get_adjacency_matrix, transpose_to_b_1_c_0, init_weight_bias


class Conv2dWithConstraint(nn.Conv2d):
    """Convolution 2d with max norm

    References
    ----------
    from braindecode.models
    """

    def __init__(self, *args, max_norm=1, **kwargs):
        self.max_norm = max_norm
        super(Conv2dWithConstraint, self).__init__(*args, **kwargs)

    def forward(self, x):
        self.weight.data = torch.renorm(
            self.weight.data, p=2, dim=0, maxnorm=self.max_norm
        )
        return super(Conv2dWithConstraint, self).forward(x)


class GraphConvolution(nn.Module):
    def __init__(self, A, in_channels, out_channels, bias=True):
        super(GraphConvolution, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.register_buffer('A', A)
        self.weight = nn.Parameter(torch.randn(self.in_channels, self.out_channels))
        if bias:
            self.bias = nn.Parameter(torch.randn(out_channels))
        else:
            self.register_parameter('bias', None)
        init_weight_bias(self)

    def forward(self, x):
        x = torch.matmul(torch.matmul(self.A, x), self.weight)
        if self.bias is not None:
            x += self.bias
        return x

    def __repr__(self):
        return f"{self.__class__.__name__}({self.in_channels}->{self.out_channels})"


class EEGNetMine(nn.Module):
    def __init__(self, n_channels, n_classes, input_window_size,
                 F1=8, D=2, F2=16, kernel_length=64, drop_p=0.5):
        super(EEGNetMine, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.F1 = F1
        self.D = D
        self.F2 = F2
        self.kernel_length = kernel_length
        self.drop_p = drop_p
        self.input_windows_size = input_window_size
        block_temporal_conv = nn.Sequential(
            # input shape: (B, C, E, T)(Batch, Channel, Electrode, Time)(64, 1, 22, 1000)
            nn.Conv2d(1, self.F1, (1, self.kernel_length),
                      stride=1, bias=False, padding=(0, self.kernel_length // 2)),
            nn.BatchNorm2d(self.F1, momentum=0.01, affine=True, eps=1e-3)
            # output shape: (B, F1, E, T)(64, 8, 22, 1000)
        )
        block_spacial_conv = nn.Sequential(
            # input shape: (B, F1, E, T)(64, 8, 22, 1000)
            Conv2dWithConstraint(self.F1, self.F1 * self.D, (self.n_channels, 1),
                                 max_norm=1, stride=1, bias=False, groups=self.F1, padding=(0, 0)),
            nn.BatchNorm2d(self.F1 * self.D, momentum=0.01, affine=True, eps=1e-3),
            nn.ELU(),
            nn.AvgPool2d(kernel_size=(1, 4), stride=(1, 4)),
            nn.Dropout(p=self.drop_p)
            # output shape: (B, F1 * D, 1, T//4)(64, 16, 1, 1000 // 4)
        )
        block_separable_conv = nn.Sequential(
            # input shape: (B, F1 * D, 1, T//4)(64, 16, 1, 1000 // 4)
            nn.Conv2d(self.F1 * self.D, self.F1 * self.D, (1, 16),
                      stride=1, bias=False, groups=self.F1 * self.D, padding=(0, 16 // 2)),
            nn.Conv2d(self.F1 * self.D, self.F2, (1, 1),
                      stride=1, bias=False, padding=(0, 0)),
            nn.BatchNorm2d(self.F2, momentum=0.01, affine=True, eps=1e-3),
            nn.ELU(),
            nn.AvgPool2d(kernel_size=(1, 8), stride=(1, 8)),
            nn.Dropout(p=self.drop_p),
            nn.Flatten()
            # output shape: (B, F2*T//32)   (64, 16*1000//4//8)
        )
        block_conv = nn.Sequential(
            block_temporal_conv,
            block_spacial_conv,
            block_separable_conv
        )
        out = block_conv(torch.ones((1, 1, self.n_channels, self.input_windows_size), dtype=torch.float32))
        block_classifier = nn.Sequential(
            # input shape: (B, F2*T//32)   (64, 16*1000//4//8)
            nn.Linear(out.cpu().data.numpy().shape[1], self.n_classes),
            nn.LogSoftmax(dim=1)
            # output shape: (B, N)   (64, 4)
        )
        self.net = nn.Sequential(
            block_conv,
            block_classifier
        )
        init_weight_bias(self.net)

    def forward(self, x):
        while len(x.shape) < 4:
            x = x.unsqueeze(-1)
        x = transpose_to_b_1_c_0(x)
        x = self.net(x)
        return x


class EEGConvGcn(nn.Module):
    def __init__(self, n_channels, n_classes, input_window_size, kernel_length=15, drop_p=0.5):
        super(EEGConvGcn, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.input_windows_size = input_window_size
        self.kernel_length = kernel_length
        self.drop_p = drop_p
        A = get_adjacency_matrix(self.n_channels, 'full')
        block_conv = nn.Sequential(
            # input shape: (B, C, E, T)(Batch, Channel, Electrode, Time)(64, 1, 22, 1000)
            nn.Conv2d(1, 1, (1, self.kernel_length), stride=1, padding='same'),
            nn.BatchNorm2d(1, affine=True),
            nn.AvgPool2d(kernel_size=(1, 2), stride=(1, 2)),
            nn.Conv2d(1, 1, (1, self.kernel_length), stride=1, padding='same'),
            nn.BatchNorm2d(1, affine=True),
            nn.ELU(),
            nn.AvgPool2d(kernel_size=(1, 2), stride=(1, 2)),
            nn.Dropout(p=self.drop_p)
            # output shape: (B, C, E, T//4)(64, 1, 22, 1000//4)
        )
        block_gcn = nn.Sequential(
            # input shape: (B, C, E, T//4)(64, 1, 22, 1000//4)
            GraphConvolution(A, self.input_windows_size // 4, self.input_windows_size // 4),
            nn.BatchNorm2d(1, affine=True),
            nn.ELU(),
            nn.AvgPool2d(kernel_size=(1, 8), stride=(1, 8)),
            nn.Dropout(p=self.drop_p),
            nn.Flatten()
            # output shape: (B, C, E*T//32)(64, 22*1000//32)
        )
        block_classifier = nn.Sequential(
            # input shape: (B, C, E*T//32)(64, 1, 22*1000//32)
            nn.Linear(self.input_windows_size // 32 * self.n_channels, 64),
            nn.Dropout(p=self.drop_p),
            nn.Linear(64, self.n_classes),
            nn.LogSoftmax(dim=1)
            # output shape: (B, N)   (64, 4)
        )
        self.net = nn.Sequential(
            block_conv,
            block_gcn,
            block_classifier
        )

    def forward(self, x):
        while len(x.shape) < 4:
            x = x.unsqueeze(-1)
        x = transpose_to_b_1_c_0(x)
        x = self.net(x)
        return x
