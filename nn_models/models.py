import torch
from torch import nn
from torch.nn import functional

from utils import get_adjacency_matrix, transpose_to_4d_input, init_weight_bias


class Conv2dWithConstraint(nn.Conv2d):
    def __init__(self, *args, max_norm=1, **kwargs):
        self.max_norm = max_norm
        super(Conv2dWithConstraint, self).__init__(*args, **kwargs)

    def forward(self, x):
        self.weight.data = torch.renorm(
            self.weight.data, p=2, dim=0, maxnorm=self.max_norm
        )
        return super(Conv2dWithConstraint, self).forward(x)


class GraphConvolution(nn.Module):
    def __init__(self, in_channels, out_channels, bias=True):
        super(GraphConvolution, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.weight = nn.Parameter(torch.randn(self.in_channels, self.out_channels))
        if bias:
            self.bias = nn.Parameter(torch.randn(out_channels))
        else:
            self.register_parameter('bias', None)
        init_weight_bias(self)

    def forward(self, x, support):
        x = torch.matmul(torch.matmul(support, x), self.weight)
        if self.bias is not None:
            x += self.bias
        x = functional.elu(x)
        return x


class GraphTemporalConvolution(nn.Module):
    def __init__(self, adjacency, in_channels, out_channels, kernel_length):
        super(GraphTemporalConvolution, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.register_buffer('adjacency', adjacency)
        self.importance = nn.Parameter(torch.randn(in_channels, self.adjacency.size()[0], self.adjacency.size()[0]))
        self.temporal_conv = nn.Conv2d(in_channels=in_channels, out_channels=out_channels,
                                       kernel_size=(1, kernel_length), stride=1, bias=False, padding='same')

    def forward(self, x):
        x = torch.matmul(torch.mul(self.adjacency, self.importance), x)
        x = self.temporal_conv(x)
        return x


class EEGNetReproduce(nn.Module):
    def __init__(self, n_channels, n_classes, input_window_size,
                 F1=8, D=2, F2=16, kernel_length=64, drop_p=0.25):
        super(EEGNetReproduce, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.F1 = F1
        self.D = D
        self.F2 = F2
        self.kernel_length = kernel_length
        self.drop_p = drop_p
        self.input_windows_size = input_window_size
        self.block_temporal_conv = nn.Sequential(
            nn.Conv2d(1, self.F1, (1, self.kernel_length),
                      stride=1, bias=False, padding='same'),
            nn.BatchNorm2d(self.F1)
        )
        self.block_spacial_conv = nn.Sequential(
            Conv2dWithConstraint(self.F1, self.F1 * self.D, (self.n_channels, 1),
                                 max_norm=1, stride=1, bias=False, groups=self.F1, padding=(0, 0)),
            nn.BatchNorm2d(self.F1 * self.D),
            nn.ELU(),
            nn.AvgPool2d(kernel_size=(1, 4), stride=(1, 4)),
            nn.Dropout(p=self.drop_p)
        )
        self.block_separable_conv = nn.Sequential(
            nn.Conv2d(self.F1 * self.D, self.F1 * self.D, (1, 16),
                      stride=1, bias=False, groups=self.F1 * self.D, padding='same'),
            nn.Conv2d(self.F1 * self.D, self.F2, (1, 1),
                      stride=1, bias=False, padding=(0, 0)),
            nn.BatchNorm2d(self.F2),
            nn.ELU(),
            nn.AvgPool2d(kernel_size=(1, 8), stride=(1, 8)),
            nn.Dropout(p=self.drop_p),
            nn.Flatten()
        )
        block_conv = nn.Sequential(
            self.block_temporal_conv,
            self.block_spacial_conv,
            self.block_separable_conv
        )
        out = block_conv(torch.ones((1, 1, self.n_channels, self.input_windows_size), dtype=torch.float32))
        self.block_classifier = nn.Sequential(
            nn.Linear(out.cpu().data.numpy().shape[1], self.n_classes),
            # nn.LogSoftmax(dim=1)
        )

    def forward(self, x):
        while len(x.shape) < 4:
            x = x.unsqueeze(-1)
        x = x.permute(0, 3, 1, 2)
        x = self.block_temporal_conv(x)
        x = self.block_spacial_conv(x)
        x = self.block_separable_conv(x)
        x = self.block_classifier(x)
        return x


class BaseCNN(nn.Module):
    def __init__(self, n_channels, n_classes, input_window_size):
        super(BaseCNN, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.input_window_size = input_window_size
        self.block_conv = nn.Sequential(
            nn.Conv2d(1, 16, (1, 8), stride=1, padding='same'),
            nn.BatchNorm2d(16, momentum=0.01, eps=1e-3),
            nn.ELU(),
            nn.AvgPool2d(kernel_size=(1, 8), stride=(1, 8)),
            nn.Dropout(p=0.5)
        )
        self.block_classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(p=0.5),
            nn.Linear(self.input_window_size // 8 * self.n_channels * 16, 64),
            nn.Linear(64, self.n_classes),
            # nn.LogSoftmax(dim=1)
        )

    def forward(self, x):
        x = transpose_to_4d_input(x)
        x = self.block_conv(x)
        x = self.block_classifier(x)
        return x


class ASGCNN(nn.Module):
    def __init__(self, n_channels, n_classes, input_window_size, graph_strategy='AG', kernel_length=8, bias_s=False):
        super(ASGCNN, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.input_windows_size = input_window_size
        self.kernel_length = kernel_length
        self.graph_strategy = graph_strategy
        adjacency = get_adjacency_matrix(self.n_channels, self.graph_strategy)
        self.register_buffer('adjacency', adjacency)
        if self.graph_strategy == 'AG':
            self.importance = nn.Parameter(torch.randn(self.n_channels, self.n_channels))
        self.block_conv = nn.Sequential(
            nn.Conv2d(1, 8, (1, self.kernel_length), stride=1, padding='same'),
            nn.BatchNorm2d(8, momentum=0.01, eps=1e-3),
            nn.ELU(),
            nn.AvgPool2d(kernel_size=(1, 8), stride=(1, 8)),

            nn.Conv2d(8, 16, (1, self.kernel_length), stride=1, padding='same'),
            nn.BatchNorm2d(16, momentum=0.01, eps=1e-3),
            nn.ELU(),
            nn.AvgPool2d(kernel_size=(1, 4), stride=(1, 4)),
            nn.Dropout(p=0.5)
        )

        self.block_gcn = GraphConvolution(self.input_windows_size // 32, self.input_windows_size // 32)

        self.block_classifier = nn.Sequential(
            nn.Conv2d(16, 16, (self.n_channels, 1), bias=bias_s, stride=1, padding='same', groups=8),
            nn.BatchNorm2d(16, momentum=0.01, eps=1e-3),
            nn.ELU(),
            nn.AvgPool2d(kernel_size=(self.n_channels, 1), stride=(self.n_channels, 1)),
            nn.Flatten(),
            nn.Dropout(p=0.6),
            nn.Linear(self.input_windows_size // 32 * 16, 64),
            nn.Linear(64, self.n_classes),
            # nn.LogSoftmax(dim=1)
        )

    def forward(self, x):
        x = transpose_to_4d_input(x)
        x = self.block_conv(x)
        if self.graph_strategy == 'AG':
            x = self.block_gcn(x, torch.mul(self.adjacency, self.importance))
        else:
            x = self.block_gcn(x, self.adjacency)
        x = self.block_classifier(x)
        return x


class STGCN(nn.Module):
    """Reference: ST-GNN for EEG Motor Imagery Classification(https://ieeexplore.ieee.org/document/9926806)

    """

    def __init__(self, n_channels, n_classes, input_window_size, kernel_length=15, drop_prob=0.5):
        super(STGCN, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.input_windows_size = input_window_size
        self.kernel_length = kernel_length
        self.drop_prob = drop_prob
        adjacency = get_adjacency_matrix(self.n_channels, 'full')
        self.block_conv = nn.Sequential(
            nn.Conv2d(1, 1, (1, self.kernel_length), stride=1, padding='same'),
            nn.BatchNorm2d(1, momentum=0.01, eps=1e-3),
            nn.ELU(),
            nn.AvgPool2d(kernel_size=(1, 4), stride=(1, 4)),
            nn.Conv2d(1, 1, (1, self.kernel_length), stride=1, padding='same'),
            nn.BatchNorm2d(1, momentum=0.01, eps=1e-3),
            nn.ELU(),
            nn.AvgPool2d(kernel_size=(1, 8), stride=(1, 8))
        )
        self.block_gcn = nn.Sequential(
            GraphConvolution(adjacency, self.input_windows_size // 32, self.input_windows_size // 32),
            nn.ELU(),
            nn.Flatten()
        )
        self.block_classifier = nn.Sequential(
            nn.Linear(self.input_windows_size // 32 * self.n_channels, 64),
            nn.Dropout(p=self.drop_prob),
            nn.Linear(64, self.n_classes),
            # nn.LogSoftmax(dim=1)
        )

    def forward(self, x):
        x = transpose_to_4d_input(x)
        x = self.block_conv(x)
        x = self.block_gcn(x)
        x = self.block_classifier(x)
        return x


class EEGNetGCN(nn.Module):
    def __init__(self, n_channels, n_classes, input_window_size,
                 F1=8, D=2, F2=16, kernel_length=32, drop_p=0.5):
        super(EEGNetGCN, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.F1 = F1
        self.D = D
        self.F2 = F2
        self.kernel_length = kernel_length
        self.drop_p = drop_p
        self.input_windows_size = input_window_size
        adjacency = get_adjacency_matrix(self.n_channels, 'full')
        self.block_temporal_conv = nn.Sequential(
            nn.Conv2d(1, self.F1, (1, self.kernel_length),
                      stride=1, bias=False, padding='same'),
            nn.BatchNorm2d(self.F1, momentum=0.01, affine=True, eps=1e-3)
        )
        self.block_spacial_conv = nn.Sequential(
            nn.Conv2d(self.F1, self.F1 * self.D, (self.n_channels, 1),
                      stride=1, bias=False, groups=self.F1, padding='same'),
            nn.BatchNorm2d(self.F1 * self.D, momentum=0.01, affine=True, eps=1e-3),
            nn.ELU(),
            nn.AvgPool2d(kernel_size=(1, 4), stride=(1, 4)),
            nn.Dropout(p=self.drop_p)
        )
        self.block_separable_conv = nn.Sequential(
            nn.Conv2d(self.F1 * self.D, self.F1 * self.D, (1, 16),
                      stride=1, bias=False, groups=self.F1 * self.D, padding=(0, 16 // 2)),
            nn.Conv2d(self.F1 * self.D, self.F2, (1, 1),
                      stride=1, bias=False, padding=(0, 0)),
            nn.BatchNorm2d(self.F2, momentum=0.01, affine=True, eps=1e-3),
            nn.ELU(),
            nn.AvgPool2d(kernel_size=(1, 8), stride=(1, 8)),
            nn.Dropout(p=self.drop_p),
        )
        block_conv = nn.Sequential(
            self.block_temporal_conv,
            self.block_spacial_conv,
            self.block_separable_conv
        )
        out = block_conv(torch.ones((1, 1, self.n_channels, self.input_windows_size), dtype=torch.float32))
        self.block_classifier = nn.Sequential(
            GraphConvolution(adjacency, out.cpu().data.numpy().shape[3], out.cpu().data.numpy().shape[3]),
            nn.BatchNorm2d(self.F2, momentum=0.01, affine=True, eps=1e-3),
            nn.ELU(),
            nn.MaxPool2d(kernel_size=(self.n_channels, 1), stride=(self.n_channels, 1)),
            nn.Flatten(),
            nn.Linear( out.cpu().data.numpy().shape[1] * out.cpu().data.numpy().shape[3], self.n_classes),
            # nn.LogSoftmax(dim=1)
        )

    def forward(self, x):
        x = transpose_to_4d_input(x)
        x = self.block_temporal_conv(x)
        x = self.block_spacial_conv(x)
        x = self.block_separable_conv(x)
        x = self.block_classifier(x)
        return x


class GCNEEGNet(nn.Module):
    def __init__(self, n_channels, n_classes, input_window_size,
                 F1=8, D=2, F2=16, kernel_length=64, drop_p=0.5):
        super(GCNEEGNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.F1 = F1
        self.D = D
        self.F2 = F2
        self.kernel_length = kernel_length
        self.drop_p = drop_p
        self.input_windows_size = input_window_size
        adjacency = get_adjacency_matrix(self.n_channels, 'full')
        self.block_temporal_conv = nn.Sequential(
            nn.Conv2d(1, self.F1, (1, self.kernel_length),
                      stride=1, bias=False, padding='same'),
            nn.BatchNorm2d(self.F1, momentum=0.01, affine=True, eps=1e-3),
            GraphTemporalConvolution(adjacency, self.F1, self.F1, 8),
            nn.BatchNorm2d(self.F1, momentum=0.01, affine=True, eps=1e-3),
            nn.ELU(),
            GraphTemporalConvolution(adjacency, self.F1, self.F1, 8),
            nn.BatchNorm2d(self.F1, momentum=0.01, affine=True, eps=1e-3),
            nn.ELU(),
            GraphTemporalConvolution(adjacency, self.F1, self.F1, 8),
            nn.BatchNorm2d(self.F1, momentum=0.01, affine=True, eps=1e-3),
            nn.ELU(),
        )
        self.block_spacial_conv = nn.Sequential(
            Conv2dWithConstraint(self.F1, self.F1 * self.D, (self.n_channels, 1),
                                 max_norm=1, stride=1, bias=False, groups=self.F1, padding=(0, 0)),
            nn.BatchNorm2d(self.F1 * self.D, momentum=0.01, affine=True, eps=1e-3),
            nn.ELU(),
            nn.AvgPool2d(kernel_size=(1, 4), stride=(1, 4)),
            nn.Dropout(p=self.drop_p)
        )
        self.block_separable_conv = nn.Sequential(
            nn.Conv2d(self.F1 * self.D, self.F1 * self.D, (1, 16),
                      stride=1, bias=False, groups=self.F1 * self.D, padding=(0, 16 // 2)),
            nn.Conv2d(self.F1 * self.D, self.F2, (1, 1),
                      stride=1, bias=False, padding=(0, 0)),
            nn.BatchNorm2d(self.F2, momentum=0.01, affine=True, eps=1e-3),
            nn.ELU(),
            nn.AvgPool2d(kernel_size=(1, 8), stride=(1, 8)),
            nn.Dropout(p=self.drop_p),
            nn.Flatten()
        )
        block_conv = nn.Sequential(
            self.block_temporal_conv,
            self.block_spacial_conv,
            self.block_separable_conv
        )
        out = block_conv(torch.ones((1, 1, self.n_channels, self.input_windows_size), dtype=torch.float32))
        self.block_classifier = nn.Sequential(
            nn.Linear(out.cpu().data.numpy().shape[1], self.n_classes),
            # nn.LogSoftmax(dim=1)
        )

    def forward(self, x):
        x = transpose_to_4d_input(x)
        x = self.block_temporal_conv(x)
        x = self.block_spacial_conv(x)
        x = self.block_separable_conv(x)
        x = self.block_classifier(x)
        return x


class ASTGCN(nn.Module):
    def __init__(self, n_channels, n_classes, input_window_size, kernel_length=64, drop_p=0.5):
        super(ASTGCN, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.input_windows_size = input_window_size
        self.kernel_length = kernel_length
        self.drop_p = drop_p
        adjacency = get_adjacency_matrix(self.n_channels, 'dis')
        self.block_conv_1 = nn.Sequential(
            nn.Conv2d(1, 16, (1, self.kernel_length), stride=1, padding='same'),
            nn.ELU(),
        )
        self.block_stb_1 = nn.Sequential(
            GraphConvolution(adjacency, self.input_windows_size, self.input_windows_size),
            nn.ELU(),
            nn.Conv2d(16, 16, (1, 8), stride=1, padding='same'),
            nn.ELU(),
            nn.AvgPool2d(kernel_size=(1, 2), stride=(1, 2)),
            nn.Dropout(p=self.drop_p)
        )
        self.block_stb_2 = nn.Sequential(
            GraphConvolution(adjacency, self.input_windows_size // 2, self.input_windows_size // 2),
            nn.ELU(),
            nn.Conv2d(16, 16, (1, 8), stride=1, padding='same'),
            nn.ELU(),
            nn.AvgPool2d(kernel_size=(1, 2), stride=(1, 2)),
            nn.Dropout(p=self.drop_p)
        )
        self.block_conv_2 = nn.Sequential(
            nn.Conv2d(16, 32, (self.n_channels, 1), stride=1, padding=(0, 0), groups=16),
            nn.BatchNorm2d(32),
            nn.ELU(),
            nn.AvgPool2d(kernel_size=(1, 4), stride=(1, 4))
        )
        self.block_classify = nn.Sequential(
            nn.Flatten(),
            nn.Linear(self.input_windows_size // 16 * 32, self.n_classes),
            # nn.Softmax(dim=1)
        )

    def forward(self, x):
        x = transpose_to_4d_input(x)
        x = self.block_conv_1(x)
        x = self.block_stb_1(x)
        x = self.block_stb_2(x)
        x = self.block_conv_2(x)
        x = self.block_classify(x)
        return x
