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
        self.importance = nn.Parameter(torch.randn(self.A.size()[0], self.A.size()[0]))
        if bias:
            self.bias = nn.Parameter(torch.randn(out_channels))
        else:
            self.register_parameter('bias', None)
        init_weight_bias(self)

    def forward(self, x):
        x = torch.matmul(torch.matmul(torch.mul(self.A, self.importance), x), self.weight)
        if self.bias is not None:
            x += self.bias
        return x


class GraphTemporalConvolution(nn.Module):
    def __init__(self, A, in_channels, out_channels, kernel_length):
        super(GraphTemporalConvolution, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.register_buffer('A', A)
        self.importance = nn.Parameter(torch.randn(in_channels, self.A.size()[0], self.A.size()[0]))
        self.temporal_conv = nn.Conv2d(in_channels=in_channels, out_channels=out_channels,
                                       kernel_size=(1, kernel_length), stride=1, bias=False, padding='same')

    def forward(self, x):
        x = torch.matmul(torch.mul(self.A, self.importance), x)
        x = self.temporal_conv(x)
        return x


class GraphAttentionLayer(nn.Module):
    """Simple GAT layer, similar to https://arxiv.org/abs/1710.10903
    """

    def __init__(self, in_features, out_features, dropout, alpha, concat=True):
        super(GraphAttentionLayer, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.dropout = dropout  # dropout参数
        self.alpha = alpha  # leakyrelu激活的参数
        self.concat = concat  # 如果为true, 再进行elu激活

        # 定义可训练参数，即论文中的W和a
        self.W = nn.Parameter(torch.zeros(size=(in_features, out_features)))
        nn.init.xavier_uniform_(self.W.data, gain=1.414)
        self.a = nn.Parameter(torch.zeros(size=(2 * out_features, 1)))
        nn.init.xavier_uniform_(self.a.data, gain=1.414)
        self.leakyrelu = nn.LeakyReLU(self.alpha)

    def forward(self, inp, adj):
        """
        inp: input_fea [N, in_features]  in_features表示节点的输入特征向量元素个数
        adj: 图的邻接矩阵 维度[N, N] 非零即一，数据结构基本知识
        """
        h = torch.mm(inp, self.W)  # [N, out_features]
        N = h.size()[0]  # N 图的节点数

        a_input = torch.cat([h.repeat(1, N).view(N * N, -1), h.repeat(N, 1)], dim=1).view(N, -1, 2 * self.out_features)
        # [N, N, 2*out_features]
        e = self.leakyrelu(torch.matmul(a_input, self.a).squeeze(2))
        # [N, N, 1] => [N, N] 图注意力的相关系数（未归一化）

        zero_vec = -1e12 * torch.ones_like(e)  # 将没有连接的边置为负无穷
        attention = torch.where(adj > 0, e, zero_vec)  # [N, N]
        # 表示如果邻接矩阵元素大于0时，则两个节点有连接，该位置的注意力系数保留，
        # 否则需要mask并置为非常小的值，原因是softmax的时候这个最小值会不考虑。
        attention = F.softmax(attention, dim=1)  # softmax形状保持不变 [N, N]，得到归一化的注意力权重！
        attention = F.dropout(attention, self.dropout, training=self.training)  # dropout，防止过拟合
        h_prime = torch.matmul(attention, h)  # [N, N].[N, out_features] => [N, out_features]
        # 得到由周围节点通过注意力权重进行更新的表示
        if self.concat:
            return F.elu(h_prime)
        else:
            return h_prime

    def __repr__(self):
        return self.__class__.__name__ + ' (' + str(self.in_features) + ' -> ' + str(self.out_features) + ')'


class EEGNetRp(nn.Module):
    def __init__(self, n_channels, n_classes, input_window_size,
                 F1=8, D=2, F2=16, kernel_length=64, drop_p=0.5):
        super(EEGNetRp, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.F1 = F1
        self.D = D
        self.F2 = F2
        self.kernel_length = kernel_length
        self.drop_p = drop_p
        self.input_windows_size = input_window_size
        A = get_adjacency_matrix(self.n_channels, 'dis')
        self.block_temporal_conv = nn.Sequential(
            # input shape: (B, C, E, T)(Batch, Channel, Electrode, Time)(64, 1, 22, 1000)
            nn.Conv2d(1, self.F1, (1, self.kernel_length),
                      stride=1, bias=False, padding='same'),
            nn.BatchNorm2d(self.F1, momentum=0.01, affine=True, eps=1e-3)
            # output shape: (B, F1, E, T)(64, 8, 22, 1000)
        )
        self.block_spacial_conv = nn.Sequential(
            # input shape: (B, F1, E, T)(64, 8, 22, 1000)
            Conv2dWithConstraint(self.F1, self.F1 * self.D, (self.n_channels, 1),
                                 max_norm=1, stride=1, bias=False, groups=self.F1, padding=(0, 0)),
            nn.BatchNorm2d(self.F1 * self.D, momentum=0.01, affine=True, eps=1e-3),
            nn.ELU(),
            nn.AvgPool2d(kernel_size=(1, 4), stride=(1, 4)),
            nn.Dropout(p=self.drop_p)
            # output shape: (B, F1 * D, 1, T//4)(64, 16, 1, 1000 // 4)
        )
        self.block_separable_conv = nn.Sequential(
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
            self.block_temporal_conv,
            self.block_spacial_conv,
            self.block_separable_conv
        )
        out = block_conv(torch.ones((1, 1, self.n_channels, self.input_windows_size), dtype=torch.float32))
        self.block_classifier = nn.Sequential(
            # input shape: (B, F2*T//32)   (64, 16*1000//4//8)
            nn.Linear(out.cpu().data.numpy().shape[1], self.n_classes),
            nn.LogSoftmax(dim=1)
            # output shape: (B, N)   (64, 4)
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


class ST_GCN(nn.Module):
    """Reference: ST-GNN for EEG Motor Imagery Classification(https://ieeexplore.ieee.org/document/9926806)

    """

    def __init__(self, n_channels, n_classes, input_window_size, kernel_length=15, drop_p=0.5):
        super(ST_GCN, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.input_windows_size = input_window_size
        self.kernel_length = kernel_length
        self.drop_p = drop_p
        A = get_adjacency_matrix(self.n_channels, 'full')
        block_conv = nn.Sequential(
            # input shape: (B, C, E, T)(Batch, Channel, Electrode, Time)(64, 1, 22, 1000)
            nn.Conv2d(1, 1, (1, self.kernel_length), stride=1, padding='same'),
            nn.BatchNorm2d(1),
            nn.ReLU(),
            nn.AvgPool2d(kernel_size=(1, 4), stride=(1, 4)),
            nn.Conv2d(1, 1, (1, self.kernel_length), stride=1, padding='same'),
            nn.BatchNorm2d(1),
            nn.ReLU(),
            nn.AvgPool2d(kernel_size=(1, 4), stride=(1, 4)),
            # output shape:
        )
        block_gcn = nn.Sequential(
            # input shape:
            GraphConvolution(A, self.input_windows_size // 16, self.input_windows_size // 16),
            nn.ReLU(),
            nn.Dropout(p=self.drop_p),
            nn.Flatten()
            # output shape:
        )
        block_classifier = nn.Sequential(
            # input shape:
            nn.Linear(self.input_windows_size // 16 * self.n_channels, 64),
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
        A = get_adjacency_matrix(self.n_channels, 'full')
        self.block_temporal_conv = nn.Sequential(
            # input shape: (B, C, E, T)(Batch, Channel, Electrode, Time)(64, 1, 22, 1000)
            nn.Conv2d(1, self.F1, (1, self.kernel_length),
                      stride=1, bias=False, padding='same'),
            nn.BatchNorm2d(self.F1, momentum=0.01, affine=True, eps=1e-3)
            # output shape: (B, F1, E, T)(64, 8, 22, 1000)
        )
        self.block_spacial_conv = nn.Sequential(
            # input shape: (B, F1, E, T)(64, 8, 22, 1000)
            nn.Conv2d(self.F1, self.F1 * self.D, (self.n_channels, 1),
                      stride=1, bias=False, groups=self.F1, padding='same'),
            nn.BatchNorm2d(self.F1 * self.D, momentum=0.01, affine=True, eps=1e-3),
            nn.ELU(),
            nn.AvgPool2d(kernel_size=(1, 4), stride=(1, 4)),
            nn.Dropout(p=self.drop_p)
            # output shape: (B, F1 * D, 1, T//4)(64, 16, 22, 1000 // 4)
        )
        self.block_separable_conv = nn.Sequential(
            # input shape: (B, F1 * D, 1, T//4)(64, 16, 22, 1000 // 4)
            nn.Conv2d(self.F1 * self.D, self.F1 * self.D, (1, 16),
                      stride=1, bias=False, groups=self.F1 * self.D, padding=(0, 16 // 2)),
            nn.Conv2d(self.F1 * self.D, self.F2, (1, 1),
                      stride=1, bias=False, padding=(0, 0)),
            nn.BatchNorm2d(self.F2, momentum=0.01, affine=True, eps=1e-3),
            nn.ELU(),
            nn.AvgPool2d(kernel_size=(1, 8), stride=(1, 8)),
            nn.Dropout(p=self.drop_p),
            # output shape: (B, F2, 22, T//32)   (64, 16, 22, 1000//4//8)
        )
        block_conv = nn.Sequential(
            self.block_temporal_conv,
            self.block_spacial_conv,
            self.block_separable_conv
        )
        out = block_conv(torch.ones((1, 1, self.n_channels, self.input_windows_size), dtype=torch.float32))
        self.block_classifier = nn.Sequential(
            # input shape: (B, F2, 22, T//32)   (64, 16, 22, 1000//4//8)
            GraphConvolution(A, out.cpu().data.numpy().shape[3], out.cpu().data.numpy().shape[3]),
            nn.BatchNorm2d(self.F2, momentum=0.01, affine=True, eps=1e-3),
            nn.ELU(),
            nn.MaxPool2d(kernel_size=(self.n_channels, 1), stride=(self.n_channels, 1)),
            nn.Flatten(),
            nn.Linear(
                out.cpu().data.numpy().shape[1] * out.cpu().data.numpy().shape[3],
                self.n_classes),
            nn.LogSoftmax(dim=1)
            # output shape: (B, N)   (64, 4)
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
        A = get_adjacency_matrix(self.n_channels, 'full')
        self.block_temporal_conv = nn.Sequential(
            # input shape: (B, C, E, T)(Batch, Channel, Electrode, Time)(64, 1, 22, 1000)
            nn.Conv2d(1, self.F1, (1, self.kernel_length),
                      stride=1, bias=False, padding='same'),
            nn.BatchNorm2d(self.F1, momentum=0.01, affine=True, eps=1e-3),
            GraphTemporalConvolution(A, self.F1, self.F1, 8),
            nn.BatchNorm2d(self.F1, momentum=0.01, affine=True, eps=1e-3),
            nn.ELU(),
            GraphTemporalConvolution(A, self.F1, self.F1, 8),
            nn.BatchNorm2d(self.F1, momentum=0.01, affine=True, eps=1e-3),
            nn.ELU(),
            GraphTemporalConvolution(A, self.F1, self.F1, 8),
            nn.BatchNorm2d(self.F1, momentum=0.01, affine=True, eps=1e-3),
            nn.ELU(),
        )
        self.block_spacial_conv = nn.Sequential(
            # input shape: (B, F1, E, T)(64, 8, 22, 1000)
            Conv2dWithConstraint(self.F1, self.F1 * self.D, (self.n_channels, 1),
                                 max_norm=1, stride=1, bias=False, groups=self.F1, padding=(0, 0)),
            nn.BatchNorm2d(self.F1 * self.D, momentum=0.01, affine=True, eps=1e-3),
            nn.ELU(),
            nn.AvgPool2d(kernel_size=(1, 4), stride=(1, 4)),
            nn.Dropout(p=self.drop_p)
            # output shape: (B, F1 * D, 1, T//4)(64, 16, 1, 1000 // 4)
        )
        self.block_separable_conv = nn.Sequential(
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
            self.block_temporal_conv,
            self.block_spacial_conv,
            self.block_separable_conv
        )
        out = block_conv(torch.ones((1, 1, self.n_channels, self.input_windows_size), dtype=torch.float32))
        self.block_classifier = nn.Sequential(
            # input shape: (B, F2*T//32)   (64, 16*1000//4//8)
            nn.Linear(out.cpu().data.numpy().shape[1], self.n_classes),
            nn.LogSoftmax(dim=1)
            # output shape: (B, N)   (64, 4)
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


class ASTGCN(nn.Module):
    def __init__(self, n_channels, n_classes, input_window_size, kernel_length=64, drop_p=0.5):
        super(ASTGCN, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.input_windows_size = input_window_size
        self.kernel_length = kernel_length
        self.drop_p = drop_p
        A = get_adjacency_matrix(self.n_channels, 'dis')
        self.block_conv_1 = nn.Sequential(
            nn.Conv2d(1, 16, (1, self.kernel_length), stride=1, padding='same'),
            nn.ELU(),
        )
        self.block_stb_1 = nn.Sequential(
            GraphConvolution(A, self.input_windows_size, self.input_windows_size),
            nn.ELU(),
            nn.Conv2d(16, 16, (1, 8), stride=1, padding='same'),
            nn.ELU(),
            nn.AvgPool2d(kernel_size=(1, 2), stride=(1, 2)),
            nn.Dropout(p=self.drop_p)
        )
        self.block_stb_2 = nn.Sequential(
            GraphConvolution(A, self.input_windows_size // 2, self.input_windows_size // 2),
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
            nn.Softmax(dim=1)
        )

    def forward(self, x):
        while len(x.shape) < 4:
            x = x.unsqueeze(-1)
        x = x.permute(0, 3, 1, 2)
        x = self.block_conv_1(x)
        x = self.block_stb_1(x)
        x = self.block_stb_2(x)
        x = self.block_conv_2(x)
        x = self.block_classify(x)
        return x
