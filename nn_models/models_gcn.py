import numpy as np
import torch
from torch import nn


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


class Expression(nn.Module):
    """Compute given expression on forward pass.

    References
    ----------
    braindecode.models

    Parameters
    ----------
    expression_fn : callable
        Should accept variable number of objects of type
        `torch.autograd.Variable` to compute its output.
    """

    def __init__(self, expression_fn):
        super(Expression, self).__init__()
        self.expression_fn = expression_fn

    def forward(self, *x):
        return self.expression_fn(*x)

    def __repr__(self):
        if hasattr(self.expression_fn, "func") and hasattr(
                self.expression_fn, "kwargs"
        ):
            expression_str = "{:s} {:s}".format(
                self.expression_fn.func.__name__, str(self.expression_fn.kwargs)
            )
        elif hasattr(self.expression_fn, "__name__"):
            expression_str = self.expression_fn.__name__
        else:
            expression_str = repr(self.expression_fn)
        return (
                self.__class__.__name__ +
                "(expression=%s) " % expression_str
        )


class Ensure4d(nn.Module):
    """Ensure the input shape to be 4D

    References
    ----------
    braindecode.model
    """

    def forward(self, x):
        while len(x.shape) < 4:
            x = x.unsqueeze(-1)
        return x


class EEGNetMine(nn.Module):
    def _get_block_temporal_conv(self):
        block_temporal_conv = nn.Sequential(
            Ensure4d(),
            Expression(_transpose_to_b_1_c_0),
            # input shape: (B, C, E, T)(Batch, Channel, Electrode, Time)(64, 1, 22, 1000)
            nn.Conv2d(1, self.F1, (1, self.kernel_length),
                      stride=1, bias=False, padding=(0, self.kernel_length // 2)),
            nn.BatchNorm2d(self.F1, momentum=0.01, affine=True, eps=1e-3)
            # output shape: (B, F1, E, T)(64, 8, 22, 1000)
        )
        return block_temporal_conv

    def _get_block_spacial_conv(self):
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
        return block_spacial_conv

    def _get_block_separable_conv(self):
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
            # output shape: (B, F2, 1, T//32)   (64, 16, 1, 1000//4//8)
            nn.Flatten()
            # output shape: (B, F2*T//32)   (64, 16*1000//4//8)
        )
        return block_separable_conv

    def _get_block_classifier(self, input_size):
        block_classifier = nn.Sequential(
            # output shape: (B, F2*T//32)   (64, 16*1000//4//8)
            nn.Linear(input_size, self.n_classes),
            nn.LogSoftmax(dim=1)
            # output shape: (B, N)   (64, 4)
        )
        return block_classifier

    def _get_net(self):
        block_conv = nn.Sequential(
            self._get_block_temporal_conv(),
            self._get_block_spacial_conv(),
            self._get_block_separable_conv()
        )
        out = block_conv(torch.ones((1, self.n_channels, self.input_windows_size, 1), dtype=torch.float32))
        block_classifier = self._get_block_classifier(out.cpu().data.numpy().shape[1])
        return nn.Sequential(
            block_conv,
            block_classifier
        )

    def __init__(self, n_channels, n_classes, input_window_size,
                 F1=8, D=2, F2=16, kernel_length=64, drop_p=0.25):
        super(EEGNetMine, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.F1 = F1
        self.D = D
        self.F2 = F2
        self.kernel_length = kernel_length
        self.drop_p = drop_p
        self.input_windows_size = input_window_size
        self.net = self._get_net()
        _init_weight_bias(self.net)

    def forward(self, x):
        x = self.net(x)
        return x


class ConvGcn(nn.Module):
    def _get_block_conv(self):
        block_conv = nn.Sequential(
            Ensure4d(),
            Expression(_transpose_to_b_1_c_0),
            # input shape: (B, C, E, T)(Batch, Channel, Electrode, Time)(64, 1, 22, 1000)
            nn.Conv2d(1, 1, (1, self.kernel_length), stride=1, padding='same'),
            nn.AvgPool2d(kernel_size=(1, 2), stride=(1, 2)),
            nn.Conv2d(1, 1, (1, self.kernel_length), stride=1, padding='same'),
            nn.AvgPool2d(kernel_size=(1, 2), stride=(1, 2))
            # output shape: (B, C, E, T//4)(64, 1, 22, 1000//4)
        )
        return block_conv

    def _get_block_gcn(self):
        block_gcn = nn.Sequential()


def _transpose_to_b_1_c_0(x):
    """transform the dimension

    Parameters
    ----------
    x: torch.Tensor
        input data

    Returns
    -------
    x.permute(0, 3, 1, 2)
    """
    return x.permute(0, 3, 1, 2)


def _init_weight_bias(model):
    """Initialize parameters of all modules by initializing weights 1
     uniform/xavier initialization, and setting biases to zero. Weights from
     batch norm layers are set to 1.

    Parameters
    ----------
    model: nn.Module
    """
    for module in model.modules():
        if hasattr(module, "weight"):
            if not ("BatchNorm" in module.__class__.__name__):
                nn.init.xavier_uniform_(module.weight, gain=1)
            else:
                nn.init.constant_(module.weight, 1)
        if hasattr(module, "bias"):
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)
