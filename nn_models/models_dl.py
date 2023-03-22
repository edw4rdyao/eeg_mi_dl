import numpy as np
import torch
from torch import nn
from torch.nn.functional import elu


class Conv2dWithConstraint(nn.Conv2d):
    """

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


class Ensure4d(nn.Module):
    def forward(self, x):
        while len(x.shape) < 4:
            x = x.unsqueeze(-1)
        return x


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


def _transpose_to_b_1_c_0(x):
    return x.permute(0, 3, 1, 2)


class EEGNetGcn(nn.Module):
    def __init__(self, n_channels, n_classes,
                 F1=8, D=2, F2=16, kernel_length=64, drop_p=0.25):
        super(EEGNetGcn, self).__init__()
        self.n_channels = n_channels
        self.n_classed = n_classes
        self.F1 = F1
        self.D = D
        self.F2 = F2
        self.kernel_length = kernel_length
        self.drop_p = drop_p
        self.ensure_4d = Ensure4d()
        self.dim_trans = Expression(_transpose_to_b_1_c_0)
        self.conv_temporal = nn.Conv2d(1, self.F1, (1, self.kernel_length),
                                       stride=1, bias=False, padding=(0, self.kernel_length // 2))
        self.bn_temporal = nn.BatchNorm2d(self.F1, momentum=0.01, affine=True, eps=1e-3),
        self.conv_spatial = Conv2dWithConstraint(self.F1, self.F1 * self.D, (self.n_channels, 1),
                                                 max_norm=1, stride=1, bias=False, groups=self.F1, padding=(0, 0))
        self.bn_1 = nn.BatchNorm2d(self.F1 * self.D, momentum=0.01, affine=True, eps=1e-3)
        self.elu_1 = Expression(elu)
        self.pool_1 = nn.AvgPool2d(kernel_size=(1, 4), stride=(1, 4))
        self.drop_1 = nn.Dropout(p=self.drop_p)

        self.conv_separable_depth = nn.Conv2d(self.F1 * self.D, self.F1 * self.D, (1, 16),
                                              stride=1, bias=False, groups=self.F1 * self.D, padding=(0, 16 // 2))
        self.conv_separable_point = nn.Conv2d(self.F1 * self.D, self.F2, (1, 1),
                                              stride=1, bias=False, padding=(0, 0))
        self.bn_2 = nn.BatchNorm2d(self.F2, momentum=0.01, affine=True, eps=1e-3)
        self.elu_2 = Expression(elu)
        self.pool_2 = nn.AvgPool2d(kernel_size=(1, 8), stride=(1, 8))
        self.drop_2 = nn.Dropout(p=self.drop_p)




