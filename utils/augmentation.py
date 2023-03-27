from braindecode.augmentation import SignFlip, FrequencyShift
import torch
from torch import nn


def get_augmentation_transform(sample_freq):
    freq_shift = FrequencyShift(
        probability=.5,
        sfreq=sample_freq,
        max_delta_freq=2.
    )
    sign_flip = SignFlip(probability=.1)
    transforms = [
        freq_shift,
        sign_flip
    ]
    return transforms


def transpose_to_b_1_c_0(x):
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


def init_weight_bias(model):
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
