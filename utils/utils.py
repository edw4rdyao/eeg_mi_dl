import os
import yaml
import json
from torch import nn
from skorch.callbacks import Callback


def read_yaml(path):
    with open(path, encoding='utf-8') as f:
        config = yaml.load(f.read(), Loader=yaml.FullLoader)
        return config


def save_config(config, save_dir):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    with open(save_dir + 'config.json', "w") as f:
        json.dump(config, f)
    return


def transpose_to_4d_input(x):
    while len(x.shape) < 4:
        x = x.unsqueeze(-1)
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


class SaveHistory(Callback):
    def __init__(self, file_path):
        self.file_path = file_path

    def on_train_end(self, net, **kwargs):
        history = net.history
        history.to_file(self.file_path + 'all_history.json')
