import torch

from .models import EEGNetReproduce, ASGCNN, ASTGCN, EEGNetGCN, GCNEEGNet, BaseCNN

cuda = torch.cuda.is_available()
