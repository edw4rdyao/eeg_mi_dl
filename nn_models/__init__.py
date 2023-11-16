import torch

from .models_gcn import EEGNetRp, ASGCNN, ASTGCN, EEGNetGCN, GCNEEGNet, BaseCNN

cuda = torch.cuda.is_available()
