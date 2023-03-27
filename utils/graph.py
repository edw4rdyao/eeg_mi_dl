import torch


def get_adjacency_matrix(n_electrodes, mode):
    A = torch.zeros(n_electrodes, n_electrodes)
    if mode == 'full':
        A = torch.ones(n_electrodes, n_electrodes)
    A = normalize_adjacency_matrix(A)
    return A


def normalize_adjacency_matrix(A):
    D = torch.pow(A.sum(1).float(), -0.5)
    D = torch.diag(D)
    normalize_A = torch.matmul(torch.matmul(D, A), D)
    return normalize_A
