import torch


def get_adjacency_matrix(n_electrodes, mode):
    A = torch.zeros(n_electrodes, n_electrodes)
    if mode == 'full':
        A = torch.ones(n_electrodes, n_electrodes)
    elif mode == 'dis':
        edges = get_edges('bci2a')
        for i, j in edges:
            A[i - 1][j - 1] = 1
            A[j - 1][i - 1] = 1
        for i in range(n_electrodes):
            A[i][i] = 1
    print("Adjacency Matrix:", A)
    A = normalize_adjacency_matrix(A)
    return A


def normalize_adjacency_matrix(A):
    D = torch.pow(A.sum(1).float(), -0.5)
    D = torch.diag(D)
    normalize_A = torch.matmul(torch.matmul(D, A), D)
    return normalize_A


def get_edges(dataset):
    edges = []
    if dataset == 'bci2a':
        edges = [
            (1, 2), (1, 3), (1, 4), (1, 5), (1, 6),
            (2, 3), (2, 7), (2, 8),
            (3, 4), (3, 9),
            (4, 5), (4, 10),
            (5, 6), (5, 11),
            (6, 12), (6, 13),
            (7, 8), (7, 14),
            (8, 9), (8, 14),
            (9, 10), (9, 15),
            (10, 11), (10, 16),
            (11, 12), (11, 17),
            (12, 13), (12, 18),
            (13, 18),
            (14, 15), (14, 19),
            (15, 16), (15, 19),
            (16, 17), (16, 20),
            (17, 18), (17, 21),
            (18, 21),
            (19, 20), (19, 22),
            (20, 21), (20, 22),
            (21, 22)
        ]
    return edges
