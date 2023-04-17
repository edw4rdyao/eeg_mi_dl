import torch
import numpy


def get_adjacency_matrix(n_electrodes, mode):
    adjacency = torch.zeros(n_electrodes, n_electrodes)
    if mode == 'full':
        adjacency = torch.ones(n_electrodes, n_electrodes)
    elif mode == 'dis':
        edges = get_edges('bci2a')
        for i, j in edges:
            adjacency[i - 1][j - 1] = 1
            adjacency[j - 1][i - 1] = 1
        for i in range(n_electrodes):
            adjacency[i][i] = 1
    # adjacency = normalize_adjacency_matrix(adjacency)
    return adjacency


def normalize_adjacency_matrix(adjacency):
    degree = torch.pow(adjacency.sum(1).float(), -0.5)
    degree = torch.diag(degree)
    normalize_adjacency = torch.matmul(torch.matmul(degree, adjacency), degree)
    return normalize_adjacency


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


def get_electrode_importance(model):
    trained_model_param = model.state_dict()
    importance = trained_model_param['importance']
    adjacency = trained_model_param['adjacency']
    importance = (importance * adjacency).cpu().numpy()
    importance = numpy.absolute(importance)
    print(importance)
    row, col = importance.shape
    edge_importance_index = numpy.argsort(importance.flatten())
    print(edge_importance_index)
    selected_electrode = set()
    for index in edge_importance_index:
        i = index // row
        j = index % col
        selected_electrode.add(i)
        selected_electrode.add(j)
    selected_electrode = list(selected_electrode)
    print(selected_electrode)
    # electrode_importance = numpy.zeros(row)
    # for i in range(row):
    #     for j in range(col):
    #         if not i == j:
    #             electrode_importance[i] += importance[i][j]
    #             electrode_importance[j] += importance[i][j]
    #         else:
    #             electrode_importance[i] += importance[i][j]
    # print(electrode_importance)
