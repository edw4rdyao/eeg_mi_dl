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


def get_electrode_importance(model_param):
    importance = model_param['importance']
    adjacency = model_param['adjacency']
    importance = (importance * adjacency).cpu().numpy()
    importance = numpy.absolute(importance)
    print(importance)
    row, col = importance.shape

    electrode_name = ['FC5', 'FC3', 'FC1', 'FCz', 'FC2', 'FC4', 'FC6', 'C5', 'C3', 'C1', 'Cz', 'C2', 'C4',
                      'C6', 'CP5', 'CP3', 'CP1', 'CPz', 'CP2', 'CP4', 'CP6', 'Fp1', 'Fpz', 'Fp2', 'AF7',
                      'AF3', 'AFz', 'AF4', 'AF8', 'F7', 'F5', 'F3', 'F1', 'Fz', 'F2', 'F4', 'F6', 'F8',
                      'FT7', 'FT8', 'T7', 'T8', 'T9', 'T10', 'TP7', 'TP8', 'P7', 'P5', 'P3', 'P1', 'Pz',
                      'P2', 'P4', 'P6', 'P8', 'PO7', 'PO3', 'POz', 'PO4', 'PO8', 'O1', 'Oz', 'O2', 'Iz']
    electrode_importance = numpy.zeros(row)
    for i in range(row):
        for j in range(col):
            if not i == j:
                electrode_importance[i] += importance[i][j]
                electrode_importance[j] += importance[i][j]
            else:
                electrode_importance[i] += importance[i][j]
    print(electrode_importance)
    top_electrode_index = numpy.argsort(-electrode_importance)
    top32_electrode_name = []
    last32_electrode_name = []
    for i in top_electrode_index[:32]:
        top32_electrode_name.append(electrode_name[i])
    for j in top_electrode_index[32:]:
        last32_electrode_name.append(electrode_name[j])
    print("top32 node", top32_electrode_name)
    print("last32 node", last32_electrode_name)
    #
    # edge_importance_index = numpy.argsort(-importance.flatten())
    # edge_top_electrode_name = []
    # for index in edge_importance_index:
    #     i = index // row
    #     j = index % col
    #     if electrode_name[i] not in edge_top_electrode_name:
    #         edge_top_electrode_name.append(electrode_name[i])
    #     if electrode_name[j] not in edge_top_electrode_name:
    #         edge_top_electrode_name.append(electrode_name[j])
    # print("edge:", edge_top_electrode_name)
