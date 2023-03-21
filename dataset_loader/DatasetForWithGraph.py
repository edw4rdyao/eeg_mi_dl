import torch
import torch_geometric
from torch.utils.data import Dataset
from . import DatasetFromBraindecode
from torch_geometric.data import Data
from braindecode.datasets import BaseConcatDataset


class DatasetForWithGraph(Dataset):
    def __init__(self, dataset, windows_dataset):
        """
        Parameters
        ----------
        dataset: DatasetFromBraindecode
        windows_dataset: BaseConcatDataset
        """
        self.ds = dataset
        if not self.ds.windows_dataset:
            raise ValueError("the dataset must has been segmented to windows dataset.")
        self.windows_dataset = windows_dataset
        self._edge_index = None
        self._get_graph_data()

    def __len__(self):
        return len(self.ds.windows_dataset)

    def _get_graph_data(self):
        edge_index_list = []
        if self.ds.dataset_name == 'bci2a':
            channel_num = self.ds.get_channel_num()
            assert channel_num == 22
            # build graph adjacency matrix
            for node_i in range(0, channel_num):
                for node_j in range(node_i, channel_num):
                    edge_index_list.append([node_i, node_j])
                    edge_index_list.append([node_j, node_i])
        edge_index_torch = torch.tensor(edge_index_list, dtype=torch.long)
        self._edge_index = edge_index_torch.t().contiguous()

    def __getitem__(self, idx):
        x = self.windows_dataset[idx][0]
        x_torch = torch.tensor(x, dtype=torch.float)
        y = self.windows_dataset[idx][1]
        data_graph = Data(x=x_torch, edge_index=self._edge_index, y=y)
        return data_graph
