import os
import csv
import math
import time
import random
import numpy as np
import matplotlib.pyplot as plt

import torch
import torchmetrics
from torchvision import transforms
from torch_scatter import scatter_add
from torch.utils.data import Dataset, random_split
import torch.nn as nn
import torch.nn.functional as F

from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.utils import add_self_loops
from torch_geometric.nn import SuperGATConv, GCNConv, GATConv, TopKPooling, global_mean_pool, global_max_pool, MessagePassing

class MyDataset(Dataset):
    def __init__(self, config, mode, fold):

        self.mode = mode
        self.n_channel = 124
        self.n_timestamps = 32
        self.edge_mode = config["edge_mode"]
        self.add_self_loops = config["add_self_loops"]
        self.k = config["k"]
        
        self.sub_id = config["sub_id"]
        self.data_path = config["data_path"]
        
        data = np.load(f'{self.data_path}/S{self.sub_id}.npy', allow_pickle=True).item()
        fold_size = int(data['x'].shape[0] * 0.1)
        if mode == 'train':
            x = np.concatenate((data['x'][:fold_size*fold], data['x'][fold_size*(fold+1):]), axis=0)
            y = np.concatenate((data['y'][:fold_size*fold], data['y'][fold_size*(fold+1):]), axis=0)
        elif mode == 'val':
            x = data['x'][fold_size*fold : fold_size*(fold+1)]
            y = data['y'][fold_size*fold : fold_size*(fold+1)]
        self.x = torch.tensor(x).to(torch.float32)
        self.y = torch.LongTensor(y - 1)

        self.edge_index = self._init_edge_index(edge_mode=self.edge_mode, k=self.k)

    def __len__(self):
        return self.x.shape[0]

    def __getitem__(self, idx):
        sample = {
            'x': self.x[idx],
            'y': self.y[idx],
            'edge_index': self.edge_index
        }
        return sample

    def _init_edge_index(self, edge_mode='complete_graph', k=None):
        
        edge_index = []
        if edge_mode == "complete_graph":
            for i in range(self.n_channel):
                for j in range(self.n_channel):
                    if i != j:
                        edge_index.append([i, j])

        elif edge_mode == 'distance_knn':
            ranked_neighbors = np.load('../data/3d_ranked_neighbors.npz')['ranked_neighbors']
            for i, neighbors in enumerate(ranked_neighbors):
                for j, neighbor in enumerate(neighbors):
                    if j < k:
                        edge_index.append([i, neighbor])

        edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()

        if self.add_self_loops:
            edge_index, _ = add_self_loops(edge_index, num_nodes=124)

        return edge_index

def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)

def generate_loader(config, fold):

    g = torch.Generator()
    g.manual_seed(0)

    dataset = MyDataset(config, mode="train", fold=fold)
    data_list = []
    for sample in dataset:
        data = Data(x=sample['x'], y=sample['y'], edge_index=sample['edge_index'], num_nodes=124)
        data_list.append(data)
    train_loader = DataLoader(data_list, batch_size=config['batch_size'], num_workers=config['num_workers'], shuffle=True, worker_init_fn=seed_worker, generator=g)

    dataset = MyDataset(config, mode="val", fold=fold)
    data_list = []
    for sample in dataset:
        data = Data(x=sample['x'], y=sample['y'], edge_index=sample['edge_index'], num_nodes=124)
        data_list.append(data)
    val_loader = DataLoader(data_list, batch_size=config['batch_size'], num_workers=config['num_workers'], shuffle=False, worker_init_fn=seed_worker, generator=g)

    return train_loader, val_loader