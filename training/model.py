import torch
import torchmetrics
from torchvision import transforms
from torch_scatter import scatter_add
from torch.utils.data import Dataset, random_split
import torch.nn as nn
import torch.nn.functional as F

from torch_geometric.data import Data
from torch_geometric.utils import add_self_loops
from torch_geometric.nn import SuperGATConv, GCNConv, GATConv, TopKPooling, global_mean_pool, global_max_pool, MessagePassing


class Model(nn.Module):

    def __init__(self, config):
        super(Model, self).__init__()
    
        self.n_layers = config["n_layers"]
        self.input_dim = config["input_dim"]
        self.inter_connect = config["inter_connect"]

        self.convs = nn.ModuleList([])
        self.bns   = nn.ModuleList([])
        for i in range(self.n_layers):
            if i == 0:
                cur_input_dim  = self.input_dim
            else:
                cur_input_dim  = cur_out_dim
            cur_hidden_dim = cur_input_dim
            cur_out_dim    = cur_hidden_dim

            self.convs.append(SuperGATConv(cur_input_dim, cur_hidden_dim, 
                                        heads=config["n_heads"],
                                        dropout=config["dropout"], 
                                        attention_type='MX',
                                        concat=False,
                                        edge_sample_ratio=config["edge_sample_ratio"], 
                                        neg_sample_ratio=config["neg_sample_ratio"], 
                                        is_undirected=False))
            self.bns.append(nn.BatchNorm1d(cur_out_dim))

        self.output_dim = cur_out_dim
        self.relu = nn.ReLU()

        extra_dim = 32 * 2 * self.n_layers          # extra feature dim resulted from global pooling
        tmp_dim = 124 + extra_dim if config["inter_connect"] is True else 124

        self.linear  = nn.Linear(124*cur_out_dim, 124)
        self.bn1 = nn.BatchNorm1d(124)
        self.linear2  = nn.Linear(tmp_dim, config['n_class'])
        
    def forward(self, x, edge_index, batch):
        
        inter_results = []

        # x: (batch size*n_channel(# of node), hidden_feature_num)
        x = x.reshape(-1, self.input_dim)

        for i in range(self.n_layers):
            x = self.convs[i](x, edge_index)
            x = self.bns[i](x)
            x = self.relu(x)
            inter_results.append(torch.cat((global_mean_pool(x, batch), global_max_pool(x, batch)), 1))

        # readout
        # out: (batch size, n_node*hidden_feature_num)
        x = x.reshape(-1, 124*self.output_dim)

        # out: (batch size, class_num)
        x = self.linear(x)
        x = self.bn1(x)
        if self.inter_connect is True:
            for inter_result in inter_results:
                x = torch.cat((x, inter_result), 1)
        x = self.relu(x)
        x = self.linear2(x)

        return x



class XaiModel(nn.Module):

    def __init__(self, config):
        super(XaiModel, self).__init__()
    
        self.n_layers = config["n_layers"]
        self.input_dim = config["input_dim"]
        self.inter_connect = config["inter_connect"]

        self.convs = nn.ModuleList([])
        self.bns   = nn.ModuleList([])
        for i in range(self.n_layers):
            if i == 0:
                cur_input_dim  = self.input_dim
            else:
                cur_input_dim  = cur_out_dim
            cur_hidden_dim = cur_input_dim
            cur_out_dim    = cur_hidden_dim

            self.convs.append(SuperGATConv(cur_input_dim, cur_hidden_dim, 
                                        heads=config["n_heads"],
                                        dropout=config["dropout"], 
                                        attention_type='MX',
                                        concat=False,
                                        edge_sample_ratio=config["edge_sample_ratio"], 
                                        neg_sample_ratio=config["neg_sample_ratio"], 
                                        is_undirected=False))
            self.bns.append(nn.BatchNorm1d(cur_out_dim))

        self.output_dim = cur_out_dim
        self.relu = nn.ReLU()

        extra_dim = 32 * 2 * self.n_layers         
        tmp_dim = 124 + extra_dim if config["inter_connect"] is True else 124

        self.linear  = nn.Linear(124*cur_out_dim, 124)
        self.bn1 = nn.BatchNorm1d(124)
        self.linear2  = nn.Linear(tmp_dim, config['n_class'])
        
    def forward(self, x, edge_index, batch):

        inter_results = []

        # x: (batch size*n_channel(# of node), hidden_feature_num)
        x = x.reshape(-1, self.input_dim)

        for i in range(self.n_layers):
            x = self.convs[i](x, edge_index)
            x = self.bns[i](x)
            x = self.relu(x)
            inter_results.append(torch.cat((global_mean_pool(x, batch), global_max_pool(x, batch)), 1))

        # readout
        # out: (batch size, n_node*hidden_feature_num)
        x = x.reshape(-1, 124*self.output_dim)

        # out: (batch size, class_num)
        x = self.linear(x)
        x = self.bn1(x)
        if self.inter_connect is True:
            for inter_result in inter_results:
                x = torch.cat((x, inter_result), 1)
        x = self.relu(x)
        x = self.linear2(x)

        return x

