import os.path as osp

import networkx as nx
import numpy as np
import torch
import torch.nn.functional as F
from torch.nn import Sequential, Linear, ReLU
from torch_geometric.datasets import TUDataset
from torch_geometric.data import DataLoader
from torch_geometric.utils import from_networkx, to_networkx
from torch_geometric.nn import GINConv, global_add_pool, MessagePassing, \
    GraphConv, GCNConv

from matplotlib import pyplot as plt
from captum.attr import Saliency, IntegratedGradients
from tqdm.notebook import tqdm as tq
from collections import defaultdict


class Net1(torch.nn.Module):
    def __init__(self, num_features, aggr='add'):
        super(Net1, self).__init__()
        dim = 32

        self.conv1 = GraphConv(num_features, dim, aggr=aggr)
        self.conv2 = GraphConv(dim, dim, aggr=aggr)
        self.conv3 = GraphConv(dim, dim, aggr=aggr)
        self.conv4 = GraphConv(dim, dim, aggr=aggr)
        self.conv5 = GraphConv(dim, dim, aggr=aggr)

        self.fc1 = Linear(5 * dim, dim)
        self.fc2 = Linear(dim, 2)

    def forward(self, x, edge_index, batch, edge_weight=None):
        xs = []
        x = F.relu(self.conv1(x, edge_index, edge_weight))
        xs.append(x)
        x = F.relu(self.conv2(x, edge_index, edge_weight))
        xs.append(x)
        x = F.relu(self.conv3(x, edge_index, edge_weight))
        xs.append(x)
        x = F.relu(self.conv4(x, edge_index, edge_weight))
        xs.append(x)
        x = F.relu(self.conv5(x, edge_index, edge_weight))
        xs.append(x)
        x = torch.cat(xs, dim=1)
        x = global_add_pool(x, batch)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x, dim=-1)


class Net2(torch.nn.Module):
    def __init__(self, num_features, aggr='add'):
        super(Net2, self).__init__()

        dim = 32

        self.conv1 = GraphConv(num_features, dim, aggr=aggr)
        self.conv2 = GraphConv(dim, dim, aggr=aggr)
        self.conv3 = GraphConv(dim, dim, aggr=aggr)
        self.conv4 = GraphConv(dim, dim, aggr=aggr)
        self.conv5 = GraphConv(dim, dim, aggr=aggr)

        self.fc1 = Linear(dim, dim)
        self.fc2 = Linear(dim, 2)

    def forward(self, x, edge_index, batch, edge_weight=None):
        x = F.relu(self.conv1(x, edge_index, edge_weight))
        x = F.relu(self.conv2(x, edge_index, edge_weight))
        x = F.relu(self.conv3(x, edge_index, edge_weight))
        x = F.relu(self.conv4(x, edge_index, edge_weight))
        x = F.relu(self.conv5(x, edge_index, edge_weight))
        x = global_add_pool(x, batch)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x, dim=-1)


class Net3(torch.nn.Module):
    def __init__(self, num_features, aggr='add'):
        super(Net3, self).__init__()
        dim = 32

        self.conv1 = GraphConv(num_features, dim, aggr=aggr)

        self.fc1 = Linear(dim, dim)
        self.fc2 = Linear(dim, 2)

    def forward(self, x, edge_index, batch, edge_weight=None):
        x = F.relu(self.conv1(x, edge_index, edge_weight))
        x = global_add_pool(x, batch)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x, dim=-1)


class Net4(torch.nn.Module):
    def __init__(self, num_features, aggr='add'):
        super(Net4, self).__init__()
        dim = 32

        self.pre_fc1 = Linear(num_features, dim)
        self.pre_fc2 = Linear(dim, dim)

        self.conv1 = GraphConv(dim, dim, aggr=aggr)
        self.conv2 = GraphConv(dim, dim, aggr=aggr)
        self.conv3 = GraphConv(dim, dim, aggr=aggr)
        self.conv4 = GraphConv(dim, dim, aggr=aggr)
        self.conv5 = GraphConv(dim, dim, aggr=aggr)

        self.fc1 = Linear(5 * dim, dim)
        self.fc2 = Linear(dim, 2)

    def forward(self, x, edge_index, batch, edge_weight=None):
        x = F.relu(self.pre_fc1(x))
        x = F.relu(self.pre_fc2(x))
        xs = []
        x = F.relu(self.conv1(x, edge_index, edge_weight))
        xs.append(x)
        x = F.relu(self.conv2(x, edge_index, edge_weight))
        xs.append(x)
        x = F.relu(self.conv3(x, edge_index, edge_weight))
        xs.append(x)
        x = F.relu(self.conv4(x, edge_index, edge_weight))
        xs.append(x)
        x = F.relu(self.conv5(x, edge_index, edge_weight))
        xs.append(x)
        x = torch.cat(xs, dim=1)
        x = global_add_pool(x, batch)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return F.log_softmax(x, dim=-1)


class Net5(torch.nn.Module):
    def __init__(self, num_features, aggr='add'):
        super(Net5, self).__init__()

        dim = 32

        self.conv1 = GraphConv(num_features, dim, aggr=aggr)
        self.conv2 = GraphConv(dim, dim, aggr=aggr)
        self.conv3 = GraphConv(dim, dim, aggr=aggr)
        self.conv4 = GraphConv(dim, dim, aggr=aggr)
        self.conv5 = GraphConv(dim, dim, aggr=aggr)

        self.fc1 = Linear(5 * dim, dim)
        self.fc2 = Linear(dim, 2)

    def forward(self, x, edge_index, batch, edge_weight=None):
        xs = []
        x = F.relu(self.conv1(x, edge_index, edge_weight))
        xs.append(x)
        x = F.relu(self.conv2(x, edge_index, edge_weight))
        xs.append(x)
        x = F.relu(self.conv3(x, edge_index, edge_weight))
        xs.append(x)
        x = F.relu(self.conv4(x, edge_index, edge_weight))
        xs.append(x)
        x = F.relu(self.conv5(x, edge_index, edge_weight))
        xs.append(x)
        x = torch.cat(xs, dim=1)
        x = global_add_pool(x, batch)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return F.log_softmax(x, dim=-1)
