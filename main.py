import json
import random
from pathlib import Path

import typer

import os.path as osp

import networkx as nx
import numpy as np
import torch
import torch.nn.functional as F
from torch.nn import Sequential, Linear, ReLU
from torch_geometric.datasets import TUDataset
from torch_geometric.data import DataLoader
from torch_geometric.utils import from_networkx, to_networkx
from torch_geometric.nn import GINConv, global_add_pool, MessagePassing, GraphConv, GCNConv
from models import *

from matplotlib import pyplot as plt
from captum.attr import Saliency, IntegratedGradients
from tqdm import tqdm as tq
from collections import defaultdict

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def explain(model, method, data, target=0):
    batch = torch.zeros(data.x.shape[0], dtype=int).to(device)

    def model_forward(edge_mask, data):
        out = model(data.x, data.edge_index, batch, edge_mask)
        return out

    input_mask = torch.ones(data.edge_index.shape[1]).requires_grad_(True).to(device)
    if method == 'ig':
        ig = IntegratedGradients(model_forward)
        mask = ig.attribute(input_mask, target=target,
                            additional_forward_args=(data.to(device),),
                            internal_batch_size=data.edge_index.shape[1])
    elif method == 'saliency':
        saliency = Saliency(model_forward)
        mask = saliency.attribute(input_mask, target=target,
                                  additional_forward_args=(data.to(device),))
    else:
        raise Exception('Unknown explanation method')

    edge_mask = np.abs(mask.cpu().detach().numpy())
    if edge_mask.max() > 0:  # avoid division by zero
        edge_mask = edge_mask / edge_mask.max()
    return edge_mask


def create_graph(ohe: bool):
    g = nx.erdos_renyi_graph(20, 0.2)
    cnt = 100
    while cnt > 2:  # we make sure that there is at most on edge between two colored nodes
        colored_nodes = random.sample(list(g.nodes()), 3)
        label = 0
        cnt = 0
        for u in colored_nodes:
            for v in colored_nodes:
                if u != v and g.has_edge(u, v):
                    label = 1
                    cnt += 1
    data = from_networkx(g)
    data.y = torch.tensor([label])

    if ohe:
        data.x = torch.zeros((g.number_of_nodes(), 2))
        data.x[:, 0] = 1
        for u in colored_nodes:
            data.x[u, 0] = 0
            data.x[u, 1] = 1
    else:
        data.x = torch.ones((g.number_of_nodes(), 1))
        for u in colored_nodes:
            data.x[u, 0] = 0

    #     data.x = torch.tensor(np.random.normal(loc=-5,size=(g.number_of_nodes(),1)))
    #     for u in colored_nodes:
    #         data.x[u,0]=np.random.normal(loc=5)
    return data


def train(model, optimizer, train_loader):
    model.train()
    loss_all = 0
    for data in train_loader:
        data = data.to(device)
        optimizer.zero_grad()
        output = model(data.x, data.edge_index, data.batch)
        loss = F.nll_loss(output, data.y)
        loss.backward()
        loss_all += loss.item() * data.num_graphs
        optimizer.step()
    return loss_all / len(train_loader.dataset)


def test(model, loader):
    model.eval()

    correct = 0
    for data in loader:
        data = data.to(device)
        output = model(data.x, data.edge_index, data.batch)
        pred = output.max(dim=1)[1]
        correct += pred.eq(data.y).sum().item()
    return correct / len(loader.dataset)


def train_and_test(model, train_loader, test_loader):
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0003, weight_decay=5e-4)
    pbar = tq(range(1, 201))
    for epoch in pbar:
        train_loss = train(model, optimizer, train_loader)
        train_acc = test(model, train_loader)
        test_acc = test(model, test_loader)
        pbar.set_postfix(train_loss=train_loss, train_acc=train_acc, test_acc=test_acc)


#         print('Epoch: {:03d}, Train Loss: {:.7f}, '
#               'Train Acc: {:.7f}, Test Acc: {:.7f}'.format(epoch, train_loss,
#                                                            train_acc, test_acc))


def calculate_avg_wrong(model, method, dataset):
    model.eval()
    avg_wrong_masks = []

    for dds in tq(dataset):
        if not dds.y.item():
            continue
        mask_sum = []
        edge_mask = explain(model, method, dds, target=dds.y.item())
        for value, u, v in list(zip(edge_mask, *dds.edge_index.cpu().numpy())):
            if dds.x[u, 0].item() == 0.0 and dds.x[v, 0].item() == 0.0:  # works in both cases of OHE
                continue
            mask_sum.append(value)
        avg_wrong_masks.append(np.mean(mask_sum))
    return avg_wrong_masks


def main(model_name: str,
         output_path: Path = typer.Argument(..., help='output path for simulation'),
         sample_count: int = typer.Option(10, help='How many times to retry the whole experiment'),
         aggr_method: str = typer.Option('add', help='Aggregation method for convolutional layers'),
         explain_method: str = typer.Option('ig', help='Explanation method to use, can be ig or saliency'),
         ohe: bool = typer.Option(True, help='Encode node colors as OHE or in a single feature')):
    if not output_path.exists():
        output_path.mkdir(parents=True)
    arguments = {
        'model_name': model_name,
        'sample_count': sample_count,
        'aggr_method': aggr_method,
        'explain_method': explain_method,
        'ohe': ohe,
    }
    json.dump(arguments, (output_path / 'args.json').open('w'), indent=4)
    print(f"Using device {device}")
    for i in tq(range(sample_count)):
        GRAPH_NUM = 1000
        dataset = [create_graph(ohe) for i in range(GRAPH_NUM)]
        test_dataset = dataset[:len(dataset) // 10]
        train_dataset = dataset[len(dataset) // 10:]
        test_loader = DataLoader(test_dataset, batch_size=64)
        train_loader = DataLoader(train_dataset, batch_size=64)
        num_features = 2 if ohe else 1
        model = eval(model_name)(num_features, aggr_method).to(device)
        train_and_test(model, train_loader, test_loader)
        avg_wrong_masks = calculate_avg_wrong(model, explain_method, dataset)
        json.dump(avg_wrong_masks, (output_path / f'sim_{i}.json').open('w'))


if __name__ == "__main__":
    typer.run(main)
