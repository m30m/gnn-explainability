import networkx as nx
import numpy as np
import torch
from captum.attr import Saliency, IntegratedGradients
from torch_geometric.data import Data
from torch_geometric.nn import GNNExplainer
from torch_geometric.utils import to_networkx

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def model_forward(edge_mask, model, node_idx, x, edge_index):
    out = model(x, edge_index, edge_mask)
    return out[[node_idx]]


def explain_sa(model, node_idx, x, edge_index, target):
    saliency = Saliency(model_forward)
    input_mask = torch.ones(edge_index.shape[1]).requires_grad_(True).to(device)
    saliency_mask = saliency.attribute(input_mask, target=target,
                                       additional_forward_args=(model, node_idx, x, edge_index))

    edge_mask = saliency_mask.cpu().numpy()
    return edge_mask


def explain_ig(model, node_idx, x, edge_index, target):
    ig = IntegratedGradients(model_forward)
    input_mask = torch.ones(edge_index.shape[1]).requires_grad_(True).to(device)
    ig_mask = ig.attribute(input_mask, target=target, additional_forward_args=(model, node_idx, x, edge_index),
                           internal_batch_size=edge_index.shape[1])

    edge_mask = ig_mask.cpu().detach().numpy()
    return edge_mask


def explain_occlusion(model, node_idx, x, edge_index, target):
    depth_limit = len(model.convs) + 1
    data = Data(x=x, edge_index=edge_index)
    pred_prob = model(data.x, data.edge_index)[node_idx][target].item()
    g = to_networkx(data)
    subgraph_nodes = []
    for k, v in nx.shortest_path_length(g, node_idx).items():
        if v < depth_limit:
            subgraph_nodes.append(k)
    subgraph = g.subgraph(subgraph_nodes)
    edge_occlusion_mask = np.ones(data.num_edges, dtype=bool)
    edge_mask = np.zeros(data.num_edges)
    edge_index_numpy = data.edge_index.cpu().numpy()
    for i in range(data.num_edges):
        u, v = list(edge_index_numpy[:, i])
        if (u, v) in subgraph.edges():
            edge_occlusion_mask[i] = False
            prob = model(data.x, data.edge_index[:, edge_occlusion_mask])[node_idx][target].item()
            edge_mask[i] = pred_prob - prob
            edge_occlusion_mask[i] = True
    return edge_mask


def explain_occlusion_undirected(model, node_idx, x, edge_index, target):
    depth_limit = len(model.convs) + 1
    data = Data(x=x, edge_index=edge_index)
    pred_prob = model(data.x, data.edge_index)[node_idx][target].item()
    g = to_networkx(data)
    subgraph_nodes = []
    for k, v in nx.shortest_path_length(g, node_idx).items():
        if v < depth_limit:
            subgraph_nodes.append(k)
    subgraph = g.subgraph(subgraph_nodes)
    edge_occlusion_mask = np.ones(data.num_edges, dtype=bool)
    edge_mask = np.zeros(data.num_edges)
    reverse_edge_map = {}
    edge_index_numpy = data.edge_index.cpu().numpy()
    for i in range(data.num_edges):
        u, v = list(edge_index_numpy[:, i])
        reverse_edge_map[(u, v)] = i

    for (u, v) in subgraph.edges():
        if u > v:  # process each edge once
            continue
        i1 = reverse_edge_map[(u, v)]
        i2 = reverse_edge_map[(v, u)]
        edge_occlusion_mask[[i1, i2]] = False
        prob = model(data.x, data.edge_index[:, edge_occlusion_mask])[node_idx][target].item()
        edge_mask[[i1, i2]] = pred_prob - prob
        edge_occlusion_mask[[i1, i2]] = True
    return edge_mask


def explain_gnnexplainer(model, node_idx, x, edge_index, target):
    explainer = GNNExplainer(model, epochs=200, log=False)
    node_feat_mask, edge_mask = explainer.explain_node(node_idx, x, edge_index)
    return edge_mask.cpu().numpy()
