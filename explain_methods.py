import networkx as nx
import numpy as np
import torch
from captum.attr import Saliency, IntegratedGradients
from torch_geometric.nn import GNNExplainer
from torch_geometric.utils import to_networkx


def make_model_forward(model):
    def model_forward(edge_mask, node_idx, data):
        out = model(data.x, data.edge_index, edge_mask)
        return out[[node_idx]]

    return model_forward


def explain_sa(model, node_idx, data, target):
    saliency = Saliency(make_model_forward(model))
    input_mask = torch.ones(data.edge_index.shape[1]).requires_grad_(True)
    saliency_mask = saliency.attribute(input_mask, target=target, additional_forward_args=(node_idx, data))

    edge_mask = np.abs(saliency_mask.numpy())
    if edge_mask.max() > 0:
        edge_mask = edge_mask / edge_mask.max()
    return edge_mask


def explain_ig(model, node_idx, data, target):
    ig = IntegratedGradients(make_model_forward(model))
    input_mask = torch.ones(data.edge_index.shape[1]).requires_grad_(True)
    ig_mask = ig.attribute(input_mask, target=target, additional_forward_args=(node_idx, data),
                           internal_batch_size=data.edge_index.shape[1])

    edge_mask = np.abs(ig_mask.detach().numpy())
    if edge_mask.max() > 0:
        edge_mask = edge_mask / edge_mask.max()
    return edge_mask


def explain_occlusion(model, node_idx, data, target):
    depth_limit = len(model.convs) + 1
    pred_prob = model(data.x, data.edge_index)[node_idx][target].item()
    g = to_networkx(data)
    subgraph_nodes = []
    for k, v in nx.shortest_path_length(g, node_idx).items():
        if v < depth_limit:
            subgraph_nodes.append(k)
    subgraph = g.subgraph(subgraph_nodes)
    edge_occlusion_mask = np.ones(data.num_edges, dtype=bool)
    edge_mask = np.zeros(data.num_edges)
    for i in range(data.num_edges):
        u, v = list(data.edge_index[:, i].numpy())
        if (u, v) in subgraph.edges():
            edge_occlusion_mask[i] = False
            prob = model(data.x, data.edge_index[:, edge_occlusion_mask])[node_idx][target].item()
            edge_mask[i] = pred_prob - prob
            edge_occlusion_mask[i] = True
    return edge_mask / (np.abs(edge_mask).max())


def explain_occlusion_undirected(model, node_idx, data, target):
    depth_limit = len(model.convs) + 1
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
    for i in range(data.num_edges):
        u, v = list(data.edge_index[:, i].numpy())
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
    return edge_mask / (np.abs(edge_mask).max())


def explain_gnnexplainer(model, node_idx, data, target):
    explainer = GNNExplainer(model, epochs=200, log=False)
    node_feat_mask, edge_mask = explainer.explain_node(node_idx, data.x, data.edge_index)
    return edge_mask.numpy()
