import random

import networkx as nx
import numpy as np
import torch
from torch_geometric.data import Data
from torch_geometric.utils import k_hop_subgraph
import mlflow
from tqdm import tqdm


def make_coo_edges_by_id(g, N, rewired_neighbors):
    edges = []
    edges_rewired = []
    for u, v, d in sorted(list(g.edges(data=True)), key=lambda x: x[2]['id']):
        edges.append([u, v])
        if u >= N:
            u = rewired_neighbors[v]
        if v >= N:
            v = rewired_neighbors[u]
        edges_rewired.append([u, v])
    return torch.tensor(edges, dtype=torch.long).t().contiguous(), \
           torch.tensor(edges_rewired, dtype=torch.long).t().contiguous()


def make_data():
    K = 10
    SZ = 100
    N = SZ * K
    P = 0.15
    Q = 0.04
    PERTURB_COUNT = 300
    DUMMY_COUNT = 500
    mlflow.log_param('P', P)
    mlflow.log_param('Q', Q)
    mlflow.log_param('N', N)
    mlflow.log_param('K', K)
    mlflow.log_param('PERTURB_COUNT', PERTURB_COUNT)
    mlflow.log_param('DUMMY_COUNT', DUMMY_COUNT)
    sizes = [SZ] * K

    probs = np.ones((K, K)) * Q
    probs += np.eye(K) * (P - Q)
    g = nx.stochastic_block_model(sizes, probs)

    colors = []
    for i in range(K):
        colors.extend([i + 1] * SZ)
    colors.extend([0] * (DUMMY_COUNT))  # color of dummy nodes
    labels = colors.copy()

    for i in random.sample(list(range(N)), PERTURB_COUNT):
        choices = list(range(1, 11))
        # choices.remove(colors[i])
        colors[i] = random.choice(choices)
        # colors[i]=0

    rewired_neighbors = {}
    dummy_node_i = 0
    for i in random.sample(list(range(N)), DUMMY_COUNT):
        g.add_edge(i, dummy_node_i + N)
        dummy_node_i += 1
        rewired_neighbor = i
        while rewired_neighbor == i:
            start_i = i - (i % 100)
            rewired_neighbor = random.randint(start_i, start_i + 99)
        rewired_neighbors[i] = rewired_neighbor
    dummy_mask = torch.zeros(len(g.nodes()), dtype=bool)
    dummy_mask[N:] = True
    g = g.to_directed()
    dummy_edges = torch.zeros(len(g.edges()), dtype=bool)
    eid = 0
    for u, v, d in g.edges(data=True):
        d['id'] = eid
        if u >= N or v >= N:
            dummy_edges[eid] = True
        eid += 1
    edge_index, edge_index_rewired = make_coo_edges_by_id(g, N, rewired_neighbors)

    features = np.zeros((len(g.nodes()), 11))

    for i, c in enumerate(colors):
        features[i, c] = 1

    data = Data(x=torch.tensor(features, dtype=torch.float),
                edge_index=edge_index,
                y=torch.tensor(labels))
    data.dummy_mask = dummy_mask
    data.dummy_edges = dummy_edges
    data.num_classes = len(set(labels))
    data.edge_index_rewired = edge_index_rewired

    return data


def evaluate_explanation(explain_function, model, test_dataset):
    accs = []
    for dss in test_dataset:
        bads = 0
        before_afters = []
        depth_limit = len(model.convs)
        tests = 0
        pbar = tqdm(range(1000))
        model_cache = model(dss.x, dss.edge_index)
        model_cache_rewired = model(dss.x, dss.edge_index_rewired)
        for node_idx in pbar:
            prob, label = model_cache[[node_idx]].softmax(dim=1).max(dim=1)
            prob_rewired, label_rewired = model_cache_rewired[[node_idx]].softmax(dim=1).max(dim=1)
            target = dss.y[node_idx].item()
            if label_rewired.item() == target and prob_rewired.item() > prob.item():
                pass
            else:
                # print(f'not true for node_idx {node_idx} first label: {label.item()} rewired label: {label_rewired.item()} diff {prob_rewired.item()-prob.item()}')
                continue
            final_mask = (k_hop_subgraph(node_idx, depth_limit, dss.edge_index)[3] &
                          k_hop_subgraph(node_idx, depth_limit, dss.edge_index_rewired)[3] &
                          dss.dummy_edges)
            final_mask = final_mask

            attribution = explain_function(model, node_idx, dss.x, dss.edge_index, target)[final_mask.cpu()]
            attribution_rewired = explain_function(model, node_idx, dss.x, dss.edge_index_rewired, target)[final_mask.cpu()]

            before_afters.append((attribution.mean(), attribution_rewired.mean()))
            if attribution.mean() > attribution_rewired.mean():
                # print(attribution.mean() , attribution_rewired.mean())
                bads += 1
                # print(node_idx)
            tests += 1
            pbar.set_postfix(bads=bads / (tests), tests=tests)
        print('Bad attributions', bads, 'Total Tests', tests)
        accs.append((1 - bads / tests))
    return accs
