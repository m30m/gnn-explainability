import random
from collections import defaultdict

import mlflow
import networkx as nx
import numpy as np
import torch
from torch_geometric.utils import from_networkx
from tqdm import tqdm as tq

from benchmarks.benchmark import Benchmark


def infection_dataset_old(max_dist=4):  # anything equal or larger than max_dist has a far away label
    g = nx.balanced_tree(3, 6)
    N = len(g.nodes())

    RANDOM_REWIRE = 200
    RANDOM_ADD = 500

    for i in range(RANDOM_REWIRE):
        edge_list = list(g.edges())
        e1, e2 = random.sample(edge_list, 2)
        u1, v1 = e1
        u2, v2 = e2
        # if all 4 nodes are not distinct, rewiring will lead to disconnecting the graph
        if len({u1, v1, u2, v2}) != 4:
            continue
        g.remove_edges_from([e1, e2])
        g.add_edge(u1, v2)
        g.add_edge(u2, v1)
        if not nx.is_connected(g):  # revert if lead to disconnected graph
            g.remove_edge(u1, v2)
            g.remove_edge(u2, v1)
            g.add_edge(u1, v1)
            g.add_edge(u2, v2)

    for i in range(RANDOM_ADD):
        node_list = list(g.nodes())
        u1, v1 = random.sample(node_list, 2)
        g.add_edge(u1, v1)

    assert nx.is_connected(g)
    infected_nodes = random.sample(list(g.nodes()), 50)
    # networkX does not provide multi source BFS so we use dijkstra
    shortest_lengths = nx.multi_source_dijkstra_path_length(g, infected_nodes)
    labels = []
    features = np.zeros((N, 2))
    for u in range(N):
        labels.append(min(max_dist, shortest_lengths[u]))
        features[u, 0] = 1
    for u in infected_nodes:
        features[u, 1] = 1
        features[u, 0] = 0

    test_nodes = []
    # make sure that test nodes have a single shortest path to infected nodes to avoid any ambiguity in explanations
    shortest_lengths_by_node = [nx.single_source_shortest_path_length(g, u) for u in infected_nodes]
    for u in g.nodes():
        min_d = shortest_lengths[u]
        if min_d >= max_dist:  # no explanation is possible for nodes too far from infected nodes
            continue
        cnt = 0

        all_shortest_paths = []
        for sl, infected_node in zip(shortest_lengths_by_node, infected_nodes):
            if sl[u] == min_d:
                all_shortest_paths.extend(list(nx.all_shortest_paths(g, infected_node, u)))
                # shortest_path = nx.single_source_shortest_path(g, infected_node)[u]
        cnt = len(all_shortest_paths)
        assert cnt > 0
        if cnt == 1:
            test_nodes.append((u, all_shortest_paths[0]))
    test_nodes = random.sample(test_nodes, 400)
    test_nodes, shortest_paths = zip(*test_nodes)
    g = g.to_directed()
    eid = 0
    for u, v, d in g.edges(data=True):
        d['id'] = eid
        eid += 1
    explanations = []

    def to_eids(path_by_node):
        eids = []
        for u, v in zip(path_by_node, path_by_node[1:]):
            eids.append(g.edges()[(u, v)]['id'])
        return eids

    for sp in shortest_paths:
        eids = to_eids(sp)
        explanations.append(eids)
    return g, features, labels, list(test_nodes), explanations


class Infection(Benchmark):
    NUM_GRAPHS = 10
    TEST_RATIO = 0.4

    @staticmethod
    def get_accuracy(correct_ids, edge_mask, edge_index):
        correct_count = 0
        correct_edges = list(zip(correct_ids, correct_ids[1:]))

        for x in np.argsort(-edge_mask)[:len(correct_ids)]:
            u, v = edge_index[:, x]
            u, v = u.item(), v.item()
            if (u, v) in correct_edges:
                correct_count += 1
        return correct_count / len(correct_edges)

    @staticmethod
    def get_accuracy_undirected(correct_ids, edge_values):
        correct_count = 0
        correct_edges = list(zip(correct_ids, correct_ids[1:]))

        top_edges = list(sorted([(-value, edge) for edge, value in edge_values.items()]))[:len(correct_ids)]
        for _, (u, v) in top_edges:
            if (u, v) in correct_edges or (v, u) in correct_edges:
                correct_count += 1
        return correct_count / len(correct_edges)

    def create_dataset(self):
        max_dist = self.num_layers  # anything larger than max_dist has a far away label
        g = nx.erdos_renyi_graph(1000, 0.004, directed=True)
        N = len(g.nodes())
        infected_nodes = random.sample(g.nodes(), 50)
        g.add_node('X')  # dummy node for easier computation, will be removed in the end
        for u in infected_nodes:
            g.add_edge('X', u)
        shortest_path_length = nx.single_source_shortest_path_length(g, 'X')
        unique_solution_nodes = []
        unique_solution_explanations = []
        labels = []
        features = np.zeros((N, 2))
        for i in range(N):
            if i == 'X':
                continue
            length = shortest_path_length.get(i, 100) - 1  # 100 is inf distance
            labels.append(min(max_dist + 1, length))
            col = 0 if i in infected_nodes else 1
            features[i, col] = 1
            if 0 < length <= max_dist:
                path_iterator = iter(nx.all_shortest_paths(g, 'X', i))
                unique_shortest_path = next(path_iterator)
                if next(path_iterator, 0) != 0:
                    continue
                unique_shortest_path.pop(0)  # pop 'X' node
                if len(unique_shortest_path) == 0:
                    continue
                unique_solution_explanations.append(unique_shortest_path)
                unique_solution_nodes.append(i)
        g.remove_node('X')
        data = from_networkx(g)
        data.x = torch.tensor(features, dtype=torch.float)
        data.y = torch.tensor(labels)
        data.unique_solution_nodes = unique_solution_nodes
        data.unique_solution_explanations = unique_solution_explanations
        data.num_classes = 1 + max_dist + 1
        print('created one')
        return data

    def evaluate_explanation(self, explain_function, model, test_dataset):
        accs = []
        misclassify_count = 0
        for data in test_dataset:
            _, pred = model(data.x, data.edge_index).max(dim=1)
            nodes_to_test = list(zip(data.unique_solution_nodes, data.unique_solution_explanations))
            nodes_to_test = self.subsample_nodes(explain_function, nodes_to_test)
            pbar = tq(nodes_to_test, disable=False)
            tested_nodes = 0
            for node_idx, correct_ids in pbar:
                if pred[node_idx] != data.y[node_idx]:
                    misclassify_count += 1
                    continue
                tested_nodes += 1
                edge_mask = explain_function(model, node_idx, data.x, data.edge_index, data.y[node_idx].item())
                explain_acc = self.get_accuracy(correct_ids, edge_mask, data.edge_index)
                accs.append(explain_acc)
                pbar.set_postfix(acc=np.mean(accs))
            mlflow.log_metric('tested_nodes_per_graph', tested_nodes)
        return accs
