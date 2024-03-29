import random
from collections import defaultdict

import mlflow
import networkx as nx
import numpy as np
import torch
from torch_geometric.utils import from_networkx
from tqdm import tqdm as tq

from benchmarks.benchmark import Benchmark
from explain_methods import explain_occlusion


class SaturationOld(Benchmark):
    NUM_GRAPHS = 100
    TEST_RATIO = 0.1
    OCCLUSION_SUBSAMPLE_PER_GRAPH = 20

    def create_dataset(self):
        K = 10
        SZ = 100
        N = SZ * K
        sizes = [SZ] * K
        P = 0.15
        Q = 0.04
        PERTURB_COUNT = 500
        mlflow.log_param('P', P)
        mlflow.log_param('Q', Q)
        mlflow.log_param('N', N)
        mlflow.log_param('K', K)
        mlflow.log_param('PERTURB_COUNT', PERTURB_COUNT)
        probs = np.ones((K, K)) * Q
        probs += np.eye(K) * (P - Q)
        g = nx.stochastic_block_model(sizes, probs, directed=True)

        colors = []
        for i in range(K):
            colors.extend([i] * SZ)
        labels = colors.copy()

        for i in random.sample(list(range(N)), PERTURB_COUNT):
            choices = list(range(10))
            colors[i] = random.choice(choices)

        infected_nodes = random.sample(list(range(N)), 15)

        # an infected node infects the neighbors in the outcome
        infected_neighbors = set()
        for i in infected_nodes:
            infected_neighbors.add(i)
            for v in g.neighbors(i):
                infected_neighbors.add(v)

        features = np.zeros((len(g.nodes()), 11))

        for i, c in enumerate(colors):
            features[i, c] = 1

        for i in infected_nodes:
            features[i, 10] = 1

        for i in infected_neighbors:
            labels[i] = labels[i] + 10

        data = from_networkx(g)
        data.x = torch.tensor(features, dtype=torch.float)
        data.y = torch.tensor(labels)
        data.num_classes = len(set(labels))

        infection_sources = defaultdict(list)
        for idx, (u, v) in enumerate(zip(*data.edge_index.cpu().numpy())):
            if features[u, -1]:
                infection_sources[int(v)].append(idx)
        explanations = []
        for node, sources in infection_sources.items():
            if node in infected_nodes:  # no edge explanation for these nodes
                continue
            if len(sources) == 1:  # unique explanation
                explanations.append((node, sources[0]))

        data.explanations = explanations
        return data

    def subsample_nodes(self, explain_function, nodes):
        if explain_function.explain_function == explain_occlusion:
            return random.sample(nodes, self.OCCLUSION_SUBSAMPLE_PER_GRAPH)
        return super().subsample_nodes(explain_function, nodes)

    def evaluate_explanation(self, explain_function, model, test_dataset, explain_name):
        accs = []
        misclassify_count = 0
        for data in test_dataset:
            _, pred = model(data.x, data.edge_index).max(dim=1)
            nodes_to_test = data.explanations
            nodes_to_test = self.subsample_nodes(explain_function, nodes_to_test)
            pbar = tq(nodes_to_test)
            infection_accuracy = 0
            for node_idx in range(len(data.x)):
                if pred[node_idx].item() // 10 == data.y[node_idx].item() // 10:
                    infection_accuracy += 1
            infection_accuracy /= len(data.x)
            mlflow.log_metric('infection_accuracy', infection_accuracy)
            print('infection_accuracy', infection_accuracy)
            for node_idx, correct_edge_id in pbar:
                if pred[node_idx] != data.y[node_idx]:
                    misclassify_count += 1
                    continue
                edge_mask = explain_function(model, node_idx, data.x, data.edge_index, data.y[node_idx].item())
                edge_pos = list(np.argsort(-edge_mask)).index(correct_edge_id)
                accs.append(edge_pos)
                pbar.set_postfix(acc=np.mean(accs))
            mlflow.log_metric('tested_nodes_per_graph', len(edge_pos))
        return accs
