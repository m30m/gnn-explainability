import random
from collections import defaultdict

import mlflow
import networkx as nx
import numpy as np
import torch
from torch_geometric.utils import from_networkx
from tqdm import tqdm as tq

from benchmarks.benchmark import Benchmark


class Saturation(Benchmark):
    NUM_GRAPHS = 100
    TEST_RATIO = 0.1

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
        g = nx.stochastic_block_model(sizes, probs)

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
                infection_sources[v].append(idx)
        explanations = []
        for node, sources in infection_sources.items():
            if node in infected_nodes:  # no edge explanation for these nodes
                continue
            if len(sources) == 1:  # unique explanation
                explanations.append((node, sources[0]))

        data.explanations = explanations
        return data

    def evaluate_explanation(self, explain_function, model, test_dataset):
        accs = []
        misclassify_count = 0
        for data in test_dataset:
            _, pred = model(data.x, data.edge_index).max(dim=1)
            pbar = tq(data.explanations)
            for node_idx, correct_edge_id in pbar:
                if pred[node_idx] != data.y[node_idx]:
                    misclassify_count += 1
                    continue
                edge_mask = explain_function(model, node_idx, data.x, data.edge_index, data.y[node_idx].item())
                edge_pos = list(np.argsort(-edge_mask)).index(correct_edge_id)
                accs.append(edge_pos)
                pbar.set_postfix(acc=np.mean(accs))
        return accs
