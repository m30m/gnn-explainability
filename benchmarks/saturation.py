import itertools
import json
import os
import random
import tempfile
from collections import defaultdict

import mlflow
import networkx as nx
import numpy as np
import torch
from torch_geometric.utils import from_networkx, to_networkx
from tqdm import tqdm as tq

from benchmarks.benchmark import Benchmark
from explain_methods import explain_occlusion


class Saturation(Benchmark):
    NUM_GRAPHS = 10
    TEST_RATIO = 0.4

    def __init__(self, sample_count, num_layers, concat_features, conv_type):
        assert num_layers == 1, "Number of layers should be 1 for saturation benchmark"
        super().__init__(sample_count, num_layers, concat_features, conv_type)
        self.run_count = 0

    def create_dataset(self):
        g = nx.Graph()
        for i in range(2020):
            g.add_node(i)

        colors = [1] * 10 + [2] * 10 + [0] * 2000
        # blue =1
        # red =2
        blue_nodes = [i for i in g.nodes() if colors[i] == 1]
        red_nodes = [i for i in g.nodes() if colors[i] == 2]
        white_nodes = [i for i in g.nodes() if colors[i] == 0]
        # labels 3: both
        # labels 2: red
        # labels 1: blue
        # labels 0: none
        labels = colors[:20] + [0] * 500 + [2] * 500 + [1] * 500 + [3] * 500
        P = 0.015  # probability of edge between two white nodes
        for u, v in itertools.combinations(white_nodes, 2):
            if random.random() < P:
                g.add_edge(u, v)

        for idx, node in enumerate(white_nodes[500:1000]):
            red_count = 1 + (idx % 10)
            for u in random.sample(red_nodes, red_count):
                g.add_edge(node, u)

        for idx, node in enumerate(white_nodes[1000:1500]):
            blue_count = 1 + (idx % 10)
            for u in random.sample(blue_nodes, blue_count):
                g.add_edge(node, u)

        nodes_to_test = []
        for idx, node in enumerate(white_nodes[1500:2000]):
            idx = idx % 100
            blue_count = 1 + (idx // 10)
            red_count = 1 + (idx % 10)
            if red_count == 1 or blue_count == 1:
                nodes_to_test.append(node)
            for u in random.sample(blue_nodes, blue_count):
                g.add_edge(node, u)
            for u in random.sample(red_nodes, red_count):
                g.add_edge(node, u)
        data = from_networkx(g)
        data.x = torch.nn.functional.one_hot(torch.tensor(colors)).float()
        data.y = torch.tensor(labels)
        data.num_classes = 4
        data.nodes_to_test = nodes_to_test
        return data

    def evaluate_explanation(self, explain_function, model, test_dataset):
        accs = []
        all_attributions = []
        for data in test_dataset:
            edge_to_id = {}
            for idx, edge in enumerate((zip(*data.edge_index.numpy()))):
                edge_to_id[edge] = idx
            g = to_networkx(data)
            nodes_to_test = data.nodes_to_test
            nodes_to_test = self.subsample_nodes(explain_function, nodes_to_test)
            pbar = tq(nodes_to_test)
            for node_idx in pbar:
                red_ids = []
                blue_ids = []
                for u in list(g.neighbors(node_idx)):
                    eid = edge_to_id[(u, node_idx)]
                    if data.x[u][1].item():
                        blue_ids.append(eid)
                    if data.x[u][2].item():
                        red_ids.append(eid)
                assert len(red_ids) == 1 or len(blue_ids) == 1, "One color should have single edge explanation"
                edge_mask = explain_function(model, node_idx, data.x, data.edge_index, data.y[node_idx].item())
                red_sum = np.sum(edge_mask[red_ids])
                blue_sum = np.sum(edge_mask[blue_ids])
                sum_ratio = min(red_sum, blue_sum) / max(red_sum, blue_sum)
                accs.append(1 if sum_ratio > 0.1 else 0)
                all_attributions.append({'red': list(edge_mask[red_ids]), 'blue': list(edge_mask[blue_ids])})
                pbar.set_postfix(acc=np.mean(accs))
            mlflow.log_metric('tested_nodes_per_graph', len(nodes_to_test))
        with tempfile.TemporaryDirectory() as tmpdir:
            file_path = os.path.join(tmpdir, 'saturation_log_%d.json' % self.run_count)
            json.dump(all_attributions, open(file_path, 'w'), indent=2)
            mlflow.log_artifact(file_path)
            self.run_count += 1
        return accs
