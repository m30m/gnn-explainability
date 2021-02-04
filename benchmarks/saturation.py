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
from scipy.stats import ks_2samp

from benchmarks.benchmark import Benchmark


class Saturation(Benchmark):
    NUM_GRAPHS = 10
    TEST_RATIO = 0.4
    LR = 0.005

    def __init__(self, sample_count, num_layers, concat_features, conv_type):
        assert num_layers == 1, "Number of layers should be 1 for saturation benchmark"
        super().__init__(sample_count, num_layers, concat_features, conv_type)
        self.run_count = defaultdict(int)

    def create_dataset(self):
        g = nx.Graph()
        for i in range(2000):
            g.add_node(i)

        colors = [1] * 10 + [2] * 10 + [0] * 1980
        # blue =1
        # red =2
        blue_nodes = [i for i in g.nodes() if colors[i] == 1]
        red_nodes = [i for i in g.nodes() if colors[i] == 2]
        white_nodes = [i for i in g.nodes() if colors[i] == 0]
        # labels 3: both
        # labels 2: red
        # labels 1: blue
        # labels 0: none
        labels = colors[:20] + [1] * 990 + [2] * 990
        P = 0.015  # probability of edge between two white nodes
        for u, v in itertools.combinations(white_nodes, 2):
            if random.random() < P:
                g.add_edge(u, v)

        blue_red_combs = list(itertools.permutations(list(range(1, 11)), 2))

        nodes_to_test = []
        for idx, node in enumerate(white_nodes):
            idx = idx % len(blue_red_combs)
            blue_count, red_count = blue_red_combs[idx]
            if abs(red_count - blue_count) > 1:
                nodes_to_test.append(node)
            for u in random.sample(blue_nodes, blue_count):
                g.add_edge(node, u)
            for u in random.sample(red_nodes, red_count):
                g.add_edge(node, u)
            if blue_count > red_count:
                labels[node] = 1
            else:
                labels[node] = 2
        labels = [x - 1 for x in labels]
        data = from_networkx(g)
        data.x = torch.nn.functional.one_hot(torch.tensor(colors)).float()
        data.y = torch.tensor(labels)
        data.num_classes = 2
        data.nodes_to_test = nodes_to_test
        return data

    def is_trained_model_valid(self, test_acc):
        return test_acc > 0.999

    def evaluate_explanation(self, explain_function, model, test_dataset, explain_name):
        accs = []
        all_attributions = []
        for data in test_dataset:
            edge_to_id = {}
            for idx, edge in enumerate((zip(*data.edge_index.cpu().numpy()))):
                edge_to_id[edge] = idx
            g = to_networkx(data)
            nodes_to_test = data.nodes_to_test
            nodes_to_test = self.subsample_nodes(explain_function, nodes_to_test)
            pbar = tq(nodes_to_test)
            for node_idx in pbar:
                red_ids = []
                blue_ids = []
                white_ids = []
                for u in list(g.neighbors(node_idx)):
                    eid = edge_to_id[(u, node_idx)]
                    if data.x[u][1].item():
                        blue_ids.append(eid)
                    elif data.x[u][2].item():
                        red_ids.append(eid)
                    else:
                        white_ids.append(eid)
                edge_mask = explain_function(model, node_idx, data.x, data.edge_index, data.y[node_idx].item())
                red_values = edge_mask[red_ids]
                blue_values = edge_mask[blue_ids]
                white_values = edge_mask[white_ids]
                minority = min(red_values, blue_values, key=len)
                pvalue = ks_2samp(white_values, minority).pvalue
                accs.append(1 - pvalue)
                all_attributions.append({'red': red_values.tolist(),
                                         'blue': blue_values.tolist(),
                                         'white': white_values.tolist()})
                pbar.set_postfix(acc=np.mean(accs))
            mlflow.log_metric('tested_nodes_per_graph', len(nodes_to_test))
        with tempfile.TemporaryDirectory() as tmpdir:
            file_path = os.path.join(tmpdir, 'saturation_%s_%d.json' % (explain_name, self.run_count[explain_name]))
            json.dump(all_attributions, open(file_path, 'w'), indent=2)
            mlflow.log_artifact(file_path)
            self.run_count[explain_name] += 1
        return accs
