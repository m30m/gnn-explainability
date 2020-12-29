import random
from collections import defaultdict

import mlflow
import networkx as nx
import numpy as np
import torch
from torch_geometric.utils import k_hop_subgraph, from_networkx
from tqdm import tqdm


from benchmarks.benchmark import Benchmark
from explain_methods import explain_pgmexplainer


class Rewiring:
    """dummy class for protecting these attributes from batch collation or movement to CUDA"""
    pass


class Community(Benchmark):
    NUM_GRAPHS = 50
    TEST_RATIO = 0.1
    EXPLANATION_SAMPLE_PER_GRAPH = 200

    def __init__(self, sample_count, num_layers, concat_features, conv_type):
        super().__init__(sample_count, num_layers, concat_features, conv_type)
        mlflow.log_param('EXPLANATION_SAMPLE_PER_GRAPH', self.EXPLANATION_SAMPLE_PER_GRAPH)

    def create_dataset(self):
        K = 10
        SZ = 100
        N = SZ * K
        P = 0.15
        Q = 0.04
        PERTURB_COUNT = 500
        REWIRE_COUNT = 50
        mlflow.log_param('P', P)
        mlflow.log_param('Q', Q)
        mlflow.log_param('N', N)
        mlflow.log_param('K', K)
        mlflow.log_param('PERTURB_COUNT', PERTURB_COUNT)
        mlflow.log_param('REWIRE_COUNT', REWIRE_COUNT)
        sizes = [SZ] * K

        probs = np.ones((K, K)) * Q
        probs += np.eye(K) * (P - Q)
        g = nx.stochastic_block_model(sizes, probs)

        colors = []
        for i in range(K):
            colors.extend([i] * SZ)
        labels = colors.copy()

        for i in random.sample(list(range(N)), PERTURB_COUNT):
            choices = list(range(10))
            # choices.remove(colors[i])
            colors[i] = random.choice(choices)
            # colors[i]=0

        features = np.zeros((len(g.nodes()), 10))
        for i, c in enumerate(colors):
            features[i, c] = 1

        data = from_networkx(g)
        data.x = torch.tensor(features, dtype=torch.float)
        data.y = torch.tensor(labels)

        edge_to_id = {}
        id_to_edge = {}
        bad_edges = defaultdict(list)
        for eid in range(data.num_edges):
            u, v = data.edge_index[:, eid]
            u, v = u.item(), v.item()
            edge_to_id[(u, v)] = eid
            id_to_edge[eid] = (u, v)
            if labels[u] != labels[v]:
                bad_edges[labels[u]].append((u, v))

        node_edits = {}
        for i in g.nodes():
            rewires = random.sample(bad_edges[labels[i]], REWIRE_COUNT)
            nodes_with_same_label = [x for x in g.nodes() if labels[x] == labels[i]]
            new_edges = set()
            edits = {}
            for u, v in rewires:
                assert labels[u] == labels[i]
                v2 = random.choice(nodes_with_same_label)
                while v2 == u or (u, v2) in edge_to_id or (u, v2) in new_edges:
                    v2 = random.choice(nodes_with_same_label)

                edits[edge_to_id[(u, v)]] = (u, v2)
                edits[edge_to_id[(v, u)]] = (v2, u)
                new_edges.add((u, v2))
            assert len(edits) == 100
            node_edits[i] = edits

        rewiring = Rewiring()
        rewiring.id_to_edge = id_to_edge
        rewiring.edge_to_id = edge_to_id
        rewiring.node_edits = node_edits
        data.rewiring = rewiring
        data.num_classes = len(set(labels))

        return data

    def subsample_nodes(self, explain_function, nodes):
        if explain_function != explain_pgmexplainer:
            return random.sample(nodes, self.EXPLANATION_SAMPLE_PER_GRAPH)
        return random.sample(nodes, self.PGMEXPLAINER_SUBSAMPLE_PER_GRAPH)

    def evaluate_explanation(self, explain_function, model, test_dataset):
        accs = []
        for dss in test_dataset:
            bads = 0
            before_afters = []
            depth_limit = len(model.convs)
            tests = 0
            nodes_to_test = list(range(1000))
            nodes_to_test = self.subsample_nodes(explain_function, nodes_to_test)
            pbar = tqdm(nodes_to_test)
            model_cache = model(dss.x, dss.edge_index)
            edge_index_rewired = dss.edge_index.clone().to(self.device)
            rewire_mask = torch.zeros(dss.num_edges, dtype=bool)
            for node_idx in pbar:
                prob, label = model_cache[[node_idx]].softmax(dim=1).max(dim=1)
                for eid, (u, v) in dss.rewiring.node_edits[node_idx].items():
                    edge_index_rewired[:, eid] = torch.tensor([u, v], dtype=int)
                    rewire_mask[eid] = True

                prob_rewired, label_rewired = model(dss.x, edge_index_rewired)[[node_idx]].softmax(dim=1).max(dim=1)
                target = dss.y[node_idx].item()
                if label_rewired.item() == target and prob_rewired.item() > prob.item():
                    final_mask = (k_hop_subgraph(node_idx, depth_limit, dss.edge_index)[3] &
                                  k_hop_subgraph(node_idx, depth_limit, edge_index_rewired)[3])

                    final_mask = final_mask.cpu() & rewire_mask
                    attribution = explain_function(model, node_idx, dss.x, dss.edge_index, target, final_mask)[
                        final_mask]
                    attribution_rewired = \
                        explain_function(model, node_idx, dss.x, edge_index_rewired, target, final_mask)[final_mask]

                    before_afters.append((attribution.mean(), attribution_rewired.mean()))
                    if attribution.mean() > attribution_rewired.mean():
                        bads += 1
                        # print(node_idx)

                    tests += 1
                    pbar.set_postfix(bads=bads / (tests), tests=tests)
                # revert to original
                for eid in dss.rewiring.node_edits[node_idx]:
                    edge_index_rewired[:, eid] = torch.tensor(dss.rewiring.id_to_edge[eid], dtype=int)
                    rewire_mask[eid] = False
            accs.append((1 - bads / tests))
        return accs
