import random

import mlflow
import networkx as nx
import numpy as np
import typer
from captum.attr import Saliency, IntegratedGradients
from torch_geometric.data import Data
from torch_geometric.utils import to_networkx
from tqdm import tqdm as tq

from models_node import *


def infection_dataset(max_dist=4):
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
    ambig_nodes = []
    ambig_exps = []
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
        else:
            ambig_nodes.append(u)
            ambig_exps.append(all_shortest_paths)
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

    ambig_explanations = []
    for exp in ambig_exps:
        all_eids = []
        for sp in exp:
            all_eids.append(to_eids(sp))
        ambig_explanations.append(all_eids)

    return g, features, labels, list(test_nodes), explanations, ambig_nodes, ambig_explanations


def make_coo_edges_by_id(g):
    edges = []
    for u, v, d in sorted(list(g.edges(data=True)), key=lambda x: x[2]['id']):
        edges.append([u, v])
    return torch.tensor(edges, dtype=torch.long).t().contiguous()


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


def explain_occlusion(model, node_idx, data, target, depth_limit=4):
    p, label = model(data.x, data.edge_index)[node_idx].max(dim=0)
    pred_prob, pred_label = p.item(), label.item()
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
            prob = model(data.x, data.edge_index[:, edge_occlusion_mask])[node_idx][pred_label].item()
            edge_mask[i] = pred_prob - prob
            edge_occlusion_mask[i] = True
    return edge_mask / (np.abs(edge_mask).max())


def explain_gnnexplainer(model, node_idx, data, target):
    explainer = GNNExplainer(model, epochs=200, log=False)
    node_feat_mask, edge_mask = explainer.explain_node(node_idx, data.x, data.edge_index)
    return edge_mask.numpy()


def get_accuracy(correct_ids, edge_mask):
    correct_count = 0
    for x in np.argsort(-edge_mask)[:len(correct_ids)]:
        if x in correct_ids:
            correct_count += 1
    return correct_count / len(correct_ids)


def main(sample_count: int = typer.Option(100, help='How many times to retry the whole experiment'),
         num_layers: int = typer.Option(3, help='Number of layers in the GNN model'),
         concat_features: bool = typer.Option(True,
                                              help='Concat embeddings of each convolutional layer for final fc layers'),
         conv_type: str = typer.Option('GCNConv',
                                       help="Convolution class. Can be GCNConv or GraphConv"),
         explain_method: str = typer.Option('sa',
                                            help="Explanation method to use. Can be ['sa','ig','gnnexplainer','occlusion'] "),
         ):
    explain_function = eval('explain_' + explain_method)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    mlflow.set_experiment('Infection Dataset')
    arguments = {
        'sample_count': sample_count,
        'explain_method': explain_method,
        'num_layers': num_layers,
        'concat_features': concat_features,
        'conv_type': conv_type
    }
    mlflow.log_params(arguments)
    print(f"Using device {device}")
    rolling_explain = []
    rolling_model_train = []
    rolling_model_test = []
    for experiment_i in tq(range(sample_count)):
        g, features, labels, test_nodes, explanations, ambig_nodes, ambig_explanations = infection_dataset()
        edge_index = make_coo_edges_by_id(g)
        data = Data(x=torch.tensor(features, dtype=torch.float),
                    edge_index=edge_index,
                    y=torch.tensor(labels))
        data.num_classes = max(labels) + 1
        data.train_mask = np.ones(data.num_nodes, dtype=bool)
        data.train_mask[list(test_nodes)] = False
        data.test_mask = ~data.train_mask
        model = Net1(data.num_node_features, data.num_classes, num_layers, concat_features, conv_type).to(device)
        data = data.to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)

        model.train()
        pbar = tq(range(200), disable=True)
        for epoch in pbar:
            optimizer.zero_grad()
            out = model(data.x, data.edge_index)
            _, pred = out.max(dim=1)
            correct = int(pred[data.train_mask].eq(data.y[data.train_mask]).sum().item())
            train_acc = correct / int(data.train_mask.sum())
            pbar.set_postfix(acc=train_acc)
            loss = F.nll_loss(out[data.train_mask], data.y[data.train_mask])
            loss.backward()
            optimizer.step()
        model.eval()
        _, pred = model(data.x, data.edge_index).max(dim=1)
        correct = int(pred[data.test_mask].eq(data.y[data.test_mask]).sum().item())
        test_acc = correct / int(data.test_mask.sum())
        print('Accuracy: {:.4f}'.format(test_acc))

        accs = []
        pbar = tq(list(zip(test_nodes, explanations)), disable=True)
        misclassify_count = 0
        for node_idx, correct_ids in pbar:
            if len(correct_ids) == 0:
                continue
            if pred[node_idx] != data.y[node_idx]:
                misclassify_count += 1
                continue
            edge_mask = explain_function(model, node_idx, data, labels[node_idx])
            explain_acc = get_accuracy(correct_ids, edge_mask)
            #     if acc<1:
            #         print(node_idx, acc)
            accs.append(explain_acc)
            pbar.set_postfix(acc=np.mean(accs))
        print(np.mean(accs), np.std(accs))
        rolling_explain.append(np.mean(accs))
        rolling_model_train.append(train_acc)
        rolling_model_test.append(test_acc)
        metrics = {
            'misclassify_count': misclassify_count,
            'train_acc': train_acc,
            'test_acc': test_acc,
            'explain_acc': np.mean(accs),
            'rolling_avg': np.mean(rolling_explain),
            'rolling_std': np.std(rolling_explain),
            'rolling_model_train_avg': np.mean(rolling_model_train),
            'rolling_model_train_std': np.std(rolling_model_train),
            'rolling_model_test_avg': np.mean(rolling_model_test),
            'rolling_model_test_std': np.std(rolling_model_test),
        }
        mlflow.log_metrics(metrics, step=experiment_i)


if __name__ == "__main__":
    typer.run(main)
