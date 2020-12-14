from collections import defaultdict

import mlflow
import numpy as np
import typer
from torch_geometric.data import Data, DataLoader
from tqdm import tqdm as tq

from infection import infection_dataset
from models_node import *
from explain_methods import *

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def make_coo_edges_by_id(g):
    edges = []
    for u, v, d in sorted(list(g.edges(data=True)), key=lambda x: x[2]['id']):
        edges.append([u, v])
    return torch.tensor(edges, dtype=torch.long).t().contiguous()


def get_accuracy(correct_ids, edge_mask, edge_index):
    correct_count = 0
    correct_edges = list(zip(correct_ids[1:], correct_ids))

    for x in np.argsort(-edge_mask)[:len(correct_ids)]:
        u, v = edge_index[:, x]
        u, v = u.item(), v.item()
        if (u, v) in correct_edges:
            correct_count += 1
    return correct_count / len(correct_edges)


def get_accuracy_undirected(correct_ids, edge_values):
    correct_count = 0
    correct_edges = list(zip(correct_ids[1:], correct_ids))

    top_edges = list(sorted([(-value, edge) for edge, value in edge_values.items()]))[:len(correct_ids)]
    for _, (u, v) in top_edges:
        if (u, v) in correct_edges or (v, u) in correct_edges:
            correct_count += 1
    return correct_count / len(correct_edges)


def train(model, optimizer, train_loader):
    model.train()
    loss_all = 0
    for data in train_loader:
        data = data.to(device)
        optimizer.zero_grad()
        output = model(data.x, data.edge_index)
        loss = F.nll_loss(output, data.y)
        loss.backward()
        loss_all += loss.item() * data.num_graphs
        optimizer.step()
    return loss_all / len(train_loader.dataset)


def test(model, loader):
    model.eval()

    correct = 0
    total = 0
    for data in loader:
        data = data.to(device)
        output = model(data.x, data.edge_index)
        pred = output.max(dim=1)[1]
        correct += pred.eq(data.y).sum().item()
        total += len(data.y)
    return correct / total


def train_and_test(model, train_loader, test_loader):
    optimizer = torch.optim.Adam(model.parameters(), lr=0.003, weight_decay=5e-4)
    pbar = tq(range(1, 201))
    for epoch in pbar:
        train_loss = train(model, optimizer, train_loader)
        train_acc = test(model, train_loader)
        test_acc = test(model, test_loader)
        pbar.set_postfix(train_loss=train_loss, train_acc=train_acc, test_acc=test_acc)
    return train_acc, test_acc


def main(sample_count: int = typer.Option(10, help='How many times to retry the whole experiment'),
         num_layers: int = typer.Option(4, help='Number of layers in the GNN model'),
         concat_features: bool = typer.Option(True,
                                              help='Concat embeddings of each convolutional layer for final fc layers'),
         conv_type: str = typer.Option('GraphConv',
                                       help="Convolution class. Can be GCNConv or GraphConv"),
         explain_method: str = typer.Option('sa',
                                            help="Explanation method to use. Can be ['sa','ig','gnnexplainer','occlusion'] "),
         ):
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
    rolling_explain = defaultdict(list)
    rolling_model_train = []
    rolling_model_test = []
    NUM_GRAPHS = 10
    TEST_RATIO = 0.4
    mlflow.log_param('num_graphs', NUM_GRAPHS)
    mlflow.log_param('test_ratio', TEST_RATIO)
    for experiment_i in tq(range(sample_count)):
        dataset = [infection_dataset() for i in range(NUM_GRAPHS)]
        split_point = int(len(dataset) * TEST_RATIO)
        test_dataset = dataset[:split_point]
        train_dataset = dataset[split_point:]
        test_loader = DataLoader(test_dataset, batch_size=1)
        train_loader = DataLoader(train_dataset, batch_size=1)
        data = dataset[0]
        model = Net1(data.num_node_features, data.num_classes, num_layers, concat_features, conv_type).to(device)
        train_acc, test_acc = train_and_test(model, train_loader, test_loader)
        model.eval()
        rolling_model_train.append(train_acc)
        rolling_model_test.append(test_acc)
        metrics = {
            'train_acc': train_acc,
            'test_acc': test_acc,
            'rolling_model_train_avg': np.mean(rolling_model_train),
            'rolling_model_train_std': np.std(rolling_model_train),
            'rolling_model_test_avg': np.mean(rolling_model_test),
            'rolling_model_test_std': np.std(rolling_model_test),
        }
        mlflow.log_metrics(metrics, step=experiment_i)

        for explain_name in ['occlusion','occlusion_undirected','gnnexplainer','sa', 'ig']:
            explain_function = eval('explain_' + explain_name)
            accs, misclassify_count = evaluate_explanation(explain_function, model, test_dataset)
            print(explain_name, np.mean(accs), np.std(accs))
            rolling_explain[explain_name].append(np.mean(accs))
            metrics = {
                f'explain_{explain_name}_acc': np.mean(accs),
                f'rolling_{explain_name}_avg': np.mean(rolling_explain[explain_name]),
                f'rolling_{explain_name}_std': np.std(rolling_explain[explain_name]),
            }
            mlflow.log_metrics(metrics, step=experiment_i)


def aggregate_directions(edge_mask, edge_index):
    edge_values = defaultdict(float)
    for x in range(len(edge_mask)):
        u, v = edge_index[:, x]
        u, v = u.item(), v.item()
        if u > v:
            u, v = v, u
        edge_values[(u, v)] += edge_mask[x]
    return edge_values


def evaluate_explanation(explain_function, model, test_dataset):
    accs = []
    misclassify_count = 0
    for data in test_dataset:
        _, pred = model(data.x, data.edge_index).max(dim=1)
        pbar = tq(list(zip(data.unique_solution_nodes, data.unique_solution_explanations)), disable=False)
        for node_idx, correct_ids in pbar:
            if len(correct_ids) == 0:
                continue
            if pred[node_idx] != data.y[node_idx]:
                misclassify_count += 1
                continue
            edge_mask = explain_function(model, node_idx, data, data.y[node_idx].item())
            edge_values = aggregate_directions(edge_mask, data.edge_index)
            explain_acc = get_accuracy_undirected(correct_ids, edge_values)
            accs.append(explain_acc)
            pbar.set_postfix(acc=np.mean(accs))
    return accs, misclassify_count


if __name__ == "__main__":
    typer.run(main)
