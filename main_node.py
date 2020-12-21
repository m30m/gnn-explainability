from collections import defaultdict
from enum import Enum

import mlflow
import numpy as np
import typer
from torch_geometric.data import Data, DataLoader
from tqdm import tqdm as tq

import infection
import community
from models_node import *
from explain_methods import *

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def train(model, optimizer, train_loader):
    model.train()
    loss_all = 0
    for data in train_loader:
        data = data.to(device)
        optimizer.zero_grad()
        output = model(data.x, data.edge_index)
        loss = F.nll_loss(output, data.y)
        loss.backward()
        loss_all += loss.item()
        optimizer.step()
    return loss_all / len(train_loader)


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
    weight_decay = 0
    lr = 0.005
    epochs = 200
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    mlflow.log_param('weight_decay', weight_decay)
    mlflow.log_param('lr', lr)
    mlflow.log_param('epochs', epochs)
    pbar = tq(range(epochs))
    for epoch in pbar:
        train_loss = train(model, optimizer, train_loader)
        train_acc = test(model, train_loader)
        test_acc = test(model, test_loader)
        pbar.set_postfix(train_loss=train_loss, train_acc=train_acc, test_acc=test_acc)
    return train_acc, test_acc


class Experiment(str, Enum):
    infection = "infection"
    community = "community"


def main(experiment: Experiment = typer.Argument(..., help="Dataset to use"),
         sample_count: int = typer.Option(10, help='How many times to retry the whole experiment'),
         num_layers: int = typer.Option(4, help='Number of layers in the GNN model'),
         concat_features: bool = typer.Option(True,
                                              help='Concat embeddings of each convolutional layer for final fc layers'),
         conv_type: str = typer.Option('GraphConv',
                                       help="Convolution class. Can be GCNConv or GraphConv"),
         explain_method: str = typer.Option('sa',
                                            help="Explanation method to use. Can be ['sa','ig','gnnexplainer','occlusion'] "),
         ):
    mlflow.set_experiment(experiment.value)
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
        if experiment == Experiment.infection:
            make_data = lambda: infection.make_data(max_dist=num_layers)
        if experiment == Experiment.community:
            make_data = community.make_data
        dataset = [make_data() for i in range(NUM_GRAPHS)]
        split_point = int(len(dataset) * TEST_RATIO)
        test_dataset = dataset[:split_point]
        train_dataset = dataset[split_point:]
        data = dataset[0]
        model = Net1(data.num_node_features, data.num_classes, num_layers, concat_features, conv_type).to(device)
        train_acc, test_acc = train_and_test(model, train_dataset, test_dataset)
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

        for explain_name in ['gnnexplainer','sa','occlusion','occlusion_undirected', 'ig']:
            explain_function = eval('explain_' + explain_name)
            accs = evaluate_explanation(explain_function, model, test_dataset, experiment)
            print(explain_name, np.mean(accs), np.std(accs))
            rolling_explain[explain_name].append(np.mean(accs))
            metrics = {
                f'explain_{explain_name}_acc': np.mean(accs),
                f'rolling_{explain_name}_avg': np.mean(rolling_explain[explain_name]),
                f'rolling_{explain_name}_std': np.std(rolling_explain[explain_name]),
            }
            mlflow.log_metrics(metrics, step=experiment_i)


def evaluate_explanation(explain_function, model, test_dataset, experiment):
    if experiment == Experiment.infection:
        return infection.evaluate_explanation(explain_function, model, test_dataset)
    if experiment == Experiment.community:
        return community.evaluate_explanation(explain_function, model, test_dataset)


if __name__ == "__main__":
    typer.run(main)
