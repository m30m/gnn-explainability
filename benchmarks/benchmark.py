import json
import os
import tempfile
import time
from collections import defaultdict
import random

import mlflow
import torch.nn.functional as F
from tqdm import tqdm as tq

from explain_methods import *
from models_node import Net1


class Benchmark(object):
    NUM_GRAPHS = 2
    TEST_RATIO = 0.5
    PGMEXPLAINER_SUBSAMPLE_PER_GRAPH = 20
    METHODS = ['pagerank', 'pgmexplainer', 'occlusion_undirected', 'distance', 'gradcam', 'gradXact', 'random', 'sa_node',
               'ig_node', 'sa', 'ig', 'gnnexplainer']

    def __init__(self, sample_count, num_layers, concat_features, conv_type):
        arguments = {
            'sample_count': sample_count,
            'num_layers': num_layers,
            'concat_features': concat_features,
            'conv_type': conv_type,
            'num_graphs': self.NUM_GRAPHS,
            'test_ratio': self.TEST_RATIO,
        }
        self.sample_count = sample_count
        self.num_layers = num_layers
        self.concat_features = concat_features
        self.conv_type = conv_type
        mlflow.log_params(arguments)
        mlflow.log_param('PGMEXPLAINER_SUBSAMPLE_PER_GRAPH', self.PGMEXPLAINER_SUBSAMPLE_PER_GRAPH)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def create_dataset(self):
        raise NotImplementedError

    def evaluate_explanation(self, explain_function, model, test_dataset):
        raise NotImplementedError

    def subsample_nodes(self, explain_function, nodes):
        if explain_function.explain_function != explain_pgmexplainer:
            return nodes
        return random.sample(nodes, self.PGMEXPLAINER_SUBSAMPLE_PER_GRAPH)

    @staticmethod
    def aggregate_directions(edge_mask, edge_index):
        edge_values = defaultdict(float)
        for x in range(len(edge_mask)):
            u, v = edge_index[:, x]
            u, v = u.item(), v.item()
            if u > v:
                u, v = v, u
            edge_values[(u, v)] += edge_mask[x]
        return edge_values

    def train(self, model, optimizer, train_loader):
        model.train()
        loss_all = 0
        for data in train_loader:
            data = data.to(self.device)
            optimizer.zero_grad()
            output = model(data.x, data.edge_index)
            loss = F.nll_loss(output, data.y)
            loss.backward()
            loss_all += loss.item()
            optimizer.step()
        return loss_all / len(train_loader)

    def test(self, model, loader):
        model.eval()

        correct = 0
        total = 0
        for data in loader:
            data = data.to(self.device)
            output = model(data.x, data.edge_index)
            pred = output.max(dim=1)[1]
            correct += pred.eq(data.y).sum().item()
            total += len(data.y)
        return correct / total

    def train_and_test(self, model, train_loader, test_loader):
        weight_decay = 0
        lr = 0.005
        epochs = 200
        optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
        mlflow.log_param('weight_decay', weight_decay)
        mlflow.log_param('lr', lr)
        mlflow.log_param('epochs', epochs)
        pbar = tq(range(epochs))
        for epoch in pbar:
            train_loss = self.train(model, optimizer, train_loader)
            train_acc = self.test(model, train_loader)
            test_acc = self.test(model, test_loader)
            pbar.set_postfix(train_loss=train_loss, train_acc=train_acc, test_acc=test_acc)
        return train_acc, test_acc

    def run(self):
        print(f"Using device {self.device}")
        all_explanations = defaultdict(list)
        for experiment_i in tq(range(self.sample_count)):
            dataset = [self.create_dataset() for i in range(self.NUM_GRAPHS)]
            split_point = int(len(dataset) * self.TEST_RATIO)
            test_dataset = dataset[:split_point]
            train_dataset = dataset[split_point:]
            data = dataset[0]
            model = Net1(data.num_node_features, data.num_classes, self.num_layers, self.concat_features,
                         self.conv_type).to(
                self.device)
            train_acc, test_acc = self.train_and_test(model, train_dataset, test_dataset)
            model.eval()
            metrics = {
                'train_acc': train_acc,
                'test_acc': test_acc,
            }
            mlflow.log_metrics(metrics, step=experiment_i)

            for explain_name in self.METHODS:
                explain_function = eval('explain_' + explain_name)
                duration_samples = []

                def time_wrapper(*args, **kwargs):
                    start_time = time.time()
                    result = explain_function(*args, **kwargs)
                    end_time = time.time()
                    duration_seconds = end_time - start_time
                    duration_samples.append(duration_seconds)
                    return result

                time_wrapper.explain_function = explain_function
                accs = self.evaluate_explanation(time_wrapper, model, test_dataset)
                print(explain_name, np.mean(accs), np.std(accs))
                all_explanations[explain_name].append(list(accs))
                metrics = {
                    f'explain_{explain_name}_acc': np.mean(accs),
                    f'explain_{explain_name}_acc_std': np.std(accs),
                    f'time_{explain_name}_s_avg': np.mean(duration_samples),
                    f'time_{explain_name}_s_std': np.std(duration_samples),
                }
                with tempfile.TemporaryDirectory() as tmpdir:
                    file_path = os.path.join(tmpdir, 'accuracies.json')
                    json.dump(all_explanations, open(file_path, 'w'), indent=2)
                    mlflow.log_artifact(file_path)
                mlflow.log_metrics(metrics, step=experiment_i)
