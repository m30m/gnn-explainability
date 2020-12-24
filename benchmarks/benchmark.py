from collections import defaultdict

import mlflow
import torch.nn.functional as F
from tqdm import tqdm as tq

from explain_methods import *
from models_node import Net1


class Benchmark(object):
    NUM_GRAPHS = 2
    TEST_RATIO = 0.5
    METHODS = ['distance', 'gradcam', 'gradXact', 'random', 'sa_node', 'ig_node', 'sa', 'ig',
               'occlusion_undirected', 'gnnexplainer']

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
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def create_dataset(self):
        raise NotImplementedError

    def evaluate_explanation(self, explain_function, model, test_dataset):
        raise NotImplementedError

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
        rolling_explain = defaultdict(list)
        rolling_model_train = []
        rolling_model_test = []
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

            for explain_name in self.METHODS:
                explain_function = eval('explain_' + explain_name)
                accs = self.evaluate_explanation(explain_function, model, test_dataset)
                print(explain_name, np.mean(accs), np.std(accs))
                rolling_explain[explain_name].append(np.mean(accs))
                metrics = {
                    f'explain_{explain_name}_acc': np.mean(accs),
                    f'explain_{explain_name}_acc_std': np.std(accs),
                    f'rolling_{explain_name}_avg': np.mean(rolling_explain[explain_name]),
                    f'rolling_{explain_name}_std': np.std(rolling_explain[explain_name]),
                }
                mlflow.log_metrics(metrics, step=experiment_i)
