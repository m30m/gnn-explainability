import mlflow
import numpy as np
import typer
from torch_geometric.data import Data
from tqdm import tqdm as tq

from infection import infection_dataset
from models_node import *


def make_coo_edges_by_id(g):
    edges = []
    for u, v, d in sorted(list(g.edges(data=True)), key=lambda x: x[2]['id']):
        edges.append([u, v])
    return torch.tensor(edges, dtype=torch.long).t().contiguous()


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
        g, features, labels, test_nodes, explanations = infection_dataset()
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
