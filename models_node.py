import torch
import torch.nn.functional as F
from torch.nn import Sequential, Linear, ReLU
from torch_geometric.nn import GNNExplainer, GINConv, MessagePassing, GCNConv, GraphConv


class Net1(torch.nn.Module):
    def __init__(self, num_node_features, num_classes, num_layers, concat_features, conv_type):
        super(Net1, self).__init__()
        dim = 32
        self.convs = torch.nn.ModuleList()
        if conv_type == 'GCNConv':
            conv_class = GCNConv
            kwargs = {'add_self_loops': False}
        elif conv_type == 'GraphConv':
            conv_class = GraphConv
            kwargs = {}
        else:
            raise RuntimeError(f"conv_type {conv_type} not supported")

        self.convs.append(conv_class(num_node_features, dim, **kwargs))
        for i in range(num_layers - 1):
            self.convs.append(conv_class(dim, dim, **kwargs))
        self.concat_features = concat_features
        if concat_features:
            self.fc = Linear(dim * num_layers + num_node_features, num_classes)
        else:
            self.fc = Linear(dim, num_classes)

    def forward(self, x, edge_index, edge_weight=None):
        xs = [x]
        for conv in self.convs:
            x = conv(x, edge_index, edge_weight)
            x = F.relu(x)
            xs.append(x)
        if self.concat_features:
            x = torch.cat(xs, dim=1)
        x = self.fc(x)
        return F.log_softmax(x, dim=1)
