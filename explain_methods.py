from typing import Union, Tuple, Any

import networkx as nx
import numpy as np
import torch
import torch.nn.functional as F
from captum._utils.common import (
    _format_additional_forward_args,
    _format_input,
    _format_output,
)
from captum._utils.gradient import (
    apply_gradient_requirements,
    compute_layer_gradients_and_eval,
    undo_gradient_requirements,
)
from captum._utils.typing import TargetType
from captum.attr import Saliency, IntegratedGradients, LayerGradCam
from torch import Tensor
from torch_geometric.data import Data
from torch_geometric.nn import GNNExplainer, MessagePassing
from torch_geometric.utils import to_networkx

from pgm_explainer import Node_Explainer

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class GraphLayerGradCam(LayerGradCam):

    def attribute(self, inputs: Union[Tensor, Tuple[Tensor, ...]], target: TargetType = None,
                  additional_forward_args: Any = None, attribute_to_layer_input: bool = False,
                  relu_attributions: bool = False) -> Union[Tensor, Tuple[Tensor, ...]]:
        inputs = _format_input(inputs)
        additional_forward_args = _format_additional_forward_args(
            additional_forward_args
        )
        gradient_mask = apply_gradient_requirements(inputs)
        # Returns gradient of output with respect to
        # hidden layer and hidden layer evaluated at each input.
        layer_gradients, layer_evals = compute_layer_gradients_and_eval(
            self.forward_func,
            self.layer,
            inputs,
            target,
            additional_forward_args,
            device_ids=self.device_ids,
            attribute_to_layer_input=attribute_to_layer_input,
        )
        undo_gradient_requirements(inputs, gradient_mask)

        summed_grads = tuple(
            torch.mean(
                layer_grad,
                dim=0,
                keepdim=True,
            )
            for layer_grad in layer_gradients
        )

        scaled_acts = tuple(
            torch.sum(summed_grad * layer_eval, dim=1, keepdim=True)
            for summed_grad, layer_eval in zip(summed_grads, layer_evals)
        )
        if relu_attributions:
            scaled_acts = tuple(F.relu(scaled_act) for scaled_act in scaled_acts)
        return _format_output(len(scaled_acts) > 1, scaled_acts)


def model_forward(edge_mask, model, node_idx, x, edge_index):
    out = model(x, edge_index, edge_mask)
    return out[[node_idx]]


def model_forward_node(x, model, edge_index, node_idx):
    out = model(x, edge_index)
    return out[[node_idx]]


def node_attr_to_edge(edge_index, node_mask):
    edge_mask = np.zeros(edge_index.shape[1])
    edge_mask += node_mask[edge_index[0].cpu().numpy()]
    edge_mask += node_mask[edge_index[1].cpu().numpy()]
    return edge_mask


def get_last_convolution_layer(model):
    last_layer = None
    for module in model.modules():
        if isinstance(module, MessagePassing):
            last_layer = module
    return last_layer


def explain_random(model, node_idx, x, edge_index, target, include_edges=None):
    return np.random.uniform(size=edge_index.shape[1])


def explain_by_last_layer(cls, model, node_idx, x, edge_index, target, include_edges=None):
    input_mask = x.clone().requires_grad_(True).to(device)
    last_layer = get_last_convolution_layer(model)
    # Captum default implementation of LayerGradCam does not average over nodes for different channels because of
    # different assumptions on tensor shapes
    layer_gc = cls(model_forward_node, last_layer)
    node_attr = layer_gc.attribute(input_mask, target=target, additional_forward_args=(model, edge_index, node_idx))
    node_attr = node_attr.cpu().detach().numpy().ravel()
    edge_mask = node_attr_to_edge(edge_index, node_attr)
    return edge_mask


def explain_gradXact(model, node_idx, x, edge_index, target, include_edges=None):
    # Captum default implementation of LayerGradCam does not average over nodes for different channels because of
    # different assumptions on tensor shapes
    return explain_by_last_layer(LayerGradCam, model, node_idx, x, edge_index, target, include_edges)


def explain_gradcam(model, node_idx, x, edge_index, target, include_edges=None):
    return explain_by_last_layer(GraphLayerGradCam, model, node_idx, x, edge_index, target, include_edges)


def explain_distance(model, node_idx, x, edge_index, target, include_edges=None):
    data = Data(x=x, edge_index=edge_index)
    g = to_networkx(data)
    length = nx.shortest_path_length(g, source=node_idx)

    def get_attr(node):
        if node in length:
            return 1 / (length[node] + 1)
        return 0

    edge_sources = edge_index[0].cpu().numpy()
    return np.array([get_attr(node) for node in edge_sources])


def explain_sa_node(model, node_idx, x, edge_index, target, include_edges=None):
    saliency = Saliency(model_forward_node)
    input_mask = x.clone().requires_grad_(True).to(device)
    saliency_mask = saliency.attribute(input_mask, target=target, additional_forward_args=(model, edge_index, node_idx),
                                       abs=False)

    node_attr = saliency_mask.cpu().numpy().sum(axis=1)
    edge_mask = node_attr_to_edge(edge_index, node_attr)
    return edge_mask


def explain_sa(model, node_idx, x, edge_index, target, include_edges=None):
    saliency = Saliency(model_forward)
    input_mask = torch.ones(edge_index.shape[1]).requires_grad_(True).to(device)
    saliency_mask = saliency.attribute(input_mask, target=target,
                                       additional_forward_args=(model, node_idx, x, edge_index))

    edge_mask = saliency_mask.cpu().numpy()
    return edge_mask


def explain_ig_node(model, node_idx, x, edge_index, target, include_edges=None):
    ig = IntegratedGradients(model_forward_node)
    input_mask = x.clone().requires_grad_(True).to(device)
    ig_mask = ig.attribute(input_mask, target=target, additional_forward_args=(model, edge_index, node_idx),
                           internal_batch_size=input_mask.shape[0])

    node_attr = ig_mask.cpu().detach().numpy().sum(axis=1)
    edge_mask = node_attr_to_edge(edge_index, node_attr)
    return edge_mask


def explain_ig(model, node_idx, x, edge_index, target, include_edges=None):
    ig = IntegratedGradients(model_forward)
    input_mask = torch.ones(edge_index.shape[1]).requires_grad_(True).to(device)
    ig_mask = ig.attribute(input_mask, target=target, additional_forward_args=(model, node_idx, x, edge_index),
                           internal_batch_size=edge_index.shape[1])

    edge_mask = ig_mask.cpu().detach().numpy()
    return edge_mask


def explain_occlusion(model, node_idx, x, edge_index, target, include_edges=None):
    depth_limit = len(model.convs) + 1
    data = Data(x=x, edge_index=edge_index)
    pred_prob = model(data.x, data.edge_index)[node_idx][target].item()
    g = to_networkx(data)
    subgraph_nodes = []
    for k, v in nx.shortest_path_length(g, node_idx).items():
        if v < depth_limit:
            subgraph_nodes.append(k)
    subgraph = g.subgraph(subgraph_nodes)
    edge_occlusion_mask = np.ones(data.num_edges, dtype=bool)
    edge_mask = np.zeros(data.num_edges)
    edge_index_numpy = data.edge_index.cpu().numpy()
    for i in range(data.num_edges):
        if include_edges is not None and not include_edges[i].item():
            continue
        u, v = list(edge_index_numpy[:, i])
        if (u, v) in subgraph.edges():
            edge_occlusion_mask[i] = False
            prob = model(data.x, data.edge_index[:, edge_occlusion_mask])[node_idx][target].item()
            edge_mask[i] = pred_prob - prob
            edge_occlusion_mask[i] = True
    return edge_mask


def explain_occlusion_undirected(model, node_idx, x, edge_index, target, include_edges=None):
    depth_limit = len(model.convs) + 1
    data = Data(x=x, edge_index=edge_index)
    pred_prob = model(data.x, data.edge_index)[node_idx][target].item()
    g = to_networkx(data)
    subgraph_nodes = []
    for k, v in nx.shortest_path_length(g, node_idx).items():
        if v < depth_limit:
            subgraph_nodes.append(k)
    subgraph = g.subgraph(subgraph_nodes)
    edge_occlusion_mask = np.ones(data.num_edges, dtype=bool)
    edge_mask = np.zeros(data.num_edges)
    reverse_edge_map = {}
    edge_index_numpy = data.edge_index.cpu().numpy()
    for i in range(data.num_edges):
        u, v = list(edge_index_numpy[:, i])
        reverse_edge_map[(u, v)] = i

    for (u, v) in subgraph.edges():
        if u > v:  # process each edge once
            continue
        i1 = reverse_edge_map[(u, v)]
        i2 = reverse_edge_map[(v, u)]
        if include_edges is not None and not include_edges[i1].item() and not include_edges[i2].item():
            continue
        edge_occlusion_mask[[i1, i2]] = False
        prob = model(data.x, data.edge_index[:, edge_occlusion_mask])[node_idx][target].item()
        edge_mask[[i1, i2]] = pred_prob - prob
        edge_occlusion_mask[[i1, i2]] = True
    return edge_mask


def explain_gnnexplainer(model, node_idx, x, edge_index, target, include_edges=None):
    explainer = GNNExplainer(model, epochs=200, log=False)
    node_feat_mask, edge_mask = explainer.explain_node(node_idx, x, edge_index)
    return edge_mask.cpu().numpy()


def explain_pgmexplainer(model, node_idx, x, edge_index, target, include_edges=None):
    explainer = Node_Explainer(model, edge_index, x, len(model.convs), print_result=0)
    explanation = explainer.explain(node_idx,target)
    node_attr = np.zeros(x.shape[0])
    for node, p_value in explanation.items():
        node_attr[node] = 1 - p_value
    edge_mask = node_attr_to_edge(edge_index, node_attr)
    return edge_mask
