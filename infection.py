import random

import networkx as nx
import numpy as np


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
    return g, features, labels, list(test_nodes), explanations