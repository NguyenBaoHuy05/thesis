"""
Graph utilities and topology definitions for Skeleton GCNs
==========================================================
Supports NTU RGB+D (25 joints) and NW-UCLA skeleton topologies.
Computes spatial adjacency matrices with customizable partitioning strategies.
"""

import numpy as np


def edge2mat(link, num_node):
    A = np.zeros((num_node, num_node))
    for i, j in link:
        A[j, i] = 1
    return A


def normalize_digraph(A):
    Dl = np.sum(A, 0)
    num_node = A.shape[0]
    Dn = np.zeros((num_node, num_node))
    for i in range(num_node):
        if Dl[i] > 0:
            Dn[i, i] = Dl[i] ** (-1)
    AD = np.dot(A, Dn)
    return AD


def get_spatial_graph(num_node, self_link, inward, outward):
    I = edge2mat(self_link, num_node)
    In = normalize_digraph(edge2mat(inward, num_node))
    Out = normalize_digraph(edge2mat(outward, num_node))
    A = np.stack((I, In, Out))
    return A


class Graph:
    """The Graph to model the skeleton topology."""

    def __init__(self, layout="ntu-rgb+d", strategy="spatial", max_hop=1, dilation=1):
        self.max_hop = max_hop
        self.dilation = dilation
        self.seq_len = 25
        self.get_edge(layout)
        self.hop_dis = self.get_hop_distance(self.num_node, self.edge, max_hop=max_hop)
        self.get_adjacency(strategy)

    def get_edge(self, layout):
        if layout == "ntu-rgb+d" or layout == "ntu_rgb_d":
            self.num_node = 25
            self_link = [(i, i) for i in range(self.num_node)]
            neighbor_base = [
                (1, 2), (2, 21), (3, 21), (4, 3), (5, 21), (6, 5), (7, 6),
                (8, 7), (9, 21), (10, 9), (11, 10), (12, 11), (13, 1),
                (14, 13), (15, 14), (16, 15), (17, 1), (18, 17), (19, 18),
                (20, 19), (22, 23), (23, 8), (24, 25), (25, 12)
            ]
            neighbor_link = [(i - 1, j - 1) for (i, j) in neighbor_base]
            self.self_link = self_link
            self.neighbor_link = neighbor_link
            self.inward = [(i, j) for (i, j) in neighbor_link]
            self.outward = [(j, i) for (i, j) in self.inward]
            self.neighbor = self.inward + self.outward
            self.edge = self.self_link + self.neighbor
            self.center = 21 - 1
        elif layout == "ntu120":
            self.get_edge("ntu-rgb+d")
        elif layout == "ucla" or layout == "nw_ucla":
            self.num_node = 20
            self_link = [(i, i) for i in range(self.num_node)]
            neighbor_base = [
                (1, 2), (2, 3), (4, 3), (5, 3), (6, 5), (7, 6), (8, 7),
                (9, 3), (10, 9), (11, 10), (12, 11), (13, 1), (14, 13),
                (15, 14), (16, 15), (17, 1), (18, 17), (19, 18), (20, 19)
            ]
            neighbor_link = [(i - 1, j - 1) for (i, j) in neighbor_base]
            self.self_link = self_link
            self.neighbor_link = neighbor_link
            self.inward = [(i, j) for (i, j) in neighbor_link]
            self.outward = [(j, i) for (i, j) in self.inward]
            self.neighbor = self.inward + self.outward
            self.edge = self.self_link + self.neighbor
            self.center = 3 - 1
        else:
            raise ValueError(f"Unknown layout: {layout}")

    def get_hop_distance(self, num_node, edge, max_hop=1):
        A = np.zeros((num_node, num_node))
        for i, j in edge:
            A[j, i] = 1
            A[i, j] = 1

        # compute hop steps
        hop_dis = np.zeros((num_node, num_node)) + np.inf
        transfer_mat = [np.linalg.matrix_power(A, d) for d in range(max_hop + 1)]
        arrive_mat = np.stack(transfer_mat) > 0
        for d in range(max_hop, -1, -1):
            hop_dis[arrive_mat[d]] = d
        return hop_dis

    def get_adjacency(self, strategy):
        valid_hop = range(0, self.max_hop + 1, self.dilation)
        adjacency = np.zeros((self.num_node, self.num_node))
        for hop in valid_hop:
            adjacency[self.hop_dis == hop] = 1
        normalize_adjacency = normalize_digraph(adjacency)

        if strategy == "uniform":
            A = np.zeros((1, self.num_node, self.num_node))
            A[0] = normalize_adjacency
            self.A = A
        elif strategy == "distance":
            A = np.zeros((len(valid_hop), self.num_node, self.num_node))
            for i, hop in enumerate(valid_hop):
                A[i][self.hop_dis == hop] = normalize_adjacency[self.hop_dis == hop]
            self.A = A
        elif strategy == "spatial":
            A = []
            for hop in valid_hop:
                a_root = np.zeros((self.num_node, self.num_node))
                a_close = np.zeros((self.num_node, self.num_node))
                a_further = np.zeros((self.num_node, self.num_node))
                for i in range(self.num_node):
                    for j in range(self.num_node):
                        if self.hop_dis[j, i] == hop:
                            if self.hop_dis[j, self.center] == self.hop_dis[i, self.center]:
                                a_root[j, i] = normalize_adjacency[j, i]
                            elif self.hop_dis[j, self.center] > self.hop_dis[i, self.center]:
                                a_close[j, i] = normalize_adjacency[j, i]
                            else:
                                a_further[j, i] = normalize_adjacency[j, i]
                if hop == 0:
                    A.append(a_root)
                else:
                    A.append(a_root + a_close)
                    A.append(a_further)
            A = np.stack(A)
            self.A = A
        else:
            raise ValueError(f"Unknown strategy: {strategy}")
