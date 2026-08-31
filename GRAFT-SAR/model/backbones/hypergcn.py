"""
Hyper-GCN Backbone Module (Complete Architecture)
=================================================
Hypergraph Convolutional Network with Multi-Scale Temporal Convolutions
and Virtual Joint Ensemble for Skeleton-Based Action Recognition.
"""

import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def conv_branch_init(conv, branches):
    weight = conv.weight
    n = weight.size(0)
    k1 = weight.size(1)
    k2 = weight.size(2)
    nn.init.normal_(weight, 0, math.sqrt(2.0 / (n * k1 * k2 * branches)))
    if conv.bias is not None:
        nn.init.constant_(conv.bias, 0)


def conv_init(conv):
    if conv.weight is not None:
        nn.init.kaiming_normal_(conv.weight, mode="fan_out")
    if conv.bias is not None:
        nn.init.constant_(conv.bias, 0)


def bn_init(bn, scale):
    nn.init.constant_(bn.weight, scale)
    nn.init.constant_(bn.bias, 0)


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        if hasattr(m, "weight"):
            nn.init.kaiming_normal_(m.weight, mode="fan_out")
        if hasattr(m, "bias") and m.bias is not None and isinstance(m.bias, torch.Tensor):
            nn.init.constant_(m.bias, 0)
    elif classname.find("BatchNorm") != -1:
        if hasattr(m, "weight") and m.weight is not None:
            m.weight.data.normal_(1.0, 0.02)
        if hasattr(m, "bias") and m.bias is not None:
            m.bias.data.fill_(0)


# =========================================================================
# Graph Adjacency Generation for HyperGCN
# =========================================================================

def edge2mat(link, num_node):
    A = np.zeros((num_node, num_node))
    for i, j in link:
        A[j, i] = 1
    return A


def get_virtual_graph_ensemble(num_node, self_link, inward, outward, virtual, nums=8):
    I = np.pad(edge2mat(self_link, num_node), ((0, virtual), (0, virtual)))
    In = np.pad(edge2mat(inward, num_node), ((0, virtual), (0, virtual)))
    Out = np.pad(edge2mat(outward, num_node), ((0, virtual), (0, virtual)))
    for i in range(virtual):
        I[num_node + i, num_node + i] = 1
        In[:num_node, num_node + i] = 1
        Out[num_node + i, :num_node] = 1
    A = I + In + Out
    A = np.repeat(A[np.newaxis, :], nums, axis=0)
    return A


class HyperGraph:
    def __init__(self, num_node=25, hyper_joints=3, labeling_mode="virtual_ensemble"):
        self.num_node = num_node
        self.hyper_joints = hyper_joints
        self.self_link = [(i, i) for i in range(num_node)]
        inward_ori_index = [
            (1, 2), (2, 21), (3, 21), (4, 3), (5, 21), (6, 5), (7, 6),
            (8, 7), (9, 21), (10, 9), (11, 10), (12, 11), (13, 1),
            (14, 13), (15, 14), (16, 15), (17, 1), (18, 17), (19, 18),
            (20, 19), (22, 8), (23, 8), (24, 12), (25, 12)
        ]
        self.inward = [(i - 1, j - 1) for (i, j) in inward_ori_index]
        self.outward = [(j, i) for (i, j) in self.inward]
        self.neighbor = self.inward + self.outward
        self.A = get_virtual_graph_ensemble(
            self.num_node, self.self_link, self.inward, self.outward, self.hyper_joints, nums=8
        )


# =========================================================================
# Hypergraph Convolution Unit
# =========================================================================

class HYPERGC(nn.Module):
    def __init__(self, in_channels, out_channels, vertex_nums=25, virtual_num=3, A=None, hyper=True, num_subset=8, rel_reduction=4):
        super(HYPERGC, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.vertex_nums = vertex_nums
        self.virtual_num = virtual_num
        self.rel_reduction = rel_reduction
        self.num_subset = num_subset
        self.hyper = hyper
        mid_in_channels = in_channels // num_subset
        mid_out_channels = out_channels // num_subset
        self.mid_in_channels = mid_in_channels
        self.mid_out_channels = mid_out_channels

        if self.hyper:
            self.hidden_channels = mid_in_channels // rel_reduction
            self.to_V = nn.Conv1d(in_channels, num_subset * self.hidden_channels, kernel_size=1, groups=num_subset)
            self.to_W = nn.Sequential(
                nn.Conv1d(in_channels, num_subset * self.hidden_channels, kernel_size=1, groups=num_subset),
                nn.LeakyReLU(),
                nn.Conv1d(num_subset * self.hidden_channels, num_subset, kernel_size=1),
                nn.Tanh()
            )
            self.hyper_joint = nn.Parameter(torch.zeros(self.virtual_num, in_channels))
            self.alpha = nn.Parameter(torch.ones(1))
            self.softmax = nn.Softmax(dim=-1)

        self.conv_d = nn.Conv2d(in_channels, out_channels, kernel_size=1, groups=num_subset)
        if A is not None:
            self.PA = nn.Parameter(torch.from_numpy(A.astype(np.float32)), requires_grad=False)
            self.edge_importance = nn.Parameter(torch.ones(A.shape))
        else:
            self.PA = None
            self.edge_importance = None

        if in_channels != out_channels:
            self.down = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1),
                nn.BatchNorm2d(out_channels)
            )
        else:
            self.down = lambda x: x
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                conv_init(m)
            elif isinstance(m, nn.BatchNorm2d):
                bn_init(m, 1)
        bn_init(self.bn, 1e-6)
        if self.hyper:
            conv_init(self.to_V)
            conv_init(self.to_W[0])
            conv_init(self.to_W[2])
        conv_init(self.conv_d)

    def hyper_norm(self, H, W):
        # Cast to float32 to avoid FP16 overflow under AMP
        H = H.float()
        W = W.float()
        w = torch.diag_embed(W)
        norm_w = torch.norm(H, 1, dim=2, keepdim=True).clamp_min(1e-8)
        w_ = w / norm_w

        H_w = H @ w
        norm_v = torch.norm(H_w, 1, dim=3, keepdim=True).clamp_min(1e-8)
        h_ = H_w / norm_v
        A = h_ @ w_ @ H.transpose(3, 2)
        # Guard against any residual NaN/Inf
        A = torch.nan_to_num(A, nan=0.0, posinf=1.0, neginf=-1.0)
        return A

    def a_norm(self, A):
        d_r = torch.norm(A, 1, dim=2, keepdim=True) + 1e-8
        return A / d_r

    def forward(self, x):
        N, C, T, V = x.size()

        h_x = self.hyper_joint
        h_x = (h_x.T).unsqueeze(1)
        x = torch.cat([x, h_x.repeat(N, 1, T, 1)], dim=-1)
        V += self.virtual_num
        A = self.PA.to(x.device)
        A = self.edge_importance * A
        A = self.a_norm(A)

        if self.hyper:
            t_x = x.mean(2)
            v_x = self.to_V(t_x)

            dis_v_x = v_x.view(N, self.num_subset, self.hidden_channels, V)
            dis_v_x = dis_v_x.permute(0, 1, 3, 2).contiguous().float()
            # Stable pairwise distance: avoids gradient explosion at d=0 from torch.cdist
            diff = dis_v_x.unsqueeze(3) - dis_v_x.unsqueeze(2)          # (N,K,V,V,hidden)
            distance_x = torch.sqrt((diff * diff).sum(-1) + 1e-6)        # (N,K,V,V)
            H = torch.zeros_like(distance_x)

            topk_v, topk_indices = torch.topk(distance_x, min(9, V), largest=False)
            topk_v = self.softmax(-topk_v)
            H = torch.scatter(H, 3, topk_indices, topk_v)

            W = self.to_W(t_x)
            H = self.hyper_norm(H, W)          # returns float32
            alpha = self.relu(self.alpha).float()
            A = A.float() + alpha * H
            A = A.to(x.dtype)                  # restore original dtype

        d_x = self.conv_d(x)
        d_x = d_x.view(N, self.num_subset, self.mid_out_channels, T, V)
        y = torch.einsum("nkuv,nkctv->nkctu", A, d_x).contiguous()
        y = y.view(N, self.out_channels, T, V)

        x = x[..., :self.vertex_nums]
        y = y[..., :self.vertex_nums]

        y = self.bn(y)
        y += self.down(x)
        y = self.relu(y)
        return y, self.hyper_joint


# =========================================================================
# Multi-Scale Temporal Convolution
# =========================================================================

class TemporalConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, dilation=1):
        super(TemporalConv, self).__init__()
        pad = (kernel_size + (kernel_size - 1) * (dilation - 1) - 1) // 2
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=(kernel_size, 1),
            padding=(pad, 0),
            stride=(stride, 1),
            dilation=(dilation, 1),
            padding_mode="replicate",
        )
        self.bn = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        return self.bn(self.conv(x))


class MultiScale_TemporalConv(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size=3,
        stride=1,
        dilations=[1, 2, 3, 4],
        residual=True,
        residual_kernel_size=1,
    ):
        super().__init__()
        assert out_channels % (len(dilations) + 2) == 0, "# out channels should be multiples of # branches"

        self.num_branches = len(dilations) + 2
        branch_channels = out_channels // self.num_branches

        if not isinstance(kernel_size, list):
            kernel_size = [kernel_size] * len(dilations)

        self.branches = nn.ModuleList()
        for ks, dilation in zip(kernel_size, dilations):
            self.branches.append(
                nn.Sequential(
                    nn.Conv2d(in_channels, branch_channels, kernel_size=1, padding=0),
                    nn.BatchNorm2d(branch_channels),
                    nn.ReLU(inplace=True),
                    TemporalConv(branch_channels, branch_channels, kernel_size=ks, stride=stride, dilation=dilation),
                )
            )

        self.branches.append(
            nn.Sequential(
                nn.Conv2d(in_channels, branch_channels, kernel_size=1, padding=0),
                nn.BatchNorm2d(branch_channels),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=(3, 1), stride=(stride, 1), padding=(1, 0)),
                nn.BatchNorm2d(branch_channels),
            )
        )

        self.branches.append(
            nn.Sequential(
                nn.Conv2d(in_channels, branch_channels, kernel_size=1, padding=0, stride=(stride, 1)),
                nn.BatchNorm2d(branch_channels),
            )
        )

        if not residual:
            self.residual = lambda x: 0
        elif (in_channels == out_channels) and (stride == 1):
            self.residual = lambda x: x
        else:
            self.residual = TemporalConv(in_channels, out_channels, kernel_size=residual_kernel_size, stride=stride)

        self.apply(weights_init)

    def forward(self, x):
        res = self.residual(x)
        branch_outs = [tempconv(x) for tempconv in self.branches]
        out = torch.cat(branch_outs, dim=1)
        out += res
        return out


class TCN_GCN_unit(nn.Module):
    def __init__(self, in_channels, out_channels, num_point, hyper_joints, A, stride=1, residual=True, kernel_size=5, dilations=[1, 2], hyper=True):
        super(TCN_GCN_unit, self).__init__()
        self.gcn1 = HYPERGC(in_channels, out_channels, num_point, hyper_joints, A, hyper=hyper)
        self.tcn1 = MultiScale_TemporalConv(
            out_channels, out_channels, kernel_size=kernel_size, stride=stride, dilations=dilations, residual=False
        )
        self.relu = nn.ReLU(inplace=True)
        if not residual:
            self.residual = lambda x: 0
        elif (in_channels == out_channels) and (stride == 1):
            self.residual = lambda x: x
        else:
            self.residual = TemporalConv(in_channels, out_channels, kernel_size=1, stride=stride)

    def forward(self, x):
        y, h_x = self.gcn1(x)
        y = self.relu(self.tcn1(y) + self.residual(x))
        return y, h_x


# =========================================================================
# HyperGCN Full Model
# =========================================================================

class HyperGCNModel(nn.Module):
    """Full Hyper-GCN Model with 128 embedding channels and 9 spatial-temporal blocks."""

    def __init__(
        self,
        num_class=60,
        num_point=25,
        num_person=2,
        in_channels=3,
        hyper_joints=3,
        embedding_channels=128,
        drop_out=0.0,
        **kwargs,
    ):
        super(HyperGCNModel, self).__init__()
        self.num_class = num_class
        self.num_point = num_point
        self.embedding_channels = embedding_channels
        self.hyper_joints = hyper_joints

        self.graph = HyperGraph(num_node=num_point, hyper_joints=hyper_joints)
        A = self.graph.A

        self.data_bn = nn.BatchNorm1d(num_person * self.embedding_channels * num_point)
        self.to_joint_embedding = nn.Linear(in_channels, self.embedding_channels)
        self.pos_embedding = nn.Parameter(torch.randn(1, self.num_point, self.embedding_channels))
        self.tanh = nn.Tanh()

        self.l1 = TCN_GCN_unit(self.embedding_channels, self.embedding_channels, num_point, hyper_joints, A)
        self.l2 = TCN_GCN_unit(self.embedding_channels, self.embedding_channels, num_point, hyper_joints, A)
        self.l3 = TCN_GCN_unit(self.embedding_channels, self.embedding_channels, num_point, hyper_joints, A)
        self.l4 = TCN_GCN_unit(self.embedding_channels, self.embedding_channels * 2, num_point, hyper_joints, A, stride=2)
        self.l5 = TCN_GCN_unit(self.embedding_channels * 2, self.embedding_channels * 2, num_point, hyper_joints, A)
        self.l6 = TCN_GCN_unit(self.embedding_channels * 2, self.embedding_channels * 2, num_point, hyper_joints, A)
        self.l7 = TCN_GCN_unit(self.embedding_channels * 2, self.embedding_channels * 4, num_point, hyper_joints, A, stride=2)
        self.l8 = TCN_GCN_unit(self.embedding_channels * 4, self.embedding_channels * 4, num_point, hyper_joints, A)
        self.l9 = TCN_GCN_unit(self.embedding_channels * 4, self.embedding_channels * 4, num_point, hyper_joints, A)

        self.feat_dim = self.embedding_channels * 4  # 512
        self.fc = nn.Linear(self.feat_dim, self.num_class)

        nn.init.normal_(self.fc.weight, 0, math.sqrt(2.0 / num_class))
        bn_init(self.data_bn, 1)
        self.drop_out = nn.Dropout(drop_out) if drop_out > 0 else nn.Identity()

    def extract_feat(self, x):
        """Input: (N, C, T, V, M) -> Output pooled feature: (N, 512)"""
        N, C, T, V, M = x.size()
        x = x.permute(0, 4, 2, 3, 1).contiguous()
        x = self.to_joint_embedding(x)
        x = x + self.pos_embedding[:, :V]
        x = self.tanh(x).permute(0, 1, 3, 4, 2).contiguous()

        x = x.view(N, M * V * self.embedding_channels, T)
        x = self.data_bn(x)
        x = x.view(N, M, V, self.embedding_channels, T).permute(0, 1, 3, 4, 2).contiguous()
        x = x.view(N * M, self.embedding_channels, T, V)

        x, _ = self.l1(x)
        x1 = x
        x, _ = self.l2(x)
        x, _ = self.l3(x + x1)

        x, _ = self.l4(x)
        x4 = x
        x, _ = self.l5(x)
        x, _ = self.l6(x + x4)

        x, _ = self.l7(x)
        x7 = x
        x, _ = self.l8(x)
        x, _ = self.l9(x + x7)

        c_new = x.size(1)
        x = x.view(N, M, c_new, -1)
        feat = x.mean(3).mean(1)
        feat = self.drop_out(feat)
        return feat

    def forward(self, x):
        feat = self.extract_feat(x)
        return self.fc(feat)

