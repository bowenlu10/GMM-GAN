# 2022.06.17-Changed for building ViG model
#            Huawei Technologies Co., Ltd. <foss@huawei.com>
import math
import torch
from torch import nn
import torch.nn.functional as F
import numpy as np


def pairwise_distance(x):
    """
    Compute pairwise distance of a point cloud.
    Args:
        x: tensor (batch_size, num_points, num_dims)
    Returns:
        pairwise distance: (batch_size, num_points, num_points)
    """

    with torch.no_grad():
        x_inner = -2*torch.matmul(x, x.transpose(2, 1))
        x_square = torch.sum(torch.mul(x, x), dim=-1, keepdim=True)
        return x_square + x_inner + x_square.transpose(2, 1)

        # x_inner = torch.matmul(x, x.transpose(2, 1))
        # x_square = torch.sum(torch.mul(x, x), dim=-1, keepdim=True)
        # return torch.exp(-0.1 * (x_square + x_square.transpose(2, 1) - 2 * x_inner))

        # x_normalized = F.normalize(x, p=2, dim=-1)
        # x_inner = torch.matmul(x_normalized, x_normalized.transpose(2, 1))
        # dist = 1 - x_inner

        # x_normalized = F.normalize(x, p=2, dim=-1)
        # x_inner = torch.matmul(x_normalized, x_normalized.transpose(2, 1))
        # x_norm = torch.norm(x_normalized, dim=-1, keepdim=True)
        # dist = 1- x_inner / (x_norm * x_norm.transpose(2, 1))
        # return dist


def part_pairwise_distance(x, start_idx=0, end_idx=1):
    """
    Compute pairwise distance of a point cloud.
    Args:
        x: tensor (batch_size, num_points, num_dims)
    Returns:
        pairwise distance: (batch_size, num_points, num_points)
    """
    with torch.no_grad():
        x_part = x[:, start_idx:end_idx]
        x_square_part = torch.sum(torch.mul(x_part, x_part), dim=-1, keepdim=True)
        x_inner = -2*torch.matmul(x_part, x.transpose(2, 1))
        x_square = torch.sum(torch.mul(x, x), dim=-1, keepdim=True)
        return x_square_part + x_inner + x_square.transpose(2, 1)


def xy_pairwise_distance(x, y):
    """
    Compute pairwise distance of a point cloud.
    Args:
        x: tensor (batch_size, num_points, num_dims)
    Returns:
        pairwise distance: (batch_size, num_points, num_points)
    """
    with torch.no_grad():
        xy_inner = -2*torch.matmul(x, y.transpose(2, 1))                # B, points, points（矩阵乘法  内积）
        x_square = torch.sum(torch.mul(x, x), dim=-1, keepdim=True)     # B, points, 1(逐元素平方)
        y_square = torch.sum(torch.mul(y, y), dim=-1, keepdim=True)     # B, points, 1
        return x_square + xy_inner + y_square.transpose(2, 1)           # B, points, points


def dense_knn_matrix(x, k=16, relative_pos=None):
    """Get KNN based on the pairwise distance.
    Args:
        x: (batch_size, num_dims, num_points, 1)
        k: int
    Returns:
        nearest neighbors: (batch_size, num_points, k) (batch_size, num_points, k)
    """
    with torch.no_grad():
        x = x.transpose(2, 1).squeeze(-1)
        batch_size, n_points, n_dims = x.shape
        ### memory efficient implementation ###
        n_part = 10000
        if n_points > n_part:
            nn_idx_list = []
            groups = math.ceil(n_points / n_part)
            for i in range(groups):
                start_idx = n_part * i
                end_idx = min(n_points, n_part * (i + 1))
                dist = part_pairwise_distance(x.detach(), start_idx, end_idx)
                if relative_pos is not None:
                    dist += relative_pos[:, start_idx:end_idx]
                _, nn_idx_part = torch.topk(-dist, k=k)
                nn_idx_list += [nn_idx_part]
            nn_idx = torch.cat(nn_idx_list, dim=1)
        else:
            dist = pairwise_distance(x.detach())
            # dist = np.exp(-0.1 * (np.linalg.norm(x1.cpu().detach().numpy() - x2.cpu().detach().numpy())))
            if relative_pos is not None:
                dist += relative_pos
            _, nn_idx = torch.topk(-dist, k=k) # b, n, k
            # nn_idx = nn_idx.expand(8,-1,-1)
        ######
        center_idx = torch.arange(0, n_points, device=x.device).repeat(batch_size, k, 1).transpose(2, 1)
    return torch.stack((nn_idx, center_idx), dim=0)     # 新的第0维上堆叠（2，B，N，K）



def xy_dense_knn_matrix(x, y, k=9, relative_pos=None):
    """Get KNN based on the pairwise distance.
    Args:
        x: (batch_size, num_dims, num_points, 1)
        k: int
    Returns:
        nearest neighbors: (batch_size, num_points, k) (batch_size, num_points, k)
    """
    with torch.no_grad():
        x = x.transpose(2, 1).squeeze(-1)           # B, points, dims
        y = y.transpose(2, 1).squeeze(-1)           # B, points, dims
        batch_size, n_points, n_dims = x.shape
        dist = xy_pairwise_distance(x.detach(), y.detach())     # B, points, points
        # dist = np.exp(-0.1 * (np.linalg.norm(x.cpu().detach().numpy() - y.cpu().detach().numpy())))
        if relative_pos is not None:
            dist += relative_pos
        _, nn_idx = torch.topk(-dist, k=k)                  # B, num_points, k
        center_idx = torch.arange(0, n_points, device=x.device).repeat(batch_size, k, 1).transpose(2, 1)        # B, num_points, k
    return torch.stack((nn_idx, center_idx), dim=0)         # 2, B, num_points, k


class DenseDilated(nn.Module):
    """
    Find dilated neighbor from neighbor list:从邻居列表中查找扩展的邻居

    edge_index: (2, batch_size, num_points, k)
    """
    def __init__(self, k=9, dilation=1, stochastic=False, epsilon=0.0):
        super(DenseDilated, self).__init__()
        self.dilation = dilation        # 1
        self.stochastic = stochastic    # false
        self.epsilon = epsilon          # 0.2
        self.k = k                      # 9

    def forward(self, edge_index):
        if self.stochastic:
            if torch.rand(1) < self.epsilon and self.training:
                num = self.k * self.dilation
                randnum = torch.randperm(num)[:self.k]
                edge_index = edge_index[:, :, :, randnum]
            else:
                edge_index = edge_index[:, :, :, ::self.dilation]
        else:
            edge_index = edge_index[:, :, :, ::self.dilation]
        return edge_index


class DenseDilatedKnnGraph(nn.Module):
    """

    Find the neighbors' indices based on dilated knn：基于扩张knn邻居索引查找
    作用：生成邻接矩阵，该矩阵描述了输入数据中点之间的邻居关系

    """
    def __init__(self, k=9, dilation=1, stochastic=False, epsilon=0.0):
        super(DenseDilatedKnnGraph, self).__init__()
        self.dilation = dilation                        # dilation:决定了在搜索最近邻时每个邻居点的范围：1
        self.stochastic = stochastic                    # 布尔值，决定是否使用随机化 false
        self.epsilon = epsilon                          # 控制随机化强度：0.2
        self.k = k                                      # 边：k
        self._dilated = DenseDilated(k, dilation, stochastic, epsilon)


    def forward(self, x, y=None, relative_pos=None):
        if y is not None:
            #### normalize
            x = F.normalize(x, p=2.0, dim=1)
            y = F.normalize(y, p=2.0, dim=1)
            ####
            edge_index = xy_dense_knn_matrix(x, y, self.k * self.dilation, relative_pos)
        else:
            #### normalize  通道归一化
            x = F.normalize(x, p=2.0, dim=1)
            ####
            edge_index = dense_knn_matrix(x, self.k * self.dilation, relative_pos)      # (2,1,4096,9)
        return self._dilated(edge_index)


if __name__ == '__main__':
    x = torch.rand(16, 3, 256, 256)

