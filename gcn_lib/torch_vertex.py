# 2022.06.17-Changed for building ViG model
#            Huawei Technologies Co., Ltd. <foss@huawei.com>
import numpy as np
import torch
# from timm.models.layers import DropPath
from torch import nn
from gcn_lib.torch_nn import BasicConv, batched_index_select, act_layer, MLP
from gcn_lib.torch_edge import DenseDilatedKnnGraph
from gcn_lib.pos_embed import get_2d_relative_pos_embed
import torch.nn.functional as F

from trainer.ScConv import *


# isomap
from sklearn.manifold import Isomap
import matplotlib.pyplot as plt
from scipy.spatial.distance import pdist, squareform


# from timm.models.layers import DropPath


class MRConv2d(nn.Module):
    """
    Max-Relative Graph Convolution (Paper: https://arxiv.org/abs/1904.03751) for dense data type：     密集数据类型的    最大相对图卷积
    聚合更新结点操作
    """

    def __init__(self, in_channels, out_channels, act='relu', norm=None, bias=True):
        super(MRConv2d, self).__init__()
        self.nn = BasicConv([in_channels * 2, out_channels], act, norm, bias)
        # self.nn = MLP([in_channels * 2, out_channels], act, norm, bias)

    def forward(self, x, edge_index, y=None):
        # 聚合    b, c, n, k(b:batchsize c:channels n:num_points k:neighbor)
        x_i = batched_index_select(x, edge_index[1])        # 源节点 batched_index_select 根据 edge_index 中的节点索引从输入 x 中选择节点的特征，分别存储在 x_i 和 x_j
        if y is not None:
            x_j = batched_index_select(y, edge_index[0])    # 目标节点
        else:
            x_j = batched_index_select(x, edge_index[0])
        # (1,256,4096,1)
        x_j, _ = torch.max(x_j - x_i, -1, keepdim=True)     # 计算 x_j - x_i 的最大值，并在最后一个维度上保持一个维度 (计算出相对值中的最大差异)  拼接得到新的输入张量

        b, c, n, _ = x.shape
        x = torch.cat([x.unsqueeze(2), x_j.unsqueeze(2)], dim=2).reshape(b, 2 * c, n, _)    # 拼接操作
        return self.nn(x)                                                                   # x: b, 2c, n, k


class EdgeConv2d(nn.Module):
    """
    Edge convolution layer (with activation, batch normalization) for dense data type
    """

    def __init__(self, in_channels, out_channels, act='relu', norm=None, bias=True):
        super(EdgeConv2d, self).__init__()
        self.nn = BasicConv([in_channels * 2, out_channels], act, norm, bias)

    def forward(self, x, edge_index, y=None):
        x_i = batched_index_select(x, edge_index[1])
        if y is not None:
            x_j = batched_index_select(y, edge_index[0])
        else:
            x_j = batched_index_select(x, edge_index[0])
        max_value, _ = torch.max(self.nn(torch.cat([x_i, x_j - x_i], dim=1)), -1, keepdim=True)
        return max_value


class GraphSAGE(nn.Module):
    """
    GraphSAGE Graph Convolution (Paper: https://arxiv.org/abs/1706.02216) for dense data type
    """

    def __init__(self, in_channels, out_channels, act='relu', norm=None, bias=True):
        super(GraphSAGE, self).__init__()
        self.nn1 = BasicConv([in_channels, in_channels], act, norm, bias)
        self.nn2 = BasicConv([in_channels * 2, out_channels], act, norm, bias)

    def forward(self, x, edge_index, y=None):
        if y is not None:
            x_j = batched_index_select(y, edge_index[0])
        else:
            x_j = batched_index_select(x, edge_index[0])
        x_j, _ = torch.max(self.nn1(x_j), -1, keepdim=True)
        return self.nn2(torch.cat([x, x_j], dim=1))


class GINConv2d(nn.Module):
    """
    GIN Graph Convolution (Paper: https://arxiv.org/abs/1810.00826) for dense data type
    """

    def __init__(self, in_channels, out_channels, act='relu', norm=None, bias=True):
        super(GINConv2d, self).__init__()
        self.nn = BasicConv([in_channels, out_channels], act, norm, bias)
        eps_init = 0.0
        self.eps = nn.Parameter(torch.Tensor([eps_init]))

    def forward(self, x, edge_index, y=None):
        if y is not None:
            x_j = batched_index_select(y, edge_index[0])
        else:
            x_j = batched_index_select(x, edge_index[0])
        x_j = torch.sum(x_j, -1, keepdim=True)
        return self.nn((1 + self.eps) * x + x_j)


class GraphConv2d(nn.Module):
    """
    Static graph convolution layer:静态图卷积层
    """

    def __init__(self, in_channels, out_channels, conv='edge', act='relu', norm=None, bias=True):   # mr
        super(GraphConv2d, self).__init__()
        if conv == 'edge':
            self.gconv = EdgeConv2d(in_channels, out_channels, act, norm, bias)
        elif conv == 'mr':
            self.gconv = MRConv2d(in_channels, out_channels, act, norm, bias)
        elif conv == 'sage':
            self.gconv = GraphSAGE(in_channels, out_channels, act, norm, bias)
        elif conv == 'gin':
            self.gconv = GINConv2d(in_channels, out_channels, act, norm, bias)
        else:
            raise NotImplementedError('conv:{} is not supported'.format(conv))

    def forward(self, x, edge_index, y=None):
        return self.gconv(x, edge_index, y)


class DyGraphConv2d(GraphConv2d):
    """
    Dynamic graph convolution layer：动态图卷积层
    继承 GraphConv2d
    """

    def __init__(self, in_channels, out_channels, kernel_size=9, dilation=1, conv='edge', act='relu',
                 norm=None, bias=True, stochastic=False, epsilon=0.0, r=1):
        super(DyGraphConv2d, self).__init__(in_channels, out_channels, conv, act, norm, bias)
        self.k = kernel_size
        self.d = dilation
        self.r = r
        self.dilated_knn_graph = DenseDilatedKnnGraph(kernel_size, dilation, stochastic, epsilon)

    # edge_index：邻接矩阵   节点特征：tensor
    def forward(self, x, relative_pos=None):
        B, C, H, W = x.shape
        y = None                # y 减少节点个数 减少计算邻接矩阵的计算量
        if self.r > 1:          # r = 1
            y = F.avg_pool2d(x, self.r, self.r)
            y = y.reshape(B, C, -1, 1).contiguous()
        x = x.reshape(B, C, -1, 1).contiguous()                     # contiguous() 来确保张量在内存中是连续性的
        edge_index = self.dilated_knn_graph(x, y, relative_pos)     # 图结构的索引，用于指示节点之间的连接关系  （邻接矩阵） (2,1,4096,9)
        x = super(DyGraphConv2d, self).forward(x, edge_index, y)    # 重用父类GraphConv2d的前向传播
        return x.reshape(B, -1, H, W).contiguous()                  # -1 表示一个占位符，它会根据其他维度的大小自动计算，通常是为了保持张量的总大小不变



class Grapher(nn.Module):
    """

    Grapher module with graph convolution and fc layers:有图卷积和全连接层的图模块

    """

    def __init__(self, in_channels, kernel_size=9, dilation=1, conv='edge', act='relu', norm=None,
                 bias=True, stochastic=False, epsilon=0.0, r=1, n=196, drop_path=0.0, relative_pos=False):
        """
        @param in_channels: 输入通道            256
        @param kernel_size: 卷积核大小               要查找的邻居数量
        @param dilation: dilated aggregation
        @param conv: 卷积类型
        @param act: 激活函数类型
        @param norm: 归一化层类型
        @param bias: 偏置
        @param stochastic: stochastic for gcn, True or False
        @param epsilon: stochastic epsilon for gcn
        @param r: number of heads       1
        @param n: number of nodes       4096
        @param drop_path: stochastic depth decay rule       随机深度衰减
        @param relative_pos: relative positional encoding
        """

        super(Grapher, self).__init__()
        self.channels = in_channels
        self.n = n
        self.r = r
        self.fc1 = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, 1, stride=1, padding=0),
            nn.InstanceNorm2d(in_channels),
        )
        self.graph_conv = DyGraphConv2d(in_channels, in_channels * 2, kernel_size, dilation, conv,
                                        act, norm, bias, stochastic, epsilon, r)
        self.fc2 = nn.Sequential(
            nn.Conv2d(in_channels * 2, in_channels, 1, stride=1, padding=0),
            nn.InstanceNorm2d(in_channels),
        )
        self.drop_path = nn.Identity()

        self.relative_pos = None
        if relative_pos:
            print('using relative_pos')
            # unsqueeze(0).unsqueeze(1):对获取的相对位置张量进行维度扩展，使其成为形状为 (1, 1, h, w) 的四维张量
            # 1,1,4096,4096
            relative_pos_tensor = torch.from_numpy(np.float32(get_2d_relative_pos_embed(in_channels,
                                                                                        int(n ** 0.5)))).unsqueeze(
                0).unsqueeze(1)
            # 使用双三次插值 (bicubic) 将相对位置信息的大小调整为 (n, n // (r * r))
            relative_pos_tensor = F.interpolate(
                relative_pos_tensor, size=(n, n // (r * r)), mode='bicubic', align_corners=False)

            self.relative_pos = nn.Parameter(-relative_pos_tensor.squeeze(1), requires_grad=False)

    def _get_relative_pos(self, relative_pos, H, W):
        if relative_pos is None or H * W == self.n:
            return relative_pos
        else:
            N = H * W
            N_reduced = N // (self.r * self.r)
            return F.interpolate(relative_pos.unsqueeze(0), size=(N, N_reduced), mode="bicubic").squeeze(0)

    def forward(self, x):
        _tmp = x
        x = self.fc1(x)
        B, C, H, W = x.shape        # tensor
        # 相对位置编码
        relative_pos = self._get_relative_pos(self.relative_pos, H, W)
        x = self.graph_conv(x, relative_pos)
        x = self.fc2(x)
        # drop_path 防止过拟合、提高模型泛化性能
        x = self.drop_path(x) + _tmp
        return x

if __name__ == '__main__':

    x = torch.rand(1,256,64,64)
    HW = 256 // 4 * 256 // 4
    model = DyGraphConv2d(in_channels=256, out_channels=512, kernel_size=9, dilation=1, conv='mr', act='gelu', norm='instance',
                    bias=True, stochastic=False, epsilon=0.2, r=1,)
    print(model(x))     # (1,256,64,64)
    print(model(x).shape)
