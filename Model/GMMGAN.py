import numpy as np
import torch.nn.functional as F
import torch
import torch.nn as nn
from torch.autograd import Variable
from gcn_lib import *
from trainer.ScConv import *

class ViGBlock(nn.Module):
    def __init__(self, in_features, HW):
        super(ViGBlock, self).__init__()
        self.k = 9  # neighbor num
        self.conv = 'mr'  # graph conv layer {edge, mr}
        self.act = 'gelu'  # activation layer {relu, prelu, leakyrelu, gelu, hswish}
        self.norm = 'instance'  # batch or instance normalization {batch, instance}
        self.bias = True  # bias of conv layer True or False
        self.dropout = 0.0  # dropout rate
        self.use_dilation = True  # use dilated knn or not
        self.epsilon = 0.2  # stochastic epsilon for gc
        self.use_stochastic = False  # stochastic for gcn, True or False
        self.drop_path = 0.0
        self.HW = HW
        vig_blocks = [
            Grapher(in_channels=in_features, kernel_size=7, dilation=1, conv=self.conv, act=self.act,
                    norm=self.norm,
                    bias=self.bias, stochastic=self.use_stochastic, epsilon=self.epsilon, r=1, n=self.HW,
                    drop_path=self.drop_path,
                    relative_pos=True),
            ScConv(in_features),
            nn.InstanceNorm2d(in_features),
            nn.ReLU(inplace=True),
            Grapher(in_features, 7, 1, self.conv, self.act, self.norm,
                    self.bias, self.use_stochastic, self.epsilon, r=1, n=self.HW, drop_path=self.drop_path,
                    relative_pos=True),
            ScConv(in_features),
            nn.InstanceNorm2d(in_features)
        ]
        self.conv_block = nn.Sequential(*vig_blocks)

    def forward(self, x):
        return x + self.conv_block(x)

class ResidualBlock(nn.Module):
    def __init__(self, in_features):
        super(ResidualBlock, self).__init__()
        conv_block = [
                      nn.ReflectionPad2d(1),
                      nn.Conv2d(in_features, in_features, 3),
                      nn.InstanceNorm2d(in_features),
                      nn.ReLU(inplace=True),
                      nn.ReflectionPad2d(1),
                      nn.Conv2d(in_features, in_features, 3),
                      nn.InstanceNorm2d(in_features)]
        self.conv_block = nn.Sequential(*conv_block)

    def forward(self, x):
        return x + self.conv_block(x)

# image genrator G
class Generator(nn.Module):
    def __init__(self, input_nc, output_nc, n_blocks=9, n_vig_blocks=1):
        super(Generator, self).__init__()
        self.pos_embed = nn.Parameter(torch.zeros(1, 256, 256 // 4, 256 // 4))
        self.HW = 256 // 4 * 256 // 4

        # image encoder
        model_head = [nn.ReflectionPad2d(2),
                      nn.Conv2d(input_nc, 64 , 7),
                      nn.InstanceNorm2d(64),
                      nn.ReLU(inplace=True)]

        in_features = 64
        out_features = in_features * 2
        for _ in range(2):
            model_head += [nn.Conv2d(in_features, out_features, 3, stride=2, padding=1),
                           nn.InstanceNorm2d(out_features),
                           nn.ReLU(inplace=True)]
            in_features = out_features
            out_features = in_features * 2

        model_body = []
        # GSCM
        for _ in range(n_vig_blocks):
           model_body += [ViGBlock(in_features, self.HW)]
        # Residual blocks
        for _ in range(n_blocks):
            model_body += [ResidualBlock(in_features)]

        # image decoder
        model_tail = []
        out_features = in_features // 2
        for _ in range(2):
            model_tail += [
                nn.ConvTranspose2d(in_features, out_features, 3, stride=2, padding=1, output_padding=1),
                nn.InstanceNorm2d(out_features),
                nn.ReLU(inplace=True)]

            in_features = out_features
            out_features = in_features // 2

        model_tail += [nn.ReflectionPad2d(3),
                       nn.Conv2d(64, output_nc, 7),
                       nn.Tanh()]

        self.model_head = nn.Sequential(*model_head)
        self.model_body = nn.Sequential(*model_body)
        self.model_tail = nn.Sequential(*model_tail)

    def forward(self, x):
        x = self.model_head(x) + self.pos_embed
        x = self.model_body(x)
        x = self.model_tail(x)
        return x


# D
class MLNet(nn.Module):
    def __init__(self, in_channal, out_dim):
        super(MLNet, self).__init__()
        dim = 32

        model = [nn.Conv2d(in_channal, dim, 3, 1, 1),
                 nn.PReLU(),
                 nn.MaxPool2d(kernel_size=2, stride=2),
                 nn.Conv2d(dim, dim * 2, 3, 1, 1),
                 nn.PReLU(),
                 nn.MaxPool2d(kernel_size=2, stride=2),
                 nn.Conv2d(dim * 2, dim * 4, 3, 1, 1),
                 nn.PReLU(),
                 nn.MaxPool2d(kernel_size=2, stride=2),
                 nn.Conv2d(dim * 4, dim * 4, 3, 1, 1),
                 nn.PReLU(),
                 nn.Conv2d(dim * 4, dim * 8, 3, 1, 1),
                 nn.PReLU(),
                 nn.MaxPool2d(kernel_size=2, stride=2),
                 nn.Conv2d(dim * 8, dim * 8, 3, 1, 1),
                 nn.PReLU(),
                 nn.Conv2d(dim * 8, dim * 16, 3, 1, 1),
                 nn.PReLU(),
                 nn.MaxPool2d(kernel_size=2, stride=2),
                 ]

        self.image_to_features = nn.Sequential(*model)
        self.features_to_prob = nn.Sequential(
            nn.Linear(1024 * dim, 256),
            nn.PReLU(),
            nn.Linear(256, 256),
            nn.PReLU(),
            nn.Linear(256, out_dim)
        )

    def forward(self, x):
        batch_size = x.size()[0]
        x = self.image_to_features(x)
        x = x.view(batch_size, -1)
        x = self.features_to_prob(x)
        x = F.normalize(x)
        return x