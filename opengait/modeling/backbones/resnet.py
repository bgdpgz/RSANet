from torch.nn import functional as F
import torch.nn as nn
import torch
from torchvision.models.resnet import BasicBlock, Bottleneck, ResNet
# from ..modules import BasicConv2d

class BasicConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, **kwargs):
        super(BasicConv2d, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size,
                              stride=stride, padding=padding, bias=False, **kwargs)

    def forward(self, x):
        x = self.conv(x)
        return x

block_map = {'BasicBlock': BasicBlock,
             'Bottleneck': Bottleneck}


class RSANet2D(ResNet):
    def __init__(self, block, channels=[32, 64, 128, 256], in_channel=1, layers=[1, 2, 2, 1], strides=[1, 2, 2, 1], maxpool=True):
        if block in block_map.keys():
            block = block_map[block]
        else:
            raise ValueError(
                "Error type for -block-Cfg-, supported: 'BasicBlock' or 'Bottleneck'.")
        self.maxpool_flag = maxpool
        super(RSANet2D, self).__init__(block, layers)

        # Not used #
        self.fc = None
        ############
        self.inplanes = channels[0]
        self.bn1 = nn.BatchNorm2d(self.inplanes)

        self.conv1 = BasicConv2d(in_channel, self.inplanes, 3, 1, 1)

        self.layer1 = self._make_layer(
            block, channels[0], layers[0], stride=strides[0], dilate=False)
        self.layer2 = self._make_layer(
            block, channels[1], layers[1], stride=strides[1], dilate=False)
        self.layer3 = self._make_layer(
            block, channels[2], layers[2], stride=strides[2], dilate=False)
        self.layer4 = self._make_layer(
            block, channels[3], layers[3], stride=strides[3], dilate=False)

    def _make_layer(self, block, planes, blocks, stride=1, dilate=False):
        if blocks >= 1:
            layer = super()._make_layer(block, planes, blocks, stride=stride, dilate=dilate)
        else:
            def layer(x): return x
        return layer

    def fm(self, x):
        n, c, s, h, w = x.size()
        embed_x = x.float()
        embed_x = embed_x.reshape(n, c, s, h * w)
        embed_x = torch.mean(embed_x, dim=-1)
        embed_x = F.normalize(embed_x, dim=1)
        x_fm = embed_x.transpose(1, 2).matmul(embed_x)
        x_att = torch.sum(x_fm, dim=2).view(n,s,1)/s
        x_att_min = torch.min(x_att, dim=1)[0].view(n,1,1)
        x_att_max = torch.max(x_att, dim=1)[0].view(n,1,1)
        x_att = (x_att-x_att_min+0.025)/(x_att_max-x_att_min+0.025) #n,s,1

        return x_att

    def attention(self, x):
        n, c, s, h, w = x.size()
        x = x.permute(0, 2, 1, 3, 4).reshape(n,s,c*h*w)
        x = x*self.x_att
        x = x.reshape(n,s,c,h,w).permute(0, 2, 1, 3, 4)
        return x


    def forward(self, x , n, s):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        if self.maxpool_flag:
            x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)

        bs = x.shape[0] // s
        x = x.view(bs, s, x.shape[1], x.shape[2], x.shape[3])
        x = x.permute(0, 2, 1, 3, 4)
        self.x_att = self.fm(x)
        x = self.attention(x)
        x = x.permute(0, 2, 1, 3, 4)
        x = x.reshape(x.shape[0] * x.shape[1], x.shape[2], x.shape[3], x.shape[4])

        x = self.layer3(x)

        x = x.view(bs, s, x.shape[1], x.shape[2], x.shape[3])
        x = x.permute(0, 2, 1, 3, 4)
        x = self.attention(x)
        x = x.permute(0, 2, 1, 3, 4)
        x = x.reshape(x.shape[0] * x.shape[1], x.shape[2], x.shape[3], x.shape[4])

        x = self.layer4(x)

        x = x.view(bs, s, x.shape[1], x.shape[2], x.shape[3])
        x = x.permute(0, 2, 1, 3, 4)
        x = self.attention(x)
        x = x.permute(0, 2, 1, 3, 4)
        x = x.reshape(x.shape[0] * x.shape[1], x.shape[2], x.shape[3], x.shape[4])


        return x, None

