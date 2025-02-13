from torch.nn import functional as F
import torch.nn as nn
import torch
import math
from torchvision.models.resnet import BasicBlock, Bottleneck, ResNet
# from ..modules import BasicConv2d
from typing import Tuple, Optional, Callable, List, Type, Any, Union
from torch import Tensor
import numpy as np

class BasicConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, **kwargs):
        super(BasicConv2d, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size,
                              stride=stride, padding=padding, bias=False, **kwargs)

    def forward(self, x):
        x = self.conv(x)
        return x


class Conv3DNoSpatial(nn.Conv3d):

    def __init__(
            self,
            in_planes: int,
            out_planes: int,
            stride: int = 1,
            padding: int = 1,
            group: int = 1,
    ) -> None:
        super(Conv3DNoSpatial, self).__init__(
            in_channels=in_planes,
            out_channels=out_planes,
            kernel_size=(3, 1, 1),
            stride=(stride, 1, 1),
            padding=(padding, 0, 0),
            groups=group,
            bias=False)

    @staticmethod
    def get_downsample_stride(stride: int) -> Tuple[int, int, int]:
        return stride, 1, 1


class Conv3DNoTemporal(nn.Conv3d):

    def __init__(
            self,
            in_planes: int,
            out_planes: int,
            stride: int = 1,
            padding: int = 1,
            group: int = 1,
    ) -> None:
        super(Conv3DNoTemporal, self).__init__(
            in_channels=in_planes,
            out_channels=out_planes,
            kernel_size=(1, 3, 3),
            stride=(1, stride, stride),
            padding=(0, padding, padding),
            groups=group,
            bias=False)

    @staticmethod
    def get_downsample_stride(stride: int) -> Tuple[int, int, int]:
        return 1, stride, stride


class Conv3D1x1(nn.Conv3d):

    def __init__(
            self,
            in_planes: int,
            out_planes: int,
            stride: int = 1,
            padding: int = 1,
            group: int = 1,
    ) -> None:
        super(Conv3D1x1, self).__init__(
            in_channels=in_planes,
            out_channels=out_planes,
            kernel_size=(1, 1, 1),
            stride=(1, stride, stride),
            padding=(0, 0, 0),
            groups=group,
            bias=False)

    @staticmethod
    def get_downsample_stride(stride: int) -> Tuple[int, int, int]:
        return 1, 1, 1


class Conv3DSimple(nn.Conv3d):
    def __init__(
            self,
            in_planes: int,
            out_planes: int,
            stride: int = 1,
            padding: int = 1,
            group: int = 1,
    ) -> None:
        super(Conv3DSimple, self).__init__(
            in_channels=in_planes,
            out_channels=out_planes,
            kernel_size=(3, 3, 3),
            stride=(1, stride, stride),
            padding=padding,
            groups=group,
            bias=False)

    @staticmethod
    def get_downsample_stride(stride: int) -> Tuple[int, int, int]:
        return 1, stride, stride





class MTB(nn.Module):
    expansion = 1

    def __init__(
            self,
            inplanes: int,
            planes: int,
            stride: int = 1,
            downsample: Optional[nn.Module] = None,
            tem_nums: int = 3,
            seq: int = 1,
    ) -> None:

        super(MTB, self).__init__()
        self.conv1 = nn.Sequential(
            Conv3DNoTemporal(inplanes, planes, stride),
            nn.BatchNorm3d(planes),
            nn.ReLU(inplace=True)
        )
        self.tem_nums = tem_nums
        self.basic_layers = nn.ModuleList()
        for i in range(tem_nums):
            if i == 0:
                self.basic_layers.append(nn.Sequential(
                    Conv3D1x1(planes, planes // 2),
                    nn.BatchNorm3d(planes // 2),
                    nn.ReLU(inplace=True)
                ))
            elif i < tem_nums - 1:
                self.basic_layers.append(nn.Sequential(
                    Conv3DNoSpatial(planes // int(math.pow(2, i)), planes // int(math.pow(2, i + 1))),
                    nn.BatchNorm3d(planes // int(math.pow(2, i + 1))),
                    nn.ReLU(inplace=True)
                ))
            else:
                self.basic_layers.append(nn.Sequential(
                    Conv3DNoSpatial(planes // int(math.pow(2, i)), planes // int(math.pow(2, i))),
                    nn.BatchNorm3d(planes // int(math.pow(2, i))),
                    nn.ReLU(inplace=True)
                ))
        self.conv3 = nn.Sequential(
            Conv3DNoTemporal(planes, planes),
            nn.BatchNorm3d(planes)
        )
        self.conv1_k = nn.Parameter(torch.tensor([[[0, 1, 0]]], dtype=torch.float32))
        self.relu1 = nn.ReLU(inplace=True)
        self.relu2 = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride
        self.seq = seq + 4
        self.eps = 1e-3
        self.iter = 0


    def fm(self, att):
        att1 = att.permute(0, 2, 1)
        self.conv1_k.data = F.softmax(self.conv1_k.data / 0.1, dim=2)
        att2 = F.conv1d(att1, self.conv1_k, stride=1, padding=1)
        att3 = att2.permute(0, 2, 1)
        return att3

    def forward(self, x: Tuple) -> Tensor:
        x, att = x
        residual = x

        out1 = self.conv1(x)
        outx = out1
        out2 = []
        for i, conv in enumerate(self.basic_layers):
            if i > 0 and i < self.tem_nums:
                outx = conv(outx)
                att = self.fm(att)
            else:
                outx = conv(outx)
            out2.append(outx)
        out = torch.cat(out2, dim=1)
        out = self.relu1(out1 + out)
        out = self.conv3(out)
        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu2(out)
        return (out, att)


block_map = {'BasicBlock': BasicBlock,
             'Bottleneck': Bottleneck,
             'MTB': MTB,}


class RSANet3D(nn.Module):
    def __init__(self, block, channels=[32, 64, 128, 256], in_channel=1, layers=[1, 2, 2, 1], strides=[1, 2, 2, 1],
                 maxpool=True):
        if block in block_map.keys():
            block = block_map[block]
        else:
            raise ValueError(
                "Error type for -block-Cfg-, supported: 'BasicBlock' or 'Bottleneck'.")
        block3D = block_map['MTB']
        self.maxpool_flag = maxpool
        super(RSANet3D, self).__init__()
        self._norm_layer = nn.BatchNorm2d
        self.dilation = 1
        self.groups = 1
        self.base_width = 64
        self.x_att = None
        self.fc = None
        ############
        self.inplanes = channels[0]
        self.bn1 = nn.BatchNorm2d(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.conv1 = BasicConv2d(in_channel, self.inplanes, 3, 1, 1)

        self.layer1 = self._make_layer(block, channels[0], layers[0], stride=strides[0], dilate=False)
        self.layer2 = self._make_layer(block, channels[1], layers[1], stride=strides[1], dilate=False)
        self.layer3 = self._make_layer_P3D(block3D, channels[2], layers[2], strides[2], seq=1)
        self.layer4 = self._make_layer_P3D(block3D, channels[3], layers[3], strides[3], seq=17)
        self._initialize_weights()

    def _make_layer(self, block, planes, blocks, stride=1, dilate=False):
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                self.conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, self.groups,
                            self.base_width, previous_dilation, norm_layer))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, groups=self.groups,
                                base_width=self.base_width, dilation=self.dilation,
                                norm_layer=norm_layer))

        return nn.Sequential(*layers)

    def _make_layer_P3D(
            self,
            block: Type[Union[BasicBlock, Bottleneck]],
            planes: int,
            blocks: int,
            stride: int = 1,
            seq: int = 1,
    ) -> nn.Sequential:
        downsample = None

        if stride != 1 or self.inplanes != planes * block.expansion:
            ds_stride = Conv3DSimple.get_downsample_stride(stride)
            downsample = nn.Sequential(
                nn.Conv3d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=ds_stride, bias=False),
                nn.BatchNorm3d(planes * block.expansion)
            )
        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, seq=seq))

        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            seq += 4
            layers.append(block(self.inplanes, planes, seq=seq))

        return nn.Sequential(*layers)

    def fm(self, x):
        n, c, s, h, w = x.size()
        embed_x = x.float()
        embed_x = embed_x.reshape(n, c, s, h * w)
        embed_x = torch.mean(embed_x, dim=-1)
        embed_x = F.normalize(embed_x, dim=1)
        x_fm = embed_x.transpose(1, 2).matmul(embed_x)
        x_att = torch.sum(x_fm, dim=2).view(n, s, 1) / s
        x_att_min = torch.min(x_att, dim=1)[0].view(n, 1, 1)
        x_att_max = torch.max(x_att, dim=1)[0].view(n, 1, 1)
        x_att = (x_att - x_att_min + 0.025) / (x_att_max - x_att_min + 0.025)  # n,s,1 gait3d:0.025 grew: 0,01
        return x_att

    def attention(self, x):
        n, c, s, h, w = x.size()
        x = x.permute(0, 2, 1, 3, 4).reshape(n, s, c * h * w)
        x = x * self.x_att
        x = x.reshape(n, s, c, h, w).permute(0, 2, 1, 3, 4)
        return x

    def forward(self, x, n=None, s=30):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        if self.maxpool_flag:
            x = self.maxpool(x)


        x = self.layer1(x)
        x = self.layer2(x)
        bs = x.shape[0] // s
        x = x.view(bs, x.shape[0] // bs, x.shape[1], x.shape[2], x.shape[3])
        x = x.permute(0, 2, 1, 3, 4)
        self.x_att = self.fm(x)
        x = self.attention(x)

        x, self.x_att = self.layer3((x, self.x_att))
        x = self.attention(x)

        x, self.x_att = self.layer4((x, self.x_att))
        x = self.attention(x)

        x = x.permute(0, 2, 1, 3, 4)
        x = x.reshape(x.shape[0] * x.shape[1], x.shape[2], x.shape[3], x.shape[4])
        return x, self.x_att

    def conv1x1(self, in_planes: int, out_planes: int, stride: int = 1) -> nn.Conv2d:
        """1x1 convolution"""
        return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)

    def _initialize_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm3d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias.data, 0.0)


