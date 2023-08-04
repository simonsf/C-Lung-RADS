# Copyright 2022 Dakewe Biotech Corporation. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 (the "License");
#   you may not use this file except in compliance with the License.
#   You may obtain a copy of the License at
#
#       http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
from typing import Any, List, Type, Union, Optional

import torch
from torch import Tensor
from torch import nn
import torch.nn.functional as F

import math

__all__ = [
    "ResNet",
    "resnet18",
]


def refine_cams(cam_original, image_shape, cam_w, cam_sigma, using_sigmoid=True):
    cam_original = F.interpolate(
        cam_original, image_shape, mode="trilinear", align_corners=True
    )
    B, C, D, H, W = cam_original.size()
    cams = []
    for idx in range(C):
        cam = cam_original[:, idx, :, :, :]
        cam = cam.view(B, -1)
        cam_min = cam.min(dim=1, keepdim=True)[0]
        cam_max = cam.max(dim=1, keepdim=True)[0]
        norm = cam_max - cam_min
        norm[norm == 0] = 1e-5
        cam = (cam - cam_min) / norm       
        cam = cam.view(B, D, H, W).unsqueeze(1)
        cams.append(cam)
    cams = torch.cat(cams, dim=1)
    if using_sigmoid:
        cams = torch.sigmoid(cam_w*(cams - cam_sigma))
    return cams


class Linear(nn.Module):
    def __init__(self, in_features, out_features, bias=True):
        super(Linear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.Tensor(out_features, in_features))
        if bias:
            self.bias = nn.Parameter(torch.Tensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input):
        return F.linear(input, self.weight, self.bias), self.weight

    def extra_repr(self):
        return 'in_features={}, out_features={}, bias={}'.format(
            self.in_features, self.out_features, self.bias is not None
        )


class _BasicBlock(nn.Module):
    expansion: int = 1

    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            stride: int,
            downsample: Optional[nn.Module] = None,
            groups: int = 1,
            base_channels: int = 64,
    ) -> None:
        super(_BasicBlock, self).__init__()
        self.stride = stride
        self.downsample = downsample
        self.groups = groups
        self.base_channels = base_channels

        self.conv1 = nn.Conv3d(in_channels, out_channels, (3, 3, 3), (stride, stride, stride), (1, 1, 1), bias=False)
        self.bn1 = nn.BatchNorm3d(out_channels)
        self.relu = nn.ReLU(True)
        self.conv2 = nn.Conv3d(out_channels, out_channels, (3, 3, 3), (1, 1, 1), (1, 1, 1), bias=False)
        self.bn2 = nn.BatchNorm3d(out_channels)

    def forward(self, x: Tensor) -> Tensor:
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out = torch.add(out, identity)
        out = self.relu(out)

        return out


class _Bottleneck(nn.Module):
    expansion: int = 4

    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            stride: int,
            downsample: Optional[nn.Module] = None,
            groups: int = 1,
            base_channels: int = 64,
    ) -> None:
        super(_Bottleneck, self).__init__()
        self.stride = stride
        self.downsample = downsample
        self.groups = groups
        self.base_channels = base_channels

        channels = int(out_channels * (base_channels / 64.0)) * groups

        self.conv1 = nn.Conv3d(in_channels, channels, (1, 1, 1), (1, 1, 1), (0, 0, 0), bias=False)
        self.bn1 = nn.BatchNorm3d(channels)
        self.conv2 = nn.Conv3d(channels, channels, (3, 3, 3), (stride, stride, stride), (1, 1, 1), groups=groups, bias=False)
        self.bn2 = nn.BatchNorm3d(channels)
        self.conv3 = nn.Conv3d(channels, int(out_channels * self.expansion), (1, 1, 1), (1, 1, 1), (0, 0, 0), bias=False)
        self.bn3 = nn.BatchNorm3d(int(out_channels * self.expansion))
        self.relu = nn.ReLU(True)

    def forward(self, x: Tensor) -> Tensor:
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out = torch.add(out, identity)
        out = self.relu(out)

        return out


class ResNet(nn.Module):

    def __init__(
            self,
            arch_cfg: List[int],
            block: Type[Union[_BasicBlock, _Bottleneck]],
            groups: int = 1,
            channels_per_group: int = 64,
            num_classes: int = 1000,
            input_channels: int = 1,
    ) -> None:
        super(ResNet, self).__init__()
        self.in_channels = 64
        self.dilation = 1
        self.groups = groups
        self.base_channels = channels_per_group

        self.conv1 = nn.Conv3d(input_channels, self.in_channels, (7, 7, 7), (2, 2, 2), (3, 3, 3), bias=False)
        self.bn1 = nn.BatchNorm3d(self.in_channels)
        self.relu = nn.ReLU(True)
        self.maxpool = nn.MaxPool3d((3, 3, 3), (2, 2, 2), (1, 1, 1))

        self.layer1 = self._make_layer(arch_cfg[0], block, 64, 1)
        self.layer2 = self._make_layer(arch_cfg[1], block, 128, 2)
        self.layer3 = self._make_layer(arch_cfg[2], block, 256, 2)
        self.layer4 = self._make_layer(arch_cfg[3], block, 512, 2)

        self.avgpool = nn.AdaptiveAvgPool3d((1, 1, 1))

        self.fc = Linear(512 * block.expansion, num_classes, bias=False)

        # Initialize neural network weights
        self._initialize_weights()

    def _make_layer(
            self,
            repeat_times: int,
            block: Type[Union[_BasicBlock, _Bottleneck]],
            channels: int,
            stride: int = 1,
    ) -> nn.Sequential:
        downsample = None

        if stride != 1 or self.in_channels != channels * block.expansion:
            downsample = nn.Sequential(
                nn.Conv3d(self.in_channels, channels * block.expansion, (1, 1, 1), (stride, stride, stride), (0, 0, 0), bias=False),
                nn.BatchNorm3d(channels * block.expansion),
            )

        layers = [
            block(
                self.in_channels,
                channels,
                stride,
                downsample,
                self.groups,
                self.base_channels
            )
        ]
        self.in_channels = channels * block.expansion
        for _ in range(1, repeat_times):
            layers.append(
                block(
                    self.in_channels,
                    channels,
                    1,
                    None,
                    self.groups,
                    self.base_channels,
                )
            )

        return nn.Sequential(*layers)

    def forward(self, x: Tensor) -> Tensor:
        out, cam = self._forward_impl(x)

        return out, cam

    # Support torch.script function
    def _forward_impl(self, x: Tensor) -> Tensor:
        _, _, D, H, W = x.size()
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.maxpool(out)

        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        feature = self.layer4(out)

        avg_featrue = self.avgpool(feature)
        avg_featrue = avg_featrue.view(avg_featrue.size(0), -1)
        out, fc_w = self.fc(avg_featrue)

        cam_classes = self.relu(F.conv3d(feature, fc_w.detach().unsqueeze(2).unsqueeze(3).unsqueeze(4), bias=None, stride=1, padding=0))
        cam_classes_refined = refine_cams(cam_classes, (D, H, W), 100, 0.4, using_sigmoid=True)

        return out, cam_classes_refined

    def _initialize_weights(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Conv3d):
                nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(module, (nn.BatchNorm3d, nn.GroupNorm)):
                nn.init.constant_(module.weight, 1)
                nn.init.constant_(module.bias, 0)


def resnet18(**kwargs: Any) -> ResNet:
    model = ResNet([2, 2, 2, 2], _BasicBlock, **kwargs)

    return model


def resnet34(**kwargs: Any) -> ResNet:
    model = ResNet([3, 4, 6, 3], _BasicBlock, **kwargs)

    return model


def resnet50(**kwargs: Any) -> ResNet:
    model = ResNet([3, 4, 6, 3], _Bottleneck, **kwargs)

    return model


def resnet101(**kwargs: Any) -> ResNet:
    model = ResNet([3, 4, 23, 3], _Bottleneck, **kwargs)

    return model


def resnet152(**kwargs: Any) -> ResNet:
    model = ResNet([3, 8, 36, 3], _Bottleneck, **kwargs)

    return model