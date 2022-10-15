"""Create model."""
# coding=utf-8
#
# /************************************************************************************
# ***
# ***    Copyright 2022 Dell(18588220928@163.com), All Rights Reserved.
# ***
# ***    File Author: Dell, 2022年 09月 25日 星期日 11:31:05 CST
# ***
# ************************************************************************************/
#
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from typing import List
import math
import pdb


def conv_bn(inp, oup, stride):
    return nn.Sequential(nn.Conv2d(inp, oup, 3, stride, 1, bias=False), nn.BatchNorm2d(oup), nn.ReLU(inplace=True))


def conv_1x1_bn(inp, oup):
    return nn.Sequential(nn.Conv2d(inp, oup, 1, 1, 0, bias=False), nn.BatchNorm2d(oup), nn.ReLU(inplace=True))


class InvertedResidual(nn.Module):
    def __init__(self, inp, oup, stride, expand_ratio):
        super(InvertedResidual, self).__init__()
        self.stride = stride
        assert stride in [1, 2]

        self.use_res_connect = self.stride == 1 and inp == oup

        self.conv = nn.Sequential(
            # pw
            nn.Conv2d(inp, inp * expand_ratio, 1, 1, 0, bias=False),
            nn.BatchNorm2d(inp * expand_ratio),
            nn.ReLU6(inplace=True),
            # dw
            nn.Conv2d(inp * expand_ratio, inp * expand_ratio, 3, stride, 1, groups=inp * expand_ratio, bias=False),
            nn.BatchNorm2d(inp * expand_ratio),
            nn.ReLU6(inplace=True),
            # pw-linear
            nn.Conv2d(inp * expand_ratio, oup, 1, 1, 0, bias=False),
            nn.BatchNorm2d(oup),
        )

    def forward(self, x):
        if self.use_res_connect:
            return x + self.conv(x)
        else:
            return self.conv(x)


class MobileNetV2(nn.Module):
    def __init__(self, n_class=1000, input_size=224, width_mult=1.0):
        super(MobileNetV2, self).__init__()
        # setting of inverted residual blocks
        self.interverted_residual_setting = [
            # t, c, n, s
            [1, 16, 1, 1],
            [6, 24, 2, 2],
            [6, 32, 3, 2],
            [6, 64, 4, 2],
            [6, 96, 3, 1],
            [6, 160, 3, 2],
            [6, 320, 1, 1],
        ]

        # building first layer
        assert input_size % 32 == 0
        input_channel = int(32 * width_mult)
        self.last_channel = int(1280 * width_mult) if width_mult > 1.0 else 1280
        self.features = [conv_bn(3, input_channel, 2)]
        # building inverted residual blocks
        for t, c, n, s in self.interverted_residual_setting:
            output_channel = int(c * width_mult)
            for i in range(n):
                if i == 0:
                    self.features.append(InvertedResidual(input_channel, output_channel, s, t))
                else:
                    self.features.append(InvertedResidual(input_channel, output_channel, 1, t))
                input_channel = output_channel
        # building last several layers
        self.features.append(conv_1x1_bn(input_channel, self.last_channel))
        # self.features.append(nn.AvgPool2d(input_size // 32))
        # make it nn.Sequential
        self.features = nn.Sequential(*self.features)

        # avgpool
        self.avgpool = nn.AvgPool2d(input_size // 32)

        # building classifier
        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(self.last_channel, n_class),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = x.view(-1, self.last_channel)
        x = self.classifier(x)
        return x


def resnet18_places365():
    # last_model = models.resnet18(pretrained=False)
    # last_model.fc.out_features = 365

    arch = "resnet18"
    last_model = models.__dict__[arch](num_classes=365)

    # pretrained_model = "models/resnet18_places365.pth"
    # if os.path.exists(pretrained_model):
    #     checkpoint = torch.load(pretrained_model, map_location=lambda storage, loc: storage)
    #     state_dict = {str.replace(k, "module.", ""): v for k, v in checkpoint["state_dict"].items()}
    #     last_model.load_state_dict(state_dict)

    return last_model


def self_attention(x):
    batch_size, in_channels, h, w = x.size()
    query = x.view(batch_size, in_channels, -1)
    key = query
    query = query.permute(0, 2, 1)

    sim_map = torch.matmul(query, key)

    ql2 = torch.norm(query, dim=2, keepdim=True)
    kl2 = torch.norm(key, dim=1, keepdim=True)
    sim_map = torch.div(sim_map, torch.matmul(ql2, kl2).clamp(min=1e-8))

    return sim_map


def MV2():
    model = MobileNetV2()

    # pretrained_model = "models/mobilenetv2.pth"
    # if os.path.exists(pretrained_model):
    #     state_dict = torch.load(pretrained_model, map_location=lambda storage, loc: storage)
    #     model.load_state_dict(state_dict)

    # remove last classifer
    model = nn.Sequential(*list(model.children())[:-1])

    return model


class L5(nn.Module):
    def __init__(self):
        super(L5, self).__init__()
        self.base_model = MV2()

        self.head = nn.Sequential(
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.75),
            nn.Linear(1280, 10),
            # nn.Softmax(dim=1)
        )

    def forward(self, x):
        x = self.base_model(x)
        x = x.view(x.size(0), -1)
        x = self.head(x)
        return x


class L1(nn.Module):
    def __init__(self):
        super(L1, self).__init__()

        self.last_out_w = nn.Linear(365, 100)
        self.last_out_b = nn.Linear(365, 1)

    def forward(self, x) -> List[torch.Tensor]:
        out_w = self.last_out_w(x)
        out_b = self.last_out_b(x)
        return out_w, out_b


# L3
class TargetNet(nn.Module):
    def __init__(self):
        super(TargetNet, self).__init__()
        # L2
        self.fc1 = nn.Linear(365, 100)
        self.bn1 = nn.BatchNorm1d(100)
        self.relu1 = nn.PReLU()
        self.drop1 = nn.Dropout(1 - 0.5)

        self.relu7 = nn.PReLU()
        # self.sig = nn.Sigmoid()
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x, w, b):
        q = self.bn1(self.fc1(x))
        q = self.drop1(self.relu1(q))
        q = F.linear(q, w, b)
        return self.softmax(q)


class TANet(nn.Module):
    def __init__(self):
        super(TANet, self).__init__()
        # Theme Understanding Network
        self.res365_last = resnet18_places365()
        self.hypernet = L1()
        self.tygertnet = TargetNet()  # L3
        self.avg = nn.AdaptiveAvgPool2d((10, 1))

        # Aesthetics Perceiving Network
        self.mobileNet = L5()

        # RGB-distribution-aware attention Network
        self.avg_RGB = nn.AdaptiveAvgPool2d((12, 12))
        self.head_rgb = nn.Sequential(nn.ReLU(), nn.Dropout(p=0.75), nn.Linear(20736, 10), nn.Softmax(dim=1))

        self.head = nn.Sequential(nn.ReLU(), nn.Dropout(p=0.75), nn.Linear(30, 1), nn.Sigmoid())

    def forward(self, x):
        # change resize to 224x224
        x = F.interpolate(x, size=(224, 224), mode="bilinear", align_corners=False)

        # x need normalize !!!
        B, C, H, W = x.size()
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
        for i in range(3):
            x[:, i, :, :] = (x[:, i, :, :] - mean[i]) / std[i]

        # RGB-distribution-aware attention Network
        x_rgb = self.avg_RGB(x)
        x_rgb = self_attention(x_rgb)
        x_rgb = self.head_rgb(x_rgb.view(x_rgb.size(0), -1))  # size() -- [1, 10]

        # Theme Understanding Network
        last_out = self.res365_last(x)  # size() -- [1, 365]
        out_w, out_b = self.hypernet(last_out)
        x2 = self.tygertnet(last_out, out_w, out_b)
        x2 = x2.unsqueeze(dim=2)
        x2 = self.avg(x2)
        x2 = x2.squeeze(dim=2)  # size() -- [1, 10]

        # Aesthetics Perceiving Network
        x1 = self.mobileNet(x)  # size() -- [1, 10]
        x = torch.cat([x1, x2, x_rgb], dim=1)  # size() -- [1, 30]

        x = self.head(x)
        return x.clamp(0.0, 1.0) * 10.0
