import torch
import torch.nn as nn
from torch.nn import functional as F


class InPlaceActivatedBatchNorm(nn.Module):
    def __init__(self, num_features):
        super().__init__()
        self.bn = nn.BatchNorm2d(num_features)
        self.leaky_relu = nn.LeakyReLU(inplace=True)

    def forward(self, x):
        x = self.bn(x)
        x = self.leaky_relu(x)
        return x


class PSPModule(nn.Module):
    """
    Reference:
        Zhao, Hengshuang, et al. *"Pyramid scene parsing network."*
    """

    def __init__(self, features=2048, out_features=512, sizes=(1, 2, 3, 6)):
        super().__init__()

        self.stages = []
        self.stages = nn.ModuleList([self._make_stage(features, out_features, size) for size in sizes])
        self.bottleneck = nn.Sequential(
            nn.Conv2d(features + len(sizes) * out_features, out_features, kernel_size=3, padding=1, dilation=1,
                      bias=False),
            InPlaceActivatedBatchNorm(out_features),
        )

    def _make_stage(self, features, out_features, size):
        prior = nn.AdaptiveAvgPool2d(output_size=(size, size))
        conv = nn.Conv2d(features, out_features, kernel_size=1, bias=False)
        bn = InPlaceActivatedBatchNorm(out_features)
        return nn.Sequential(prior, conv, bn)

    def forward(self, feats):
        h, w = feats.size(2), feats.size(3)
        priors = [F.interpolate(input=stage(feats), size=(h, w), mode='bilinear', align_corners=True) for stage in
                  self.stages] + [feats]
        bottle = self.bottleneck(torch.cat(priors, 1))
        return bottle


class DecoderModule(nn.Module):
    """
    Parsing Branch Decoder Module.

    """

    def __init__(self, num_classes):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(512, 256, kernel_size=1, padding=0, dilation=1, bias=False),
            InPlaceActivatedBatchNorm(256)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(256, 48, kernel_size=1, stride=1, padding=0, dilation=1, bias=False),
            InPlaceActivatedBatchNorm(48)
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(304, 256, kernel_size=1, padding=0, dilation=1, bias=False),
            InPlaceActivatedBatchNorm(256),
            nn.Conv2d(256, 256, kernel_size=1, padding=0, dilation=1, bias=False),
            InPlaceActivatedBatchNorm(256)
        )

        self.conv4 = nn.Conv2d(256, num_classes, kernel_size=1, padding=0, dilation=1, bias=True)

    def forward(self, xt, xl):
        _, _, h, w = xl.size()
        xt = F.interpolate(self.conv1(xt), size=(h, w), mode='bilinear', align_corners=True)
        xl = self.conv2(xl)
        x = torch.cat([xt, xl], dim=1)
        x = self.conv3(x)
        seg = self.conv4(x)
        return seg, x


class EdgeModule(nn.Module):
    def __init__(self, in_fea=[256, 512, 1024], mid_fea=256, out_fea=2):
        super().__init__()

        self.conv1 = nn.Sequential(
            nn.Conv2d(in_fea[0], mid_fea, kernel_size=1, padding=0, dilation=1, bias=False),
            InPlaceActivatedBatchNorm(mid_fea)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_fea[1], mid_fea, kernel_size=1, padding=0, dilation=1, bias=False),
            InPlaceActivatedBatchNorm(mid_fea)
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(in_fea[2], mid_fea, kernel_size=1, padding=0, dilation=1, bias=False),
            InPlaceActivatedBatchNorm(mid_fea)
        )
        self.conv4 = nn.Conv2d(mid_fea, out_fea, kernel_size=3, padding=1, dilation=1, bias=True)
        self.conv5 = nn.Conv2d(out_fea * 3, out_fea, kernel_size=1, padding=0, dilation=1, bias=True)

    def forward(self, x1, x2, x3):
        _, _, h, w = x1.size()

        edge1_fea = self.conv1(x1)
        edge1 = self.conv4(edge1_fea)
        edge2_fea = self.conv2(x2)
        edge2 = self.conv4(edge2_fea)
        edge3_fea = self.conv3(x3)
        edge3 = self.conv4(edge3_fea)

        edge2_fea = F.interpolate(edge2_fea, size=(h, w), mode='bilinear', align_corners=True)
        edge3_fea = F.interpolate(edge3_fea, size=(h, w), mode='bilinear', align_corners=True)
        edge2 = F.interpolate(edge2, size=(h, w), mode='bilinear', align_corners=True)
        edge3 = F.interpolate(edge3, size=(h, w), mode='bilinear', align_corners=True)

        edge = torch.cat([edge1, edge2, edge3], dim=1)
        edge_fea = torch.cat([edge1_fea, edge2_fea, edge3_fea], dim=1)
        edge = self.conv5(edge)

        return edge, edge_fea


class ResNet(nn.Module):
    def __init__(self, block, layers, num_classes):

        self.inplanes = 128

        super().__init__()
        self.conv1 = conv3x3(3, 64, stride=2)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu1 = nn.ReLU(inplace=False)
        self.conv2 = conv3x3(64, 64)
        self.bn2 = nn.BatchNorm2d(64)
        self.relu2 = nn.ReLU(inplace=False)
        self.conv3 = conv3x3(64, 128)
        self.bn3 = nn.BatchNorm2d(128)
        self.relu3 = nn.ReLU(inplace=False)

        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=1, dilation=2)

        self.context_encoding = PSPModule()
        self.edge = EdgeModule()
        self.decoder = DecoderModule(num_classes)

        self.fusion = nn.Sequential(
            nn.Conv2d(1024, 256, kernel_size=1, padding=0, dilation=1, bias=False),
            InPlaceActivatedBatchNorm(256),
            nn.Dropout2d(0.1),
            nn.Conv2d(256, num_classes, kernel_size=1, padding=0, dilation=1, bias=True)
        )

    def _make_layer(self, block, planes, blocks, stride=1, dilation=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion)
            )

        layers = [block(self.inplanes, planes, stride, dilation=dilation, downsample=downsample)]

        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(
                block(self.inplanes, planes, dilation=dilation))

        return nn.Sequential(*layers)

    def forward(self, x):
        # Parsing Branch
        x = self.relu1(self.bn1(self.conv1(x)))
        x = self.relu2(self.bn2(self.conv2(x)))
        x = self.relu3(self.bn3(self.conv3(x)))
        x1 = self.maxpool(x)
        x2 = self.layer1(x1)
        x3 = self.layer2(x2)
        x4 = self.layer3(x3)
        x5 = self.layer4(x4)
        x = self.context_encoding(x5)
        parsing_result, parsing_fea = self.decoder(x, x2)
        # Edge Branch
        edge_result, edge_fea = self.edge(x2, x3, x4)
        # Fusion Branch
        x = torch.cat([parsing_fea, edge_fea], dim=1)
        fusion_result = self.fusion(x)
        return fusion_result


def conv3x3(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)