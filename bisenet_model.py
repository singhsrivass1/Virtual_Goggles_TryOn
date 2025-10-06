# models/bisenet_model.py (final clean version for CelebAMask-HQ weights)

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet18


# ---------- building blocks ----------

class ConvBNReLU(nn.Module):
    def __init__(self, in_chan, out_chan, ks=3, stride=1, padding=1):
        super(ConvBNReLU, self).__init__()
        self.conv = nn.Conv2d(in_chan, out_chan, ks, stride, padding, bias=False)
        self.bn = nn.BatchNorm2d(out_chan)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu(self.bn(self.conv(x)))


class AttentionRefinementModule(nn.Module):
    def __init__(self, in_chan, out_chan):
        super(AttentionRefinementModule, self).__init__()
        self.conv = ConvBNReLU(in_chan, out_chan, ks=3, stride=1, padding=1)
        self.conv_atten = nn.Conv2d(out_chan, out_chan, 1)
        self.bn_atten = nn.BatchNorm2d(out_chan)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        feat = self.conv(x)
        atten = F.adaptive_avg_pool2d(feat, (1, 1))
        atten = self.conv_atten(atten)
        atten = self.bn_atten(atten)
        atten = self.sigmoid(atten)
        return torch.mul(feat, atten)


class FeatureFusionModule(nn.Module):
    def __init__(self, in_chan, out_chan):
        super(FeatureFusionModule, self).__init__()
        self.convblk = ConvBNReLU(in_chan, out_chan, ks=1, stride=1, padding=0)
        self.conv1 = nn.Conv2d(out_chan, out_chan // 4, 1)
        self.conv2 = nn.Conv2d(out_chan // 4, out_chan, 1)
        self.relu = nn.ReLU(inplace=True)
        self.sigmoid = nn.Sigmoid()

    def forward(self, sp, cp):
        feat = torch.cat([sp, cp], dim=1)
        feat = self.convblk(feat)
        atten = F.adaptive_avg_pool2d(feat, (1, 1))
        atten = self.conv1(atten)
        atten = self.relu(atten)
        atten = self.conv2(atten)
        atten = self.sigmoid(atten)
        return feat + torch.mul(feat, atten)


class BiSeNetOutput(nn.Module):
    def __init__(self, in_chan, mid_chan, n_classes):
        super(BiSeNetOutput, self).__init__()
        self.conv = ConvBNReLU(in_chan, mid_chan, ks=3, stride=1, padding=1)
        self.conv_out = nn.Conv2d(mid_chan, n_classes, 1)

    def forward(self, x):
        return self.conv_out(self.conv(x))


# ---------- complete BiSeNet ----------

class BiSeNet(nn.Module):
    def __init__(self, n_classes=19):
        super(BiSeNet, self).__init__()
        self.cp = ContextPath()
        self.ffm = FeatureFusionModule(256, 256)
        self.conv_out = BiSeNetOutput(256, 256, n_classes)
        self.conv_out16 = BiSeNetOutput(128, 64, n_classes)
        self.conv_out32 = BiSeNetOutput(128, 64, n_classes)

    def forward(self, x):
        feat_res8, feat_res16, feat_cp8, feat_cp16 = self.cp(x)
        feat_fuse = self.ffm(feat_res8, feat_cp8)

        out = self.conv_out(feat_fuse)
        out16 = self.conv_out16(feat_cp8)
        out32 = self.conv_out32(feat_cp16)

        out = F.interpolate(out, size=x.size()[2:], mode='bilinear', align_corners=True)
        out16 = F.interpolate(out16, size=x.size()[2:], mode='bilinear', align_corners=True)
        out32 = F.interpolate(out32, size=x.size()[2:], mode='bilinear', align_corners=True)

        return out, out16, out32


class ContextPath(nn.Module):
    def __init__(self):
        super(ContextPath, self).__init__()
        resnet = resnet18(pretrained=False)
        self.resnet = resnet

        self.arm16 = AttentionRefinementModule(256, 128)
        self.arm32 = AttentionRefinementModule(512, 128)

        self.conv_head32 = ConvBNReLU(128, 128, ks=3, stride=1, padding=1)
        self.conv_head16 = ConvBNReLU(128, 128, ks=3, stride=1, padding=1)

        self.conv_avg = ConvBNReLU(512, 128, ks=1, stride=1, padding=0)

    def forward(self, x):
        feat2 = self.resnet.conv1(x)
        feat2 = self.resnet.bn1(feat2)
        feat2 = self.resnet.relu(feat2)
        feat2 = self.resnet.maxpool(feat2)

        feat4 = self.resnet.layer1(feat2)
        feat8 = self.resnet.layer2(feat4)
        feat16 = self.resnet.layer3(feat8)
        feat32 = self.resnet.layer4(feat16)

        avg = F.adaptive_avg_pool2d(feat32, (1, 1))
        avg = self.conv_avg(avg)
        avg_up = F.interpolate(avg, size=feat32.size()[2:], mode='bilinear', align_corners=True)

        feat32_arm = self.arm32(feat32)
        feat32_sum = feat32_arm + avg_up
        feat32_up = F.interpolate(feat32_sum, size=feat16.size()[2:], mode='bilinear', align_corners=True)
        feat32_up = self.conv_head32(feat32_up)

        feat16_arm = self.arm16(feat16)
        feat16_sum = feat16_arm + feat32_up
        feat16_up = F.interpolate(feat16_sum, size=feat8.size()[2:], mode='bilinear', align_corners=True)
        feat16_up = self.conv_head16(feat16_up)

        return feat8, feat16, feat16_up, feat32_up
