import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from einops.layers.torch import Rearrange
import math

class ResidualBlock1(nn.Module):
    """Residual Block with instance normalization."""
    def __init__(self, dim_in, dim_out):
        super(ResidualBlock1, self).__init__()
        self.main = nn.Sequential(
            nn.Conv2d(dim_in, dim_out, kernel_size=3, stride=1, padding=1, bias=False),
            nn.InstanceNorm2d(dim_out, affine=True, track_running_stats=True),
            nn.ReLU(inplace=True)
        )
        self.sc = nn.Sequential(
            nn.Conv2d(dim_out, dim_out, kernel_size=3, stride=1, padding=1, bias=False),
            nn.InstanceNorm2d(dim_out, affine=True, track_running_stats=True)
        )
    def forward(self, x):
        s_c = self.main(x)
        y = self.sc(s_c)
        return s_c + y


class Generator(nn.Module):
    """Generator network."""

    def __init__(self, conv_dim=64, c_dim=5, repeat_num=6,device = 'cuda:1'):
        super(Generator, self).__init__()


        self.layer1 = nn.Sequential(
            nn.Conv2d(1 + c_dim, conv_dim, kernel_size=7, stride=1, padding=3, bias=False),
            nn.InstanceNorm2d(conv_dim, affine=True, track_running_stats=True),
            nn.ReLU(inplace=True)
        )

        # Down-sampling layers.
        curr_dim = conv_dim
        self.enc_layer1 = nn.Sequential(
            nn.Conv2d(curr_dim, curr_dim * 2, kernel_size=4, stride=2, padding=1, bias=False),
            nn.InstanceNorm2d(curr_dim * 2, affine=True, track_running_stats=True),
            nn.ReLU(inplace=True)
        )
        curr_dim = curr_dim * 2


        self.enc_layer2 = nn.Sequential(
            nn.Conv2d(curr_dim, curr_dim * 2, kernel_size=4, stride=2, padding=1, bias=False),
            nn.InstanceNorm2d(curr_dim * 2, affine=True, track_running_stats=True),
            nn.ReLU(inplace=True)
        )
        curr_dim = curr_dim * 2

        self.enc_layer3 = nn.Sequential(
            nn.Conv2d(curr_dim, curr_dim * 2, kernel_size=4, stride=2, padding=1, bias=False),
            nn.InstanceNorm2d(curr_dim * 2, affine=True, track_running_stats=True),
            nn.ReLU(inplace=True)
        )
        curr_dim = curr_dim * 2




        # Up-sampling layers.
        self.gate1 = AttentionGate(F_g=curr_dim, F_l=curr_dim, n_coefficients=curr_dim // 2)
        self.dec_bn1 = ResidualBlock1(dim_in=2 * curr_dim, dim_out=curr_dim)
        self.dec_layer1 = nn.Sequential(
            nn.ConvTranspose2d(curr_dim, curr_dim // 2, kernel_size=4, stride=2, padding=1, bias=False),
            nn.InstanceNorm2d(curr_dim // 2, affine=True, track_running_stats=True),
            nn.ReLU(inplace=True)
        )
        curr_dim = curr_dim // 2

        self.gate2 = AttentionGate(F_g=curr_dim, F_l=curr_dim, n_coefficients=curr_dim // 2)
        self.dec_bn2 = ResidualBlock1(dim_in=2 * curr_dim, dim_out=curr_dim)
        self.dec_layer2 = nn.Sequential(
            nn.ConvTranspose2d(curr_dim, curr_dim // 2, kernel_size=4, stride=2, padding=1, bias=False),
            nn.InstanceNorm2d(curr_dim // 2, affine=True, track_running_stats=True),
            nn.ReLU(inplace=True)
        )
        curr_dim = curr_dim // 2


        self.gate3 = AttentionGate(F_g=curr_dim, F_l=curr_dim, n_coefficients=curr_dim // 2)
        self.dec_bn3 = ResidualBlock1(dim_in=2 * curr_dim, dim_out=curr_dim)
        self.dec_layer3 = nn.Sequential(
            nn.ConvTranspose2d(curr_dim, curr_dim // 2, kernel_size=4, stride=2, padding=1, bias=False),
            nn.InstanceNorm2d(curr_dim // 2, affine=True, track_running_stats=True),
            nn.ReLU(inplace=True)
        )
        curr_dim = curr_dim // 2
        # self.bn3 = ResidualBlock1(dim_in=3 * curr_dim, dim_out=curr_dim)
        self.conv = nn.Conv2d(curr_dim, 1, kernel_size=7, stride=1, padding=3, bias=False)

        self.fusion1 = MEFusion1(norm=nn.BatchNorm2d, act=nn.ReLU)
        self.fusion2 = MEFusion2(norm=nn.BatchNorm2d, act=nn.ReLU)
    def forward(self, x,bs=1):
        y = self.layer1(x)
        y1 = self.enc_layer1(y)
        sc = []
        sc.append(y1)
        y2 = self.enc_layer2(y1)

        sc.append(y2)
        y3 = self.enc_layer3(y2)
        sc.append(y3)
        encoder_out = y3

        fused_feature = self.fusion1([y1, y2, y3])

        i = 0
        sc1 = self.gate1(gate=fused_feature, skip_connection=fused_feature)
        out1 = torch.concat([fused_feature, sc1], dim=1)
        out1 = self.dec_bn1(out1)
        out1 = self.dec_layer1(out1 )

        i += 1
        sc2 = self.gate2(gate=out1, skip_connection=sc[-1 - i])
        out2 = torch.concat([out1, sc2], dim=1)
        out2 = self.dec_bn2(out2)
        out2 = self.dec_layer2(out2)

        i += 1
        sc3 = self.gate3(gate=out2, skip_connection=sc[-1 - i])
        out3 = torch.concat([out2, sc3], dim=1)
        out3 = self.dec_bn3(out3)
        out3 = self.dec_layer3(out3)



        fused_feature_2 = self.fusion2([out1, out2 , out3])

        output = self.conv(fused_feature_2)
        if self.training:
            # spatial binary mask
            mask = torch.ones(output.size(0), 1, output.size(-2), output.size(-1)).to(output.device) * 0.95
            mask = torch.bernoulli(mask).float()
            output = mask * output + (1. - mask) * x
        fake1 = torch.tanh(output + x)
        return fake1


class MEFusion1(nn.Module):
    def __init__(self, norm, act):
        super().__init__()
        self.fusi_conv = nn.Sequential(
            nn.Conv2d(896, 512, 1, bias=False),
            norm(512),
            act(),
        )

        self.attn_conv = nn.ModuleList()
        self.reduce_conv = nn.ModuleList()
        for i in range(3):
            self.reduce_conv.append(nn.Conv2d(
                [128, 256, 512][i], 512, 1, bias=False))
            self.attn_conv.append(nn.Sequential(
                nn.Conv2d(512, 512, 1, bias=False),
                norm(512),
                act(),
            ))

        self.pool = nn.AdaptiveAvgPool2d(1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, feature_list):
        target_size = feature_list[-1].size()[2:]
        feature_list = [F.interpolate(feat, size=target_size, mode='bilinear', align_corners=False) for feat in
                        feature_list]
        fusi_feature = torch.cat(feature_list, dim=1).contiguous()
        fusi_feature = self.fusi_conv(fusi_feature)
        for i in range(3):
            x = feature_list[i]

            x = self.reduce_conv[i](x)
            attn = self.attn_conv[i](fusi_feature)#修改过
            attn = self.pool(attn)
            attn = self.sigmoid(attn)
            x = attn * x + x
            feature_list[i] = x
        return feature_list[0] + feature_list[1] + feature_list[2]

class MEFusion2(nn.Module):
    def __init__(self, norm, act):
        super().__init__()
        self.fusi_conv = nn.Sequential(
            nn.Conv2d(448, 64, 1, bias=False),
            norm(64),
            act(),
        )

        self.attn_conv = nn.ModuleList()
        self.reduce_conv = nn.ModuleList()
        for i in range(3):
            self.reduce_conv.append(nn.Conv2d(
                [256, 128, 64][i], 64, 1, bias=False))
            self.attn_conv.append(nn.Sequential(
                nn.Conv2d(64, 64, 1, bias=False),
                norm(64),
                act(),
            ))

        self.pool = nn.AdaptiveAvgPool2d(1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, feature_list):
        target_size = feature_list[-1].size()[2:]
        feature_list = [F.interpolate(feat, size=target_size, mode='bilinear', align_corners=False) for feat in
                        feature_list]
        fusi_feature = torch.cat(feature_list, dim=1).contiguous()
        fusi_feature = self.fusi_conv(fusi_feature)
        for i in range(3):
            x = feature_list[i]
            x = self.reduce_conv[i](x)
            attn = self.attn_conv[i](fusi_feature)#修改过
            attn = self.pool(attn)
            attn = self.sigmoid(attn)
            x = attn * x + x
            feature_list[i] = x
        return feature_list[0] + feature_list[1] + feature_list[2]
class Discriminator(nn.Module):
    """Discriminator network with PatchGAN."""

    def __init__(self, image_size=128, conv_dim=64, c_dim=5, repeat_num=6):
        super(Discriminator, self).__init__()
        layers = []
        layers.append(nn.Conv2d(1, conv_dim, kernel_size=4, stride=2, padding=1))
        layers.append(nn.LeakyReLU(0.01))

        curr_dim = conv_dim
        for i in range(1, repeat_num):
            layers.append(nn.Conv2d(curr_dim, curr_dim * 2, kernel_size=4, stride=2, padding=1))
            layers.append(nn.LeakyReLU(0.01))
            curr_dim = curr_dim * 2

        kernel_size = int(image_size / np.power(2, repeat_num))
        self.main = nn.Sequential(*layers)
        self.conv1 = nn.Conv2d(curr_dim, 1, kernel_size=3, stride=1, padding=1, bias=False)

    def forward(self, x):
        h = self.main(x)
        out_src = self.conv1(h)
        return h, out_src


def conv3x3(in_planes: int, out_planes: int, stride: int = 1, groups: int = 1, dilation: int = 1) -> nn.Conv2d:
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False, dilation=dilation)




class AMA(nn.Module):
    def __init__(self, channels, c2=None, factor=32):
        super(AMA, self).__init__()
        self.groups = factor
        assert channels // self.groups > 0
        self.softmax = nn.Softmax(-1)
        self.agp = nn.AdaptiveAvgPool2d((1, 1))
        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))
        self.gn = nn.GroupNorm(channels // self.groups, channels // self.groups)
        self.conv1x1 = nn.Conv2d(channels // self.groups, channels // self.groups, kernel_size=1, stride=1, padding=0)
        self.conv3x3 = nn.Conv2d(channels // self.groups, channels // self.groups, kernel_size=3, stride=1, padding=1)
        # 新增的异常区域增强卷积
        self.abnormal_enhancer = nn.Conv2d(channels // self.groups, channels // self.groups, kernel_size=1)

    def channel_shuffle(self, x , groups):
        b, c, h, w = x.size()
        channels_per_group = c // groups
        x = x.view(b, groups, channels_per_group, h, w)
        x = x.permute(0, 2, 1, 3, 4).contiguous()
        x = x.view(b, c, h, w)
        return x
    def forward(self, x):
        b, c, h, w = x.size()
        group_x = x.reshape(b * self.groups, -1, h, w)
        x_h = self.pool_h(group_x)
        x_w = self.pool_w(group_x).permute(0, 1, 3, 2)
        hw = self.conv1x1(torch.cat([x_h, x_w], dim=2))
        x_h, x_w = torch.split(hw, [h, w], dim=2)
        x1 = self.gn(group_x * x_h.sigmoid() * x_w.permute(0, 1, 3, 2).sigmoid())
        x2 = self.conv3x3(group_x)
        abnormal_weight = self.abnormal_enhancer(x1).sigmoid()
        x1 = x1 * abnormal_weight + x1
        x11 = self.softmax(self.agp(x1).reshape(b * self.groups, -1, 1).permute(0, 2, 1))
        x12 = x2.reshape(b * self.groups, c // self.groups, -1)
        x21 = self.softmax(self.agp(x2).reshape(b * self.groups, -1, 1).permute(0, 2, 1))
        x22 = x1.reshape(b * self.groups, c // self.groups, -1)
        weights = (torch.matmul(x11, x12) + torch.matmul(x21, x22)).reshape(b * self.groups, 1, h, w)
        reweighted_output = (group_x * weights.sigmoid()).reshape(b, c, h, w)
        reweighted_output = self.channel_shuffle(reweighted_output, groups=2)
        return reweighted_output

class PixelWeight(nn.Module):
    def __init__(self, F_g, F_l, n_coefficients):
        super(PixelWeight, self).__init__()
        self.pix_weight1 = nn.Sequential(
            nn.Conv2d(F_g, F_l, kernel_size=1, stride=1, bias=True),
            nn.InstanceNorm2d(F_l)
        )

        self.ama = AMA(F_l)
        self.pix_weight2 = nn.Sequential(
            nn.Conv2d(F_l, F_g, kernel_size=1, stride=1, bias=True),
            nn.InstanceNorm2d(F_g)
        )


    def forward(self, gate, skip_connection):
        x1 = self.attention(skip_connection)
        skip_weight = self.pix_weight1(x1)
        combined_weight = skip_weight + gate
        combined_weight_2 = self.pix_weight2(combined_weight)  # 相加的权重进行卷积操作，再降维度
        combined_weight_normalize = self.norm(combined_weight_2, combined_weight_2.size())
        return combined_weight_normalize

    def attention(self, x):

        x = self.ama(x)
        return x
    def norm(self, x, shape):
        assert x.shape == shape, "x 的形状与传递的 shape 不匹配"
        num_elements = torch.prod(torch.tensor(shape))
        assert x.numel() == num_elements, "元素总数不匹配"
        x = x.view(shape[0], -1, shape[2] * shape[3])
        x = torch.softmax(x, dim=-1)
        x = x.view(shape)
        return x

class AttentionGate(nn.Module):
    def __init__(self, F_g, F_l, n_coefficients):
        super(AttentionGate, self).__init__()
        self.pixel_weight = PixelWeight(F_g=F_g, F_l=F_l, n_coefficients=n_coefficients)
    def forward(self, gate, skip_connection):
        out = self.pixel_weight(gate, skip_connection)
        return out
