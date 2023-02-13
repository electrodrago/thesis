import torch
from torch import nn as nn
from torch.nn import functional as F

from ..utils.arch_utils import ResidualBlockNoBN, flow_warp, make_layer
from .SPyNet_arch import SpyNet


class BasicVSR(nn.Module):
    """A recurrent network for video SR. Now only x4 is supported.

    Args:
        num_feat (int): Number of channels. Default: 64.
        num_block (int): Number of residual blocks for each branch. Default: 15
        spynet_path (str): Path to the pretrained weights of SPyNet. Default: None.
    """

    def __init__(self, num_feat=64, num_block=15, spynet_path=None):
        super().__init__()
        self.num_feat = num_feat

        # alignment
        self.spynet = SpyNet(spynet_path)

        # propagation
        self.backward_trunk = ConvResidualBlocks(num_feat + 3, num_feat, num_block)
        self.forward_trunk = ConvResidualBlocks(num_feat + 3, num_feat, num_block)

        # reconstruction
        self.fusion = nn.Conv2d(num_feat * 2, num_feat, 1, 1, 0, bias=True)
        self.upconv1 = nn.Conv2d(num_feat, num_feat * 4, 3, 1, 1, bias=True)
        self.upconv2 = nn.Conv2d(num_feat, 64 * 4, 3, 1, 1, bias=True)
        self.conv_hr = nn.Conv2d(64, 64, 3, 1, 1)
        self.conv_last = nn.Conv2d(64, 3, 3, 1, 1)

        self.pixel_shuffle = nn.PixelShuffle(2)

        # activation functions
        self.lrelu = nn.LeakyReLU(negative_slope=0.1, inplace=True)

    def get_flow(self, x):
        b, n, c, h, w = x.size()

        x_1 = x[:, :-1, :, :, :].reshape(-1, c, h, w)
        x_2 = x[:, 1:, :, :, :].reshape(-1, c, h, w)

        flows_backward = self.spynet(x_1, x_2).view(b, n - 1, 2, h, w)
        flows_forward = self.spynet(x_2, x_1).view(b, n - 1, 2, h, w)
        # flows_backward: [b, 14, 2, h, w]

        return flows_forward, flows_backward

    def forward(self, x):
        """Forward function of BasicVSR.

        Args:
            x: Input frames with shape (b, n, c, h, w). n is the temporal dimension / number of frames.
        """
        flows_forward, flows_backward = self.get_flow(x)
        b, n, _, h, w = x.size()

        # backward branch
        out_l = []
        feat_prop = x.new_zeros(b, self.num_feat, h, w)
        # feat_prop: [b, 64, h, w]
        for i in range(n - 1, -1, -1):
            x_i = x[:, i, :, :, :]
            # start to warp from frame 13 -> 1
            if i < n - 1:
                # flow: [b, 2, h, w]
                flow = flows_backward[:, i, :, :, :]
                feat_prop = flow_warp(feat_prop, flow.permute(0, 2, 3, 1))
            # Concat at channel dim [b, 64 + 3, h, w]
            feat_prop = torch.cat([x_i, feat_prop], dim=1)
            # Feed to residual block
            feat_prop = self.backward_trunk(feat_prop)
            # Insert at the head of list because backward branch
            out_l.insert(0, feat_prop)

        # forward branch
        feat_prop = torch.zeros_like(feat_prop)
        for i in range(0, n):
            x_i = x[:, i, :, :, :]
            if i > 0:
                flow = flows_forward[:, i - 1, :, :, :]
                feat_prop = flow_warp(feat_prop, flow.permute(0, 2, 3, 1))

            feat_prop = torch.cat([x_i, feat_prop], dim=1)
            feat_prop = self.forward_trunk(feat_prop)

            # upsample
            out = torch.cat([out_l[i], feat_prop], dim=1)
            out = self.lrelu(self.fusion(out))
            out = self.lrelu(self.pixel_shuffle(self.upconv1(out)))
            out = self.lrelu(self.pixel_shuffle(self.upconv2(out)))
            out = self.lrelu(self.conv_hr(out))
            out = self.conv_last(out)
            base = F.interpolate(x_i, scale_factor=4, mode='bilinear', align_corners=False)
            out += base
            out_l[i] = out

        return torch.stack(out_l, dim=1)


class ConvResidualBlocks(nn.Module):
    """Conv and residual block used in BasicVSR.

    Args:
        num_in_ch (int): Number of input channels. Default: 3.
        num_out_ch (int): Number of output channels. Default: 64.
        num_block (int): Number of residual blocks. Default: 15.
    """

    def __init__(self, num_in_ch=3, num_out_ch=64, num_block=15):
        super().__init__()
        self.main = nn.Sequential(
            nn.Conv2d(num_in_ch, num_out_ch, 3, 1, 1, bias=True), nn.LeakyReLU(negative_slope=0.1, inplace=True),
            make_layer(ResidualBlockNoBN, num_block, num_feat=num_out_ch))

    def forward(self, fea):
        return self.main(fea)

"""
def forward(self, x):
        flows_forward, flows_backward = self.get_flow(x)
        # b, 15, 3, h, w
        b, n, _, h, w = x.size()

        feat = self.feat_extract(x)

        # Feature propagation dict for grid, back_trunk, for_trunk
        fp_dict = dict()
        feat_prop = torch.zeros((b, self.num_feat, h, w))

        # backward branch 1
        for i in range(n - 1, -1, -1):
            feat_i = feat[:, i, :, :, :]
            # Start to warp from frame idx 13 to 0
            if i < n - 1:
                flow = flows_backward[:, i, :, :, :]
                feat_prop = flow_warp(feat_prop, flow.permute(0, 2, 3, 1))
            # Concat to dim [b, 64 * 2, h, w]
            feat_prop = torch.cat([feat_i, feat_prop], dim=1)
            feat_prop = self.backward_trunk_1(feat_prop)
            fp_dict['back_trunk_1'].insert(0, feat_prop)

        # forward branch 1
        feat_prop = torch.zeros_like(feat_prop)
        for i in range(0, n):
            feat_i = feat[:, i, :, :, :]
            # Start to warp from frame idx 0 to 13
            if i > 0:
                flow = flows_forward[:, i - 1, :, :, :]
                feat_prop = flow_warp(feat_prop, flow.permute(0, 2, 3, 1))

            feat_grid = []
            for j in fp_dict.keys():
                feat_grid.append(fp_dict[j][i])
            feat_grid = torch.cat(feat_grid, dim=1)

            # Concat to dim [b, 64 * 3, h, w]
            feat_prop = torch.cat([feat_i, feat_grid, feat_prop], dim=1)
            feat_prop = self.forward_trunk_1(feat_prop)
            fp_dict['for_trunk_1'].append(feat_prop)

        # backward branch 2
        feat_prop = torch.zeros_like(feat_prop)
        for i in range(n - 1, -1, -1):
            feat_i = feat[:, i, :, :, :]
            # Start to warp from frame idx 0 to 13
            if i < n - 1:
                flow = flows_backward[:, i - 1, :, :, :]
                feat_prop = flow_warp(feat_prop, flow.permute(0, 2, 3, 1))

            feat_grid = []
            for j in fp_dict.keys():
                feat_grid.append(fp_dict[j][i])
            feat_grid = torch.cat(feat_grid, dim=1)

            # Concat to dim [b, 64 * 4, h, w]
            feat_prop = torch.cat([feat_i, feat_grid, feat_prop], dim=1)
            feat_prop = self.backward_trunk_2(feat_prop)
            fp_dict['back_trunk_2'].insert(0, feat_prop)

        # forward branch 2
        feat_prop = torch.zeros_like(feat_prop)
        for i in range(0, n):
            feat_i = feat[:, i, :, :, :]
            # Start to warp from frame idx 0 to 13
            if i > 0:
                flow = flows_forward[:, i - 1, :, :, :]
                feat_prop = flow_warp(feat_prop, flow.permute(0, 2, 3, 1))

            feat_grid = []
            for j in fp_dict.keys():
                feat_grid.append(fp_dict[j][i])
            feat_grid = torch.cat(feat_grid, dim=1)

            # Concat to dim [b, 64 * 5, h, w]
            feat_prop = torch.cat([feat_i, feat_grid, feat_prop], dim=1)
            feat_prop = self.forward_trunk_2(feat_prop)
            fp_dict['for_trunk_2'].append(feat_prop)

        out_l = []
        for i in range(0, n):
            out = feat[:, i, :, :, :]
            for j in fp_dict.keys():
                out = torch.cat([out, fp_dict[j][i]], dim=1)
            out = self.lrelu(self.fusion(out))
            out = self.lrelu(self.pixel_shuffle(self.upconv1(out)))
            out = self.lrelu(self.pixel_shuffle(self.upconv2(out)))
            out = self.lrelu(self.conv_hr(out))
            out = self.conv_last(out)
            base = self.img_upsample(x[:, i, :, :, :])
            out += base
            out_l.append(out)

        return torch.stack(out_l, dim=1)
"""