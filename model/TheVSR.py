import torch
from torch import nn as nn
from torch.nn import functional as F

from .arch_utils import ResidualBlockNoBN, flow_warp, make_layer
from .SPyNet_arch import SpyNet


class TheVSR(nn.Module):
    """A recurrent network for video SR. Now only x4 is supported.

    Args:
        num_feat (int): Number of channels. Default: 64.
        num_block (int): Number of residual blocks for each branch. Default: 15
        spynet_path (str): Path to the pretrained weights of SPyNet. Default: None.
    """

    def __init__(self, num_feat=64, num_block=15, spynet_path=None):
        super().__init__()
        self.num_feat = num_feat

        # feature extraction module
        self.feat_extract = ConvResidualBlocks(3, num_feat, 5)

        # alignment
        self.spynet = SpyNet(spynet_path)

        # propagation
        # feat_prop + feat: 64 + 64 channels
        self.backward_trunk_1 = \
            ConvResidualBlocks(num_feat * 2, num_feat, num_block)

        # feat_prop + back_trunk_1 (b1) + feat: 64 + 64 + 64 c
        self.forward_trunk_1 = \
            ConvResidualBlocks(num_feat * 3, num_feat, num_block)

        # feat_prop + b1 + f1 + feat: 64 + 64 + 64 + 64 c
        self.backward_trunk_2 = \
            ConvResidualBlocks(num_feat * 4, num_feat, num_block)

        # feat_prop + b1 + f1 + b2 + feat: 64 + 64 + 64 + 64 + 64 c
        self.forward_trunk_2 = \
            ConvResidualBlocks(num_feat * 5, num_feat, num_block)

        # reconstruction
        self.fusion = \
            nn.Conv2d(num_feat * 5, num_feat, 1, 1, 0, bias=True)

        self.upconv1 = nn.Conv2d(num_feat, num_feat * 4, 3, 1, 1, bias=True)
        self.upconv2 = nn.Conv2d(num_feat, 64 * 4, 3, 1, 1, bias=True)

        self.conv_hr = nn.Conv2d(64, 64, 3, 1, 1)
        self.conv_last = nn.Conv2d(64, 3, 3, 1, 1)

        self.img_upsample = \
            nn.Upsample(scale_factor=4, mode='bilinear', align_corners=False)

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

    def propagate_each(self, feat, fp_dict, flows, trunk, back=True):
        b, n, _, h, w = feat.size()
        if back:
            range_ = range(0, n)
            start_warp = 0
        else:
            range_ = range(n - 1, -1, -1)
            start_warp = n - 1

        out_prop = []
        feat_prop = flows.new_zeros(b, self.num_feat, h, w)
        for i in range_:
            feat_i = feat[:, i, :, :, :]
            if back:
                if i < start_warp:
                    flow = flows[:, i, :, :, :]
                    feat_prop = flow_warp(feat_prop, flow.permute(0, 2, 3, 1))
            else:
                if i > start_warp:
                    flow = flows[:, i, :, :, :]
                    feat_prop = flow_warp(feat_prop, flow.permute(0, 2, 3, 1))
            feat_grid = []
            if fp_dict:
                for j in fp_dict.keys():
                    feat_grid.append(fp_dict[j][i])
            # Insert feature to head and prop to tail
            feat_grid.insert(0, feat_i)
            feat_grid.append(feat_prop)
            feat_prop = torch.cat(feat_grid, dim=1)
            out_prop.append(trunk(feat_prop))
        return out_prop

    def forward(self, x):
        flows_forward, flows_backward = self.get_flow(x)
        # b, 15, 3, h, w
        b, n, _, h, w = x.size()

        feat = self.feat_extract(x)

        # Feature propagation dict for grid: back_trunk, for_trunk
        fp_dict = dict()

        # backward branch 1
        bb1 = self.propagate_each(feat, None, flows_backward, self.backward_trunk_1, back=True)
        fp_dict['back_trunk_1'] = []
        for i in range(len(bb1)):
            fp_dict['back_trunk_1'].insert(0, i)

        # forward branch 1
        fb1 = self.propagate_each(feat, fp_dict, flows_forward, self.forward_trunk_1, back=False)
        fp_dict['for_trunk_1'] = fb1

        # backward branch 2
        bb2 = self.propagate_each(feat, fp_dict, flows_backward, self.backward_trunk_2, back=True)
        fp_dict['back_trunk_2'] = []
        for i in range(len(bb2)):
            fp_dict['back_trunk_2'].insert(0, i)

        # forward branch 2
        fb2 = self.propagate_each(feat, fp_dict, flows_forward, self.forward_trunk_2, back=False)
        fp_dict['for_trunk_2'] = fb2

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
