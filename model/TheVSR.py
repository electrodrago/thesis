import sys, os
import torch
from torch import nn as nn
from collections import OrderedDict

sys.path.append(os.path.dirname(__file__))
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))


from utils.arch_utils import ResidualBlockNoBN, flow_warp, make_layer
from model.SPyNet_arch import SpyNet


class TheVSR(nn.Module):
    def __init__(self, num_feat=64, num_block=10, spynet_path=None):
        super().__init__()
        self.num_feat = num_feat

        # feature extraction module
        self.feat_extract = ConvResidualBlocks(3, num_feat // 2, 5)
        self.feat_extract_clean = ConvResidualBlocks(3, num_feat // 2, 5)

        # alignment
        self.spynet = SpyNet(spynet_path)

        # propagation
        # feat + feat_prop: 64 * 2 channels
        self.backward_trunk_1 = \
            ConvResidualBlocks(num_feat * 2, num_feat, num_block)

        # feat + back_trunk_1 (b1) + feat_prop: 64 * 3 channels
        self.forward_trunk_1 = \
            ConvResidualBlocks(num_feat * 3, num_feat, num_block)

        # feat + b1 + f1 + feat_prop: 64 * 4 channels
        self.backward_trunk_2 = \
            ConvResidualBlocks(num_feat * 4, num_feat, num_block)

        # feat + b1 + f1 + b2 + feat_prop: 64 * 5 channels
        self.forward_trunk_2 = \
            ConvResidualBlocks(num_feat * 5, num_feat, num_block)

        # reconstruction
        # feat + b1 + f1 + b2 + f2: 64 * 5 channels
        self.fusion = ConvResidualBlocks(5 * num_feat, num_feat, 5)

        self.upconv1 = nn.Conv2d(num_feat, num_feat * 4, 3, 1, 1, bias=True)
        self.upconv2 = nn.Conv2d(num_feat, 64 * 4, 3, 1, 1, bias=True)

        self.conv_hr = nn.Conv2d(64, 64, 3, 1, 1)
        self.conv_last = nn.Conv2d(64, 3, 3, 1, 1)

        self.img_upsample = \
            nn.Upsample(scale_factor=4, mode='bilinear', align_corners=False)

        self.pixel_shuffle = nn.PixelShuffle(2)

        # activation functions
        self.lrelu = nn.LeakyReLU(negative_slope=0.1, inplace=True)

        # img cleaning module
        self.image_cleaning = nn.Sequential(
            ConvResidualBlocks(3, self.num_feat, 15),
            nn.Conv2d(self.num_feat, 3, 3, 1, 1, bias=True),
        )

        self.is_mirror_extended = False
        self.spynet.requires_grad_(False)

    def check_if_mirror_extended(self, lqs):
        """Check whether the input is a mirror-extended sequence.

        If mirror-extended, the i-th (i=0, ..., t-1) frame is equal to the (t-1-i)-th frame.

        Args:
            lqs (tensor): Input low quality (LQ) sequence with shape (n, t, c, h, w).
        """

        if lqs.size(1) % 2 == 0:
            lqs_1, lqs_2 = torch.chunk(lqs, 2, dim=1)
            if torch.norm(lqs_1 - lqs_2.flip(1)) == 0:
                self.is_mirror_extended = True

    def get_flow(self, x):
        b, n, c, h, w = x.size()

        x_1 = x[:, :-1, :, :, :].reshape(-1, c, h, w)
        x_2 = x[:, 1:, :, :, :].reshape(-1, c, h, w)

        flows_backward = self.spynet(x_1, x_2).view(b, n - 1, 2, h, w)
        # flows_forward = self.spynet(x_2, x_1).view(b, n - 1, 2, h, w)
        if self.is_mirror_extended:  # flows_forward = flows_backward.flip(1)
            flows_forward = flows_backward.flip(1)
        else:
            flows_forward = self.spynet(x_2, x_1).view(b, n - 1, 2, h, w)
        # flows_backward: [b, 14, 2, h, w]

        return flows_forward, flows_backward

    def propagate_each(self, feat, fp_dict, flows: torch.Tensor, trunk, back=True):
        b, n, _, h, w = feat.size()
        if back:
            range_ = range(n - 1, -1, -1)
            start_warp = n - 1
        else:
            range_ = range(0, n)
            start_warp = 0

        out_prop = []
        # Tensor same dtype, same device
        feat_prop = feat.new_zeros(b, self.num_feat, h, w)
        for i in range_:
            feat_i = feat[:, i, :, :, :]
            # feat_i: [b, 64, 64, 64]

            if back:
                if i < start_warp:
                    flow = flows[:, i, :, :, :]
                    feat_prop = flow_warp(feat_prop, flow.permute(0, 2, 3, 1))
            else:
                if i > start_warp:
                    flow = flows[:, i - 1, :, :, :]
                    feat_prop = flow_warp(feat_prop, flow.permute(0, 2, 3, 1))
            # feat_prop: [b, 64, 64, 64]

            feat_grid = [feat_i]

            # Grid propagation
            if fp_dict:
                for j in fp_dict.keys():
                    feat_grid.append(fp_dict[j][i])

            # Insert feature to head and prop to tail
            feat_grid.append(feat_prop)
            feat_grid = torch.cat(feat_grid, dim=1)
            feat_prop = feat_prop + trunk(feat_grid)
            out_prop.append(feat_prop)

        return out_prop

    def forward(self, lqs: torch.Tensor, return_lqs=False):
        # b, 15, 3, h, w
        b, n, c, h, w = lqs.size()

        lqs_clean = lqs.detach().clone()

        for _ in range(0, 3):  # at most 3 cleaning, determined empirically
            lqs_clean = lqs_clean.view(-1, c, h, w)
            residues = self.image_cleaning(lqs_clean)
            lqs_clean = (lqs_clean + residues).view(b, n, c, h, w)

            # determine whether to continue cleaning
            if torch.mean(torch.abs(residues)) < 1.0:
                break

        self.check_if_mirror_extended(lqs)

        flows_forward, flows_backward = self.get_flow(lqs)

        # Input: shape(lqs): [b, 3, 64, 64]
        # Output: shape(feat): [b, 32, 64, 64]
        feat = self.feat_extract(lqs.view(-1, c, h, w))
        feat_clean = self.feat_extract_clean(lqs_clean.view(-1, c, h, w))

        feat = feat.view(b, n, -1, h, w)
        feat_clean = feat_clean.view(b, n, -1, h, w)
        feat = torch.cat([feat, feat_clean], dim=2)
        # feat: [b, 15, 64, 64, 64]

        # Feature propagation dict for grid: back_trunk, for_trunk
        fp_dict = OrderedDict()

        # backward branch 1
        bb1 = self.propagate_each(feat, None, flows_backward, self.backward_trunk_1, back=True)
        fp_dict['back_trunk_1'] = []
        for i in bb1:
            fp_dict['back_trunk_1'].insert(0, i)

        # forward branch 1
        fb1 = self.propagate_each(feat, fp_dict, flows_forward, self.forward_trunk_1, back=False)
        fp_dict['for_trunk_1'] = fb1

        # backward branch 2
        bb2 = self.propagate_each(feat, fp_dict, flows_backward, self.backward_trunk_2, back=True)
        fp_dict['back_trunk_2'] = []
        for i in bb2:
            fp_dict['back_trunk_2'].insert(0, i)

        # forward branch 2
        fb2 = self.propagate_each(feat, fp_dict, flows_forward, self.forward_trunk_2, back=False)
        fp_dict['for_trunk_2'] = fb2

        out_l = []
        for i in range(0, n):
            out = feat[:, i, :, :, :]
            for trunk in fp_dict.keys():
                out = torch.cat([out, fp_dict[trunk][i]], dim=1)

            out = self.fusion(out)
            out = self.lrelu(self.pixel_shuffle(self.upconv1(out)))
            out = self.lrelu(self.pixel_shuffle(self.upconv2(out)))
            out = self.lrelu(self.conv_hr(out))
            out = self.conv_last(out)
            base = self.img_upsample(lqs[:, i, :, :, :])
            out += base
            out_l.append(out)

        if return_lqs:
            return torch.stack(out_l, dim=1), lqs
        else:
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
            nn.Conv2d(num_in_ch, num_out_ch, 3, 1, 1, bias=True),
            nn.LeakyReLU(negative_slope=0.1, inplace=True),
            make_layer(ResidualBlockNoBN, num_block, num_feat=num_out_ch))

    def forward(self, fea):
        return self.main(fea)
