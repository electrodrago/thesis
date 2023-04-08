import torch
import torch.nn as nn
from torch.nn import init as init
from torch.nn.modules.batchnorm import _BatchNorm
from torch.nn import functional as F
from collections import OrderedDict
import math
from mmcv.runner import load_checkpoint
from mmedit.models.registry import BACKBONES
from mmedit.utils import get_root_logger


@BACKBONES.register_module()
class RealBasicVSRNet(nn.Module):
    def __init__(self, num_feat=64, num_block=20, spynet_path=None):
        super().__init__()
        self.num_feat = num_feat

        # feature extraction module
        self.feat_extract = ConvResidualBlocks(3, num_feat // 2, 5)
        self.feat_extract_clean = ConvResidualBlocks(3, num_feat // 2, 5)

        # alignment
        self.spynet = SpyNet(spynet_path)

        # propagation
        # feat_1 + feat_prop_1 + feat_prop_2: 32 + 32 + 32 channels
        # b1
        self.backward_trunk_1 = \
            ConvResidualBlocks(num_feat * 3 // 2, num_feat // 2, num_block)
        
        # feat_2 + feat_prop_1 + feat_prop_2: 32 + 32 + 32 channels
        # b2
        self.backward_trunk_2 = \
            ConvResidualBlocks(num_feat * 3 // 2, num_feat // 2, num_block)

        # feat_1 + b1 + b2 + feat_prop_1 + feat_prop_2: 32 + 32 + 32 + 32 + 32 channels
        # f1
        self.forward_trunk_1 = \
            ConvResidualBlocks(num_feat * 5 // 2, num_feat // 2, num_block)

        # feat_2 + b1 + b2 + feat_prop_1 + feat_prop_2: 32 + 32 + 32 + 32 + 32 channels
        # f2
        self.forward_trunk_2 = \
            ConvResidualBlocks(num_feat * 5 // 2, num_feat // 2, num_block)

        # reconstruction
        # b1 + f1 + b2 + f2: 32 + 32 + 32 + 32 channels
        self.fusion = ConvResidualBlocks(2 * num_feat, num_feat, 5)

        self.upconv1 = nn.Conv2d(num_feat, num_feat * 4, 3, 1, 1, bias=True)
        self.upconv2 = nn.Conv2d(num_feat, 64 * 4, 3, 1, 1, bias=True)

        self.conv_hr = nn.Conv2d(64, 64, 3, 1, 1)
        self.conv_last = nn.Conv2d(64, 3, 3, 1, 1)

        self.channel_shuffle = nn.ChannelShuffle(2)

        self.img_upsample = \
            nn.Upsample(scale_factor=4, mode='bilinear', align_corners=False)

        self.pixel_shuffle = nn.PixelShuffle(2)

        # activation functions
        self.lrelu = nn.LeakyReLU(negative_slope=0.1, inplace=True)

        # img cleaning module
        self.image_cleaning = nn.Sequential(
            ConvResidualBlocks(3, self.num_feat, num_block),
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

    def propagate_back(self, feat_1, feat_2, flows: torch.Tensor):
        b, n, _, h, w = feat_1.size()
        
        range_ = range(n - 1, -1, -1)
        start_warp = n - 1

        out_prop_1 = []
        out_prop_2 = []
        # Tensor same dtype, same device
        feat_prop_1 = feat_1.new_zeros(b, self.num_feat // 2, h, w)
        feat_prop_2 = feat_2.new_zeros(b, self.num_feat // 2, h, w)
        for i in range_:
            feat_i_1 = feat_1[:, i, :, :, :]
            feat_i_2 = feat_2[:, i, :, :, :]
            # feat_i_x: [b, 32, 64, 64]

            if i < start_warp:
                flow = flows[:, i, :, :, :]
                feat_prop_1 = flow_warp(feat_prop_1, flow.permute(0, 2, 3, 1))
                feat_prop_2 = flow_warp(feat_prop_2, flow.permute(0, 2, 3, 1))
            
            # feat_prop_x: [b, 32, 64, 64]
            feat_grid_1 = [feat_i_1, feat_prop_1, feat_prop_2]
            feat_grid_2 = [feat_i_2, feat_prop_1, feat_prop_2]

            # To tensor shape: [b, 32 * 3, 64, 64]
            feat_grid_1 = torch.cat(feat_grid_1, dim=1)
            feat_grid_2 = torch.cat(feat_grid_2, dim=1)

            # Backward cross propagation
            feat_prop_1 = feat_prop_1 + self.backward_trunk_1(feat_grid_1)
            feat_prop_2 = feat_prop_2 + self.backward_trunk_2(feat_grid_2)

            out_prop_1.append(feat_prop_1)
            out_prop_2.append(feat_prop_2)

        return out_prop_1, out_prop_2
    
    def propagate_for(self, feat_1, feat_2, b1, b2, flows: torch.Tensor):
        b, n, _, h, w = feat_1.size()
        
        range_ = range(0, n)
        start_warp = 0

        out_prop_1 = []
        out_prop_2 = []
        # Tensor same dtype, same device
        feat_prop_1 = feat_1.new_zeros(b, self.num_feat // 2, h, w)
        feat_prop_2 = feat_2.new_zeros(b, self.num_feat // 2, h, w)
        for i in range_:
            feat_i_1 = feat_1[:, i, :, :, :]
            feat_i_2 = feat_2[:, i, :, :, :]
            # feat_i_x: [b, 32, 64, 64]

            if i > start_warp:
                flow = flows[:, i - 1, :, :, :]
                feat_prop_1 = flow_warp(feat_prop_1, flow.permute(0, 2, 3, 1))
                feat_prop_2 = flow_warp(feat_prop_2, flow.permute(0, 2, 3, 1))
            # feat_prop_x: [b, 32, 64, 64]

            # feat_prop_x: [b, 32, 64, 64]
            feat_grid_1 = [feat_i_1, feat_prop_1, feat_prop_2, b1[i], b2[i]]
            feat_grid_2 = [feat_i_2, feat_prop_1, feat_prop_2, b1[i], b2[i]]

            # To tensor shape: [b, 32 * 5, 64, 64]
            feat_grid_1 = torch.cat(feat_grid_1, dim=1)
            feat_grid_2 = torch.cat(feat_grid_2, dim=1)

            # Backward cross propagation
            feat_prop_1 = feat_prop_1 + self.forward_trunk_1(feat_grid_1)
            feat_prop_2 = feat_prop_2 + self.forward_trunk_2(feat_grid_2)

            out_prop_1.append(feat_prop_1)
            out_prop_2.append(feat_prop_2)

        return out_prop_1, out_prop_2

    def forward(self, lqs: torch.Tensor, return_lqs=False):
        # b, 30, 3, h, w
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
        feat = self.channel_shuffle(feat.view(-1, self.num_feat, h, w)).view(b, n, -1, h, w)

        feat = torch.split(feat, 32, dim=2)
        feat_1, feat_2 = feat[0], feat[1]
        # feat_x: [b, 30, 32, 64, 64]

        # Feature propagation: back_trunk, for_trunk
        b1, b2 = self.propagate_back(feat_1, feat_2, flows_backward)
        f1, f2 = self.propagate_for(feat_1, feat_2, b1, b2, flows_forward)

        out_l = []
        for i in range(0, n):
            out = torch.cat([b1[i], b2[i], f1[i], f2[i]], dim=1)

            out = self.fusion(out)
            out = self.lrelu(self.pixel_shuffle(self.upconv1(out)))
            out = self.lrelu(self.pixel_shuffle(self.upconv2(out)))
            out = self.lrelu(self.conv_hr(out))
            out = self.conv_last(out)
            base = self.img_upsample(lqs[:, i, :, :, :])
            out += base
            out_l.append(out)

        if return_lqs:
            return torch.stack(out_l, dim=1), lqs_clean
        else:
            return torch.stack(out_l, dim=1)
    

    def init_weights(self, pretrained=None, strict=True):
        """Init weights for models.
        Args:
            pretrained (str, optional): Path for pretrained weights. If given
                None, pretrained weights will not be loaded. Defaults: None.
            strict (boo, optional): Whether strictly load the pretrained model.
                Defaults to True.
        """
        if isinstance(pretrained, str):
            logger = get_root_logger()
            load_checkpoint(self, pretrained, strict=strict, logger=logger)
        elif pretrained is not None:
            raise TypeError(f'"pretrained" must be a str or None. '
                            f'But received {type(pretrained)}.')


@torch.no_grad()
def default_init_weights(module_list, scale=1, bias_fill=0, **kwargs):
    """Initialize network weights.

    Args:
        module_list (list[nn.Module] | nn.Module): Modules to be initialized.
        scale (float): Scale initialized weights, especially for residual
            blocks. Default: 1.
        bias_fill (float): The value to fill bias. Default: 0
        kwargs (dict): Other arguments for initialization function.
    """
    if not isinstance(module_list, list):
        module_list = [module_list]
    for module in module_list:
        for m in module.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, **kwargs)
                m.weight.data *= scale
                if m.bias is not None:
                    m.bias.data.fill_(bias_fill)
            elif isinstance(m, nn.Linear):
                init.kaiming_normal_(m.weight, **kwargs)
                m.weight.data *= scale
                if m.bias is not None:
                    m.bias.data.fill_(bias_fill)
            elif isinstance(m, _BatchNorm):
                init.constant_(m.weight, 1)
                if m.bias is not None:
                    m.bias.data.fill_(bias_fill)


class ResidualBlockNoBN(nn.Module):
    """Residual block without BN.

    Args:
        num_feat (int): Channel number of intermediate features.
            Default: 64.
        res_scale (float): Residual scale. Default: 1.
        pytorch_init (bool): If set to True, use pytorch default init,
            otherwise, use default_init_weights. Default: False.
    """

    def __init__(self, num_feat=64, res_scale=1, pytorch_init=False):
        super(ResidualBlockNoBN, self).__init__()
        self.res_scale = res_scale
        self.conv1 = nn.Conv2d(num_feat, num_feat, 3, 1, 1, bias=True)
        self.conv2 = nn.Conv2d(num_feat, num_feat, 3, 1, 1, bias=True)
        self.relu = nn.ReLU(inplace=True)

        if not pytorch_init:
            default_init_weights([self.conv1, self.conv2], 0.1)

    def forward(self, x):
        identity = x
        out = self.conv2(self.relu(self.conv1(x)))
        return identity + out * self.res_scale


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


def flow_warp(x, flow, interp_mode='bilinear', padding_mode='zeros', align_corners=True):
    """Warp an image or feature map with optical flow.

    Args:
        x (Tensor): Tensor with size (n, c, h, w).
        flow (Tensor): Tensor with size (n, h, w, 2), normal value.
        interp_mode (str): 'nearest' or 'bilinear'. Default: 'bilinear'.
        padding_mode (str): 'zeros' or 'border' or 'reflection'.
            Default: 'zeros'.
        align_corners (bool): Before pytorch 1.3, the default value is
            align_corners=True. After pytorch 1.3, the default value is
            align_corners=False. Here, we use the True as default.

    Returns:
        Tensor: Warped image or feature map.
    """
    assert x.size()[-2:] == flow.size()[1:3]
    _, _, h, w = x.size()
    # create mesh grid: TODO: why not tensor zeros here?
    grid_y, grid_x = torch.meshgrid(torch.arange(0, h).type_as(x), torch.arange(0, w).type_as(x))
    grid = torch.stack((grid_x, grid_y), 2).float()  # W(x), H(y), 2
    grid.requires_grad = False

    # vgrid: [b, h, w, 2]
    # grid: [h, w, 2]
    # flow: [b, h, w, 2]

    vgrid = grid + flow
    # scale grid to [-1,1]
    vgrid_x = 2.0 * vgrid[:, :, :, 0] / max(w - 1, 1) - 1.0
    vgrid_y = 2.0 * vgrid[:, :, :, 1] / max(h - 1, 1) - 1.0
    vgrid_scaled = torch.stack((vgrid_x, vgrid_y), dim=3)
    # x: [b, 64, h, w]
    output = F.grid_sample(x, vgrid_scaled, mode=interp_mode, padding_mode=padding_mode, align_corners=align_corners)

    # output: [b, 64, h, w]
    return output


def resize_flow(flow, size_type, sizes, interp_mode='bilinear', align_corners=False):
    """Resize a flow according to ratio or shape.

    Args:
        flow (Tensor): Precomputed flow. shape [N, 2, H, W].
        size_type (str): 'ratio' or 'shape'.
        sizes (list[int | float]): the ratio for resizing or the final output
            shape.
            1) The order of ratio should be [ratio_h, ratio_w]. For
            downsampling, the ratio should be smaller than 1.0 (i.e., ratio
            < 1.0). For upsampling, the ratio should be larger than 1.0 (i.e.,
            ratio > 1.0).
            2) The order of output_size should be [out_h, out_w].
        interp_mode (str): The mode of interpolation for resizing.
            Default: 'bilinear'.
        align_corners (bool): Whether align corners. Default: False.

    Returns:
        Tensor: Resized flow.
    """
    _, _, flow_h, flow_w = flow.size()
    if size_type == 'ratio':
        output_h, output_w = int(flow_h * sizes[0]), int(flow_w * sizes[1])
    elif size_type == 'shape':
        output_h, output_w = sizes[0], sizes[1]
    else:
        raise ValueError(f'Size type should be ratio or shape, but got type {size_type}.')

    input_flow = flow.clone()
    ratio_h = output_h / flow_h
    ratio_w = output_w / flow_w
    input_flow[:, 0, :, :] *= ratio_w
    input_flow[:, 1, :, :] *= ratio_h
    resized_flow = F.interpolate(
        input=input_flow, size=(output_h, output_w), mode=interp_mode, align_corners=align_corners)
    return resized_flow


def make_layer(basic_block, num_basic_block, **kwarg):
    """Make layers by stacking the same blocks.

    Args:
        basic_block (nn.module): nn.module class for basic block.
        num_basic_block (int): number of blocks.

    Returns:
        nn.Sequential: Stacked blocks in nn.Sequential.
    """
    layers = []
    for _ in range(num_basic_block):
        layers.append(basic_block(**kwarg))
    return nn.Sequential(*layers)


class BasicModule(nn.Module):
    """Basic Module for SpyNet.
    """

    def __init__(self):
        super(BasicModule, self).__init__()

        self.basic_module = nn.Sequential(
            nn.Conv2d(in_channels=8, out_channels=32, kernel_size=7, stride=1, padding=3), nn.ReLU(inplace=False),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=7, stride=1, padding=3), nn.ReLU(inplace=False),
            nn.Conv2d(in_channels=64, out_channels=32, kernel_size=7, stride=1, padding=3), nn.ReLU(inplace=False),
            nn.Conv2d(in_channels=32, out_channels=16, kernel_size=7, stride=1, padding=3), nn.ReLU(inplace=False),
            nn.Conv2d(in_channels=16, out_channels=2, kernel_size=7, stride=1, padding=3))

    def forward(self, tensor_input):
        return self.basic_module(tensor_input)


class SpyNet(nn.Module):
    """SpyNet architecture.

    Args:
        load_path (str): path for pretrained SpyNet. Default: None.
    """

    def __init__(self, load_path=None):
        super(SpyNet, self).__init__()
        self.basic_module = nn.ModuleList([BasicModule() for _ in range(6)])
        if load_path:
            self.load_state_dict(torch.load(load_path, map_location=lambda storage, loc: storage)['params'])

        self.register_buffer('mean', torch.Tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer('std', torch.Tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def preprocess(self, tensor_input):
        tensor_output = (tensor_input - self.mean) / self.std
        return tensor_output

    def process(self, ref, supp):
        flow = []

        ref = [self.preprocess(ref)]
        supp = [self.preprocess(supp)]

        for level in range(5):
            ref.insert(0, F.avg_pool2d(input=ref[0], kernel_size=2, stride=2, count_include_pad=False))
            supp.insert(0, F.avg_pool2d(input=supp[0], kernel_size=2, stride=2, count_include_pad=False))

        flow = ref[0].new_zeros(
            [ref[0].size(0), 2,
             int(math.floor(ref[0].size(2) / 2.0)),
             int(math.floor(ref[0].size(3) / 2.0))])

        for level in range(len(ref)):
            upsampled_flow = F.interpolate(input=flow, scale_factor=2, mode='bilinear', align_corners=True) * 2.0

            if upsampled_flow.size(2) != ref[level].size(2):
                upsampled_flow = F.pad(input=upsampled_flow, pad=[0, 0, 0, 1], mode='replicate')
            if upsampled_flow.size(3) != ref[level].size(3):
                upsampled_flow = F.pad(input=upsampled_flow, pad=[0, 1, 0, 0], mode='replicate')

            flow = self.basic_module[level](torch.cat([
                ref[level],
                flow_warp(
                    supp[level], upsampled_flow.permute(0, 2, 3, 1), interp_mode='bilinear', padding_mode='border'),
                upsampled_flow
            ], 1)) + upsampled_flow

        return flow

    def forward(self, ref, supp):
        assert ref.size() == supp.size()

        h, w = ref.size(2), ref.size(3)
        w_floor = math.floor(math.ceil(w / 32.0) * 32.0)
        h_floor = math.floor(math.ceil(h / 32.0) * 32.0)

        ref = F.interpolate(input=ref, size=(h_floor, w_floor), mode='bilinear', align_corners=False)
        supp = F.interpolate(input=supp, size=(h_floor, w_floor), mode='bilinear', align_corners=False)

        flow = F.interpolate(input=self.process(ref, supp), size=(h, w), mode='bilinear', align_corners=False)

        flow[:, 0, :, :] *= float(w) / float(w_floor)
        flow[:, 1, :, :] *= float(h) / float(h_floor)

        return flow
