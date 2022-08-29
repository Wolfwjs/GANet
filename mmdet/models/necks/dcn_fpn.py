# --------------------------------------------------------
# GANet
# Copyright (c) 2022 SenseTime
# @Time    : 2022/04/23
# @Author  : Yinchao Ma
# @Email   : imyc@mail.ustc.edu.cn
# --------------------------------------------------------

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import ConvModule, xavier_init

from mmdet.models.dense_heads import LanePointsConv
from mmdet.core import auto_fp16
from ..builder import NECKS
from mmcv import Timer

class PositionEmbeddingSine(nn.Module):
    """
    This is a more standard version of the position embedding, very similar to the one
    used by the Attention is all you need paper, generalized to work on images.
    """

    def __init__(self,
                 num_pos_feats=64,
                 temperature=10000,
                 normalize=False,
                 scale=None):
        super().__init__()
        self.num_pos_feats = num_pos_feats
        self.temperature = temperature
        self.normalize = normalize
        if scale is not None and normalize is False:
            raise ValueError("normalize should be True if scale is passed")
        if scale is None:
            scale = 2 * math.pi
        self.scale = scale

    def forward(self, mask):
        assert mask is not None
        not_mask = ~mask
        y_embed = not_mask.cumsum(1, dtype=torch.float32)
        x_embed = not_mask.cumsum(2, dtype=torch.float32)
        if self.normalize:
            eps = 1e-6
            y_embed = y_embed / (y_embed[:, -1:, :] + eps) * self.scale
            x_embed = x_embed / (x_embed[:, :, -1:] + eps) * self.scale

        dim_t = torch.arange(
            self.num_pos_feats, dtype=torch.float32, device=mask.device)
        dim_t = self.temperature ** (2 * (dim_t // 2) / self.num_pos_feats)

        pos_x = x_embed[:, :, :, None] / dim_t
        pos_y = y_embed[:, :, :, None] / dim_t
        pos_x = torch.stack(
            (pos_x[:, :, :, 0::2].sin(), pos_x[:, :, :, 1::2].cos()),
            dim=4).flatten(3)
        pos_y = torch.stack(
            (pos_y[:, :, :, 0::2].sin(), pos_y[:, :, :, 1::2].cos()),
            dim=4).flatten(3)
        pos = torch.cat((pos_y, pos_x), dim=3).permute(0, 3, 1, 2)
        return pos


def build_position_encoding(hidden_dim, shape):
    mask = torch.zeros(shape, dtype=torch.bool)
    pos_module = PositionEmbeddingSine(hidden_dim // 2)
    pos_embs = pos_module(mask)
    return pos_embs


class AttentionLayer(nn.Module):
    """ Position attention module"""

    def __init__(self, in_dim, out_dim, ratio=4, stride=1):
        super(AttentionLayer, self).__init__()
        self.chanel_in = in_dim
        norm_cfg = dict(type='BN', requires_grad=True)
        act_cfg = dict(type='ReLU')
        self.pre_conv = ConvModule(
            in_dim,
            out_dim,
            kernel_size=3,
            stride=stride,
            padding=1,
            norm_cfg=norm_cfg,
            act_cfg=act_cfg,
            inplace=False)
        self.query_conv = nn.Conv2d(
            in_channels=out_dim, out_channels=out_dim // ratio, kernel_size=1)
        self.key_conv = nn.Conv2d(
            in_channels=out_dim, out_channels=out_dim // ratio, kernel_size=1)
        self.value_conv = nn.Conv2d(
            in_channels=out_dim, out_channels=out_dim, kernel_size=1)
        self.final_conv = ConvModule(
            out_dim,
            out_dim,
            kernel_size=3,
            padding=1,
            norm_cfg=norm_cfg,
            act_cfg=act_cfg)
        self.softmax = nn.Softmax(dim=-1)
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, x, pos=None):
        """
            inputs :
                x : input feature maps( B X C X H X W)
            returns :
                out : attention value + input feature
                attention: B X (HxW) X (HxW)
        """
        x = self.pre_conv(x)
        m_batchsize, _, height, width = x.size()
        if pos is not None:
            x = x + pos
        proj_query = self.query_conv(x).view(m_batchsize, -1,
                                             width * height).permute(0, 2, 1)
        proj_key = self.key_conv(x).view(m_batchsize, -1, width * height)

        energy = torch.bmm(proj_query, proj_key)
        attention = self.softmax(energy)
        attention = attention.permute(0, 2, 1)
        proj_value = self.value_conv(x).view(m_batchsize, -1, width * height)
        out = torch.bmm(proj_value, attention)
        out = out.view(m_batchsize, -1, height, width)
        proj_value = proj_value.view(m_batchsize, -1, height, width)
        out_feat = self.gamma * out + x
        out_feat = self.final_conv(out_feat)
        return out_feat


class TransConvEncoderModule(nn.Module):
    def __init__(self, in_dim, attn_in_dims, attn_out_dims, strides, ratios, downscale=True, pos_shape=None):
        super(TransConvEncoderModule, self).__init__()
        if downscale:
            stride = 2
        else:
            stride = 1
        # self.first_conv = ConvModule(in_dim, 2*in_dim, kernel_size=3, stride=stride, padding=1)
        # self.final_conv = ConvModule(attn_out_dims[-1], attn_out_dims[-1], kernel_size=3, stride=1, padding=1)
        attn_layers = []
        for dim1, dim2, stride, ratio in zip(attn_in_dims, attn_out_dims, strides, ratios):
            attn_layers.append(AttentionLayer(dim1, dim2, ratio, stride))
        if pos_shape is not None:
            self.attn_layers = nn.ModuleList(attn_layers)
        else:
            self.attn_layers = nn.Sequential(*attn_layers)
        self.pos_shape = pos_shape
        self.pos_embeds = []
        if not self.training:
            self.pos_shape[0] = 1
        if pos_shape is not None:
            for dim in attn_out_dims:
                pos_embed = build_position_encoding(dim, pos_shape).cuda()
                self.pos_embeds.append(pos_embed)

    def forward(self, src):
        # src = self.first_conv(src)
        if self.pos_shape is None:
            src = self.attn_layers(src)
        else:
            for layer, pos in zip(self.attn_layers, self.pos_embeds):
                src = layer(src, pos.to(src.device))
        # src = self.final_conv(src)
        return src


@NECKS.register_module()
class DeformFPN(nn.Module):
    """
    Feature Pyramid Network.

    This is an implementation of - Feature Pyramid Networks for Object
    Detection (https://arxiv.org/abs/1612.03144)

    Args:
        in_channels (List[int]): Number of input channels per scale.
        out_channels (int): Number of output channels (used at each scale)
        num_outs (int): Number of output scales.
        start_level (int): Index of the start input backbone level used to
            build the feature pyramid. Default: 0.
        end_level (int): Index of the end input backbone level (exclusive) to
            build the feature pyramid. Default: -1, which means the last level.
        add_extra_convs (bool): Whether to add conv layers on top of the
            original feature maps. Default: False.
        extra_convs_on_inputs (bool): Whether to apply extra conv on
            the original feature from the backbone. Default: False.
        relu_before_extra_convs (bool): Whether to apply relu before the extra
            conv. Default: False.
        no_norm_on_lateral (bool): Whether to apply norm on lateral.
            Default: False.
        conv_cfg (dict): Config dict for convolution layer. Default: None.
        norm_cfg (dict): Config dict for normalization layer. Default: None.
        act_cfg (str): Config dict for activation layer in ConvModule.
            Default: None.

    Example:
        >>> import torch
        >>> in_channels = [2, 3, 5, 7]
        >>> scales = [340, 170, 84, 43]
        >>> inputs = [torch.rand(1, c, s, s)
        ...           for c, s in zip(in_channels, scales)]
        >>> self = FPN(in_channels, 11, len(in_channels)).eval()
        >>> outputs = self.forward(inputs)
        >>> for i in range(len(outputs)):
        ...     print(f'outputs[{i}].shape = {outputs[i].shape}')
        outputs[0].shape = torch.Size([1, 11, 340, 340])
        outputs[1].shape = torch.Size([1, 11, 170, 170])
        outputs[2].shape = torch.Size([1, 11, 84, 84])
        outputs[3].shape = torch.Size([1, 11, 43, 43])
    """

    def __init__(self,
                 in_channels,
                 out_channels,
                 start_level=0,
                 end_level=-1,
                 dcn_point_num=[9,7,5,3],
                 deconv_layer=[True, True, True, True],
                 add_extra_convs=False,
                 extra_convs_on_inputs=True,
                 relu_before_extra_convs=False,
                 no_norm_on_lateral=False,
                 use_latern=False,
                 use_res=False,
                 deconv_before=True,
                 trans_cfg=None,
                 dconv_cfg=None,
                 trans_idx=-1,
                 trans_mode="replace",
                 conv_cfg=None,
                 norm_cfg=None,
                 act_cfg=None,
                 dcn_only_cls=False):
        super(DeformFPN, self).__init__()
        assert isinstance(in_channels, list)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_ins = len(in_channels)
        self.relu_before_extra_convs = relu_before_extra_convs
        self.no_norm_on_lateral = no_norm_on_lateral
        self.fp16_enabled = False
        self.deconv_layer = deconv_layer
        self.deconv_before = deconv_before
        self.use_res = use_res
        self.trans_cfg = trans_cfg
        self.trans_mode = trans_mode
        self.dcn_only_cls = dcn_only_cls
        if self.trans_cfg is not None:
            in_channels[-1] = trans_cfg['attn_out_dims'][-1]
            self.trans_idx = trans_idx
            self.trans_head = TransConvEncoderModule(**trans_cfg)

        if end_level == -1:
            self.backbone_end_level = self.num_ins

        self.start_level = start_level
        self.end_level = end_level
        self.add_extra_convs = add_extra_convs
        self.extra_convs_on_inputs = extra_convs_on_inputs

        self.lateral_convs = nn.ModuleList()
        self.fpn_convs = nn.ModuleList()
        self.def_convs = nn.ModuleList()


        if dconv_cfg is None:
            feat_channels = 256
            stacked_convs = 3
        else:
            feat_channels = dconv_cfg['feat_channels']
            stacked_convs = dconv_cfg['stacked_convs']
        for i in range(self.start_level, self.backbone_end_level):
            l_conv = ConvModule(
                in_channels[i],
                out_channels,
                1,
                conv_cfg=conv_cfg,
                norm_cfg=norm_cfg if not self.no_norm_on_lateral else None,
                act_cfg=act_cfg,
                inplace=False)
            def_conv = LanePointsConv(
                in_channels=out_channels,
                feat_channels=feat_channels,
                point_feat_channels=out_channels,
                num_points=dcn_point_num[i],
                use_latern=use_latern,
                stacked_convs=stacked_convs)
            fpn_conv = ConvModule(
                out_channels,
                out_channels,
                3,
                padding=1,
                conv_cfg=conv_cfg,
                norm_cfg=norm_cfg,
                act_cfg=act_cfg,
                inplace=False)

            self.lateral_convs.append(l_conv)
            self.def_convs.append(def_conv)
            self.fpn_convs.append(fpn_conv)


    # default init_weights for conv(msra) and norm in ConvModule
    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                xavier_init(m, distribution='uniform')

    @auto_fp16()
    def forward(self, inputs):
        inputs = list(inputs)
        if len(inputs) > len(self.in_channels):
            del inputs[0]

        if self.trans_cfg is not None:
            trans_feat = self.trans_head(inputs[self.trans_idx])
            if self.trans_mode == 'replace':
                inputs[-1] = trans_feat
        # build laterals
        # adjust the channels layer3,4,5
        laterals = []
        deform_points = []
        for i, (lateral_conv, deform_conv) in enumerate(zip(self.lateral_convs, self.def_convs)):
            if self.deconv_layer[i] and self.deconv_before:
                mid_feat = lateral_conv(inputs[i + self.start_level])
                d_feat, points = deform_conv(mid_feat)
                if self.use_res:
                    laterals.append(d_feat+mid_feat)
                else:
                    laterals.append(d_feat)
                deform_points.append(points)
            else:
                d_feat = lateral_conv(inputs[i + self.start_level])
                laterals.append(d_feat)
                deform_points.append(None)

        # build top-down path
        # the prev layer align to curr layer and add
        used_backbone_levels = len(laterals) # 3
        for i in range(used_backbone_levels - 1, 0, -1):
            prev_shape = laterals[i - 1].shape[2:]
            laterals[i - 1] += F.interpolate(
                laterals[i], size=prev_shape, mode='nearest')
            if i-1 == 0:
                aux_feat = laterals[0]
            if self.deconv_layer[i-1] and not self.deconv_before:
                if self.use_res:
                    mid_feat, points = self.def_convs[i-1](laterals[i - 1])
                    laterals[i - 1] = mid_feat + laterals[i - 1]
                else:
                    laterals[i - 1], points = self.def_convs[i-1](laterals[i - 1])
                deform_points[i-1] = points

        # build outputs
        # part 1: from original levels
        # adjust the feature by conv
        outs = [
            self.fpn_convs[i](laterals[i]) for i in range(used_backbone_levels)
        ]
        
        if self.dcn_only_cls:
            output = {
                "features": tuple(outs),
                "aux_feat": aux_feat,
                "deform_points": tuple(deform_points)
            }
        else:
            output = {
                "features": tuple(outs),
                "aux_feat": None,
                "deform_points": tuple(deform_points)
            }

        return output
