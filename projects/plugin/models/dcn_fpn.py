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

from .dense_heads import LanePointsConv
from mmcv.runner import auto_fp16
from mmdet.models import NECKS
from mmdet.models.utils import SinePositionalEncoding

def build_position_encoding(hidden_dim, shape):
    mask = torch.zeros(shape, dtype=torch.bool)
    pos_module = SinePositionalEncoding(hidden_dim // 2)
    pos_embs = pos_module(mask)
    return pos_embs

class AttentionLayer(nn.Module):
    """ Position attention module"""

    def __init__(self, in_dim, out_dim, ratio=4, stride=1): # (512, 64, 4, 1)
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
            in_channels=out_dim, out_channels=out_dim // ratio, kernel_size=1) # 64,16,1
        self.key_conv = nn.Conv2d(
            in_channels=out_dim, out_channels=out_dim // ratio, kernel_size=1) # 64,16,1
        self.value_conv = nn.Conv2d(
            in_channels=out_dim, out_channels=out_dim, kernel_size=1) # 64,64,1
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
            x += pos
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
    def __init__(self, 
            attn_in_dims, 
            attn_out_dims, 
            strides, 
            ratios, 
            pos_shape=None):
        super(TransConvEncoderModule, self).__init__()
        # self.first_conv = ConvModule(in_dim, 2*in_dim, kernel_size=3, stride=stride, padding=1)
        # self.final_conv = ConvModule(attn_out_dims[-1], attn_out_dims[-1], kernel_size=3, stride=1, padding=1)
        attn_layers = []
        for dim1, dim2, stride, ratio in zip(attn_in_dims, attn_out_dims, strides, ratios):
            attn_layers.append(AttentionLayer(dim1, dim2, ratio, stride))
        if pos_shape is not None:
            self.attn_layers = nn.ModuleList(attn_layers)
            # 它是一个容器，可以储存不同的module，并将每个module添加到网络之中
            # nn.ModuleList的内部没有实现forward()函数，故需要使用forward()函数对其进行调用。
        else:
            self.attn_layers = nn.Sequential(*attn_layers)
            # 它里面的模块是按照顺序进行排列的，所以需要保证前一个模块的输出大小和下一个模块的输入大小一致。
        self.pos_shape = pos_shape
        self.pos_embeds = []
        # 过这两个attn layer，shape是不变的
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

    def __init__(self,
                 in_channels, # list
                 out_channels, # int
                 start_level=0,
                 end_level=-1,
                 dcn_point_num=[9,7,5,3],
                 deconv_layer=[True, True, True, True],
                 add_extra_convs=False, # 可以是字符串
                 # extra_convs_on_inputs=True,
                 relu_before_extra_convs=False,
                 no_norm_on_lateral=False,
                 use_latern=False,
                 use_res=False,
                 deconv_before=False,
                 trans_cfg=None,
                 dconv_cfg=None,
                 trans_idx=-1,
                 trans_mode="replace",
                 conv_cfg=None,
                 norm_cfg=None,
                 act_cfg=None,
                 dcn_only_cls=False):
        super(DeformFPN, self).__init__()
        self.deconv_layer = deconv_layer # [True, False, False]
        self.deconv_before = deconv_before # False
        self.use_res = use_res # false
        self.trans_cfg = trans_cfg # {'in_dim': 512, 'attn_in_dims': [512, 64], 'attn_out_dims': [64, 64], 'strides': [1, 1], 'ratios': [4, 4], 'pos_shape': (1, 10, 25)}
        self.trans_mode = trans_mode # 'replace'
        self.dcn_only_cls = dcn_only_cls # true
        if self.trans_cfg is not None:
            # in_channels要因为attn操作而改变channel数
            in_channels[-1] = trans_cfg['attn_out_dims'][-1] # [128, 256, 64], because do att to last layer, channel become to 64
            self.trans_idx = trans_idx # -1, do attention to the last layer fm
            self.trans_head = TransConvEncoderModule(**trans_cfg) # this is attention
            ## todo
        '''
        super(DeformFPN, self).__init__(
            in_channels,
            out_channels,
            num_outs=len(in_channels),
            start_level=start_level,
            end_level=end_level,
            add_extra_convs=add_extra_convs,
            relu_before_extra_convs=relu_before_extra_convs,
            no_norm_on_lateral=no_norm_on_lateral,
            conv_cfg=conv_cfg,
            norm_cfg=norm_cfg,
            act_cfg=act_cfg
        )
        '''
        num_outs = len(in_channels)
        upsample_cfg = {'mode': 'nearest'}
        assert isinstance(in_channels, list)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_ins = len(in_channels)
        self.num_outs = num_outs
        self.relu_before_extra_convs = relu_before_extra_convs # false
        self.no_norm_on_lateral = no_norm_on_lateral # false
        self.fp16_enabled = False
        self.upsample_cfg = upsample_cfg.copy() # {'mode': 'nearest'}

        if end_level == -1 or end_level == self.num_ins - 1:
            self.backbone_end_level = self.num_ins # 3
            assert num_outs >= self.num_ins - start_level
        else:
            # if end_level is not the last level, no extra level is allowed
            self.backbone_end_level = end_level + 1
            assert end_level < self.num_ins
            assert num_outs == end_level - start_level + 1
        self.start_level = start_level # 0
        self.end_level = end_level # -1
        self.add_extra_convs = add_extra_convs # false
        assert isinstance(add_extra_convs, (str, bool))
        if isinstance(add_extra_convs, str):
            # Extra_convs_source choices: 'on_input', 'on_lateral', 'on_output'
            assert add_extra_convs in ('on_input', 'on_lateral', 'on_output')
        elif add_extra_convs:  # True
            self.add_extra_convs = 'on_input' # todo

        self.lateral_convs = nn.ModuleList()
        self.fpn_convs = nn.ModuleList()

        for i in range(self.start_level, self.backbone_end_level): # 0,1,2
            l_conv = ConvModule(
                in_channels[i],
                out_channels,
                1,
                conv_cfg=conv_cfg,
                norm_cfg=norm_cfg if not self.no_norm_on_lateral else None,
                act_cfg=act_cfg,
                inplace=False)
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
            self.fpn_convs.append(fpn_conv)

        # add extra conv layers (e.g., RetinaNet)
        extra_levels = num_outs - self.backbone_end_level + self.start_level
        if self.add_extra_convs and extra_levels >= 1:
            for i in range(extra_levels):
                if i == 0 and self.add_extra_convs == 'on_input':
                    in_channels = self.in_channels[self.backbone_end_level - 1]
                else:
                    in_channels = out_channels
                extra_fpn_conv = ConvModule(
                    in_channels,
                    out_channels,
                    3,
                    stride=2,
                    padding=1,
                    conv_cfg=conv_cfg,
                    norm_cfg=norm_cfg,
                    act_cfg=act_cfg,
                    inplace=False)
                self.fpn_convs.append(extra_fpn_conv)
        
        # 此外构建deformconv_conv
        self.def_convs = nn.ModuleList()
        if dconv_cfg is None:
            feat_channels = 256 ## set this for LanePointsConv
            stacked_convs = 3
        else:
            feat_channels = dconv_cfg['feat_channels']
            stacked_convs = dconv_cfg['stacked_convs']
        for i in range(self.start_level, self.backbone_end_level): # 0,1,2, produce for every level
            def_conv = LanePointsConv(
                in_channels=out_channels,
                feat_channels=feat_channels,
                point_feat_channels=out_channels,
                num_points=dcn_point_num[i],
                use_latern=use_latern,
                stacked_convs=stacked_convs)
            self.def_convs.append(def_conv)
            
    # default init_weights for conv(msra) and norm in ConvModule
    '''
    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                xavier_init(m, distribution='uniform')
    '''

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                xavier_init(m, distribution='uniform')

    @auto_fp16()
    def forward(self, inputs):
        inputs = list(inputs)
        if len(inputs) > len(self.in_channels):
            del inputs[0] # del the first
        assert len(inputs) == len(self.in_channels)

        # 对trans_idx=-1进行attn
        if self.trans_cfg is not None:
            trans_feat = self.trans_head(inputs[self.trans_idx]) # torch.Size([32, 64, 10, 25])
            if self.trans_mode == 'replace':
                inputs[self.trans_idx] = trans_feat
                
        # build laterals,如果deform_before就是在这里deform
        laterals = [None]*self.num_ins
        deform_points = [None]*self.num_ins
        
        for i, (lateral_conv, deform_conv) in enumerate(zip(self.lateral_convs, self.def_convs)): # len=3
            mid_feat = lateral_conv(inputs[i + self.start_level])
            laterals[i]=mid_feat
            if self.deconv_before and self.deconv_layer[i]:
                d_feat, points = deform_conv(mid_feat)
                deform_points[i]=points
                if self.use_res:
                    laterals[i]=d_feat+mid_feat
                else:
                    laterals[i]=d_feat
                    
        # build top-down path
        used_backbone_levels = len(laterals) # 3
        for i in range(used_backbone_levels - 1, 0, -1):
            # 对这3个特征进行逆序操作，只训练3-1次就可以
            prev_shape = laterals[i - 1].shape[2:]
            # 前一个也就是上面的那个
            laterals[i - 1] += F.interpolate(
                laterals[i], size=prev_shape, **self.upsample_cfg)
            # 将本特征上采样后加到上面的那个
            if i-1 == 0:
                aux_feat = laterals[0]
            # aux_feat是正常经过采样加和之后的fpn，没有deform的特征（在不是deform_before的时候）
            # 下面判断加和后得到的上面的那层是否需要进行一下dconv
            # 这样的设置使得最后一层不可能是dconv，因为就没有判断它
            if self.deconv_layer[i-1] and not self.deconv_before: # do deform to (i-1) pos, [True, False, False], when i=1, do
                d_feat, points = self.def_convs[i-1](laterals[i - 1])
                if self.use_res:
                    laterals[i - 1] = d_feat + laterals[i - 1]
                else:
                    laterals[i - 1] = d_feat
                deform_points[i-1] = points

        # build outputs
        outs = [
            self.fpn_convs[i](laterals[i]) for i in range(used_backbone_levels)
        ]
        
        # 默认情况add_extra_convs=false
        output = dict(
            features=tuple(outs),
            aux_feat=None,
            deform_points=tuple(deform_points)
        )
        if self.dcn_only_cls:
            output.update(aux_feat=aux_feat)
        return output

