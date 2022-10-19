# --------------------------------------------------------
# GANet
# Copyright (c) 2022 SenseTime
# @Time    : 2022/04/23
# @Author  : Jinsheng Wang
# @Email   : jswang@stu.pku.edu.cn
# --------------------------------------------------------

import torch
from torch import nn
import torch.nn.functional as F
import numpy as np

from mmdet.models import build_loss
from mmdet.models import HEADS
from .dense_heads.ctnet_head import CtnetHead


def compute_locations(shape, device):
    pos = torch.arange(
        0, shape[-1], step=1, dtype=torch.float32, device=device)
    pos = pos.reshape((1, 1, -1))
    pos = pos.repeat(shape[0], shape[1], 1)
    return pos


def make_mask(shape=(1, 80, 200), device=torch.device('cuda')):
    x_coord = torch.arange(0, shape[-1], step=1, dtype=torch.float32, device=device)
    x_coord = x_coord.reshape(1, 1, -1)
    # x_coord = np.repeat(x_coord, shape[1], 1)
    x_coord = x_coord.repeat(1, shape[1], 1)
    y_coord = torch.arange(0, shape[-2], step=1, dtype=torch.float32, device=device)
    y_coord = y_coord.reshape(1, -1, 1)
    y_coord = y_coord.repeat(1, 1, shape[-1])
    coord_mat = torch.cat((x_coord, y_coord), axis=0)
    # print('coord_mat shape{}'.format(coord_mat.shape))
    return coord_mat

class UpSampleLayer(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(UpSampleLayer, self).__init__()
        self.Conv_BN_ReLU_2 = nn.Sequential(
            nn.Conv2d(in_channels=in_ch, out_channels=out_ch, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(),
            nn.Conv2d(in_channels=out_ch, out_channels=out_ch, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU()
        )
        # self.upsample = nn.Sequential(
        #     nn.ConvTranspose2d(in_channels=out_ch, out_channels=out_ch, kernel_size=3, stride=2, padding=1,
        #                        output_padding=1),
        #     nn.BatchNorm2d(out_ch),
        #     nn.ReLU()
        # )

    def forward(self, x):
        out = self.Conv_BN_ReLU_2(x)
        # out = self.upsample(out)
        out = F.interpolate(input=out, scale_factor=2, mode='bilinear')
        return out


# from mmdet.models import AnchorFreeHead
@HEADS.register_module()
class GANetHeadFast(nn.Module):
    def __init__(self,
                 num_classes, # 1种lane关键点，num_lane_classes
                 in_channels, # 输入特征，也就是经过neck之后的dim=64
                 hm_idx, # 只用neck的第一个hm
                 loss_cfg=None,
                 kpt_thr=0.3,
                 root_thr=1,
                 train_cfg=None,
                 test_cfg=None,
                ):
        super(GANetHeadFast, self).__init__() # 必须要
        self.num_classes = num_classes
        self.hm_idx = hm_idx
        self.loss = build_loss(loss_cfg)
        self.kpt_thr = kpt_thr
        self.root_thr = root_thr

        self.keypts_head = CtnetHead(
            heads=dict(hm=num_classes),
            channels_in=in_channels,
            final_kernel=1,
            head_conv=in_channels)

        self.offset_head = CtnetHead(
            heads=dict(offset_map=2),
            channels_in=in_channels,
            final_kernel=1,
            head_conv=in_channels)

        self.reg_head = CtnetHead(
            heads=dict(offset_map=2),
            channels_in=in_channels,
            final_kernel=1,
            head_conv=in_channels)

    def forward(self, x):
        f_hm = x['features'][self.hm_idx]
        aux_feat = x['aux_feat']
        kpts_hm = self.keypts_head(f_hm)['hm']
        kpts_hm = torch.clamp(kpts_hm.sigmoid(), min=1e-4, max=1 - 1e-4) ## 之前漏加
        if aux_feat is not None:
            f_hm = aux_feat
        pts_offset = self.offset_head(f_hm)['offset_map']
        int_offset = self.reg_head(f_hm)['offset_map']
        return dict(
            kpts_hm=kpts_hm,
            pts_offset=pts_offset, 
            int_offset=int_offset
        )
                
    def forward_train(self, x, targets):
        head_dict = self.forward(x)
        deform_points = x['deform_points']
        losses = self.loss(head_dict,targets,deform_points)
        return losses
    
    def forward_test(self, x):
        head_dict = self.forward(x)
        output = self.ktdet_decode_fast(
            head_dict['kpts_hm'],head_dict['pts_offset'],head_dict['int_offset'])
        return output

    def ktdet_decode_fast(self, kpts_hm, pts_offset, int_offset):
        def _nms(heat):
            hmax = nn.functional.max_pool2d(heat, (1, 3), stride=(1, 1), padding=(0, 1))
            keep = (hmax == heat).float()
            return heat * keep

        heat_nms = _nms(kpts_hm)

        # generate root centers array from offset map parallel
        offset_split = torch.split(pts_offset, 1, dim=1)
        mask = torch.lt(offset_split[1], self.root_thr)  # offset < 1
        mask_nms = torch.gt(heat_nms, self.kpt_thr)  # key point score > 0.3
        mask_low = mask * mask_nms
        mask_low = mask_low[0, 0].transpose(1, 0).detach().cpu().numpy()
        idx = np.where(mask_low)
        root_center_arr = np.array(idx, dtype=int).transpose()

        # generate roots by coord add offset parallel
        heat_nms = heat_nms.squeeze(0).permute(1, 2, 0).detach()
        pts_offset = pts_offset.squeeze(0).permute(1, 2, 0).detach()
        int_offset = int_offset.squeeze(0).permute(1, 2, 0).detach()
        coord_mat = make_coordmat(shape=kpts_hm.shape[1:])  # 0.2ms
        coord_mat = coord_mat.permute(1, 2, 0)
        # print('\nkpt thr:', thr)
        heat_mat = heat_nms.repeat(1, 1, 2)
        root_mat = coord_mat + pts_offset
        align_mat = coord_mat + int_offset
        inds_mat = torch.where(heat_mat > self.kpt_thr)
        root_arr = root_mat[inds_mat].reshape(-1, 2).cpu().numpy()
        align_arr = align_mat[inds_mat].reshape(-1, 2).cpu().numpy()
        kpt_seeds = np.concatenate([align_arr[:,None,:],root_arr[:,None,:]], axis=1)
        return root_center_arr, kpt_seeds


def make_coordmat(shape=(1, 80, 200), device=torch.device('cuda')):
    x_coord = torch.arange(0, shape[-1], step=1, dtype=torch.float32, device=device)
    x_coord = x_coord.reshape(1, 1, -1)
    # x_coord = np.repeat(x_coord, shape[1], 1)
    x_coord = x_coord.repeat(1, shape[1], 1)
    y_coord = torch.arange(0, shape[-2], step=1, dtype=torch.float32, device=device)
    y_coord = y_coord.reshape(1, -1, 1)
    y_coord = y_coord.repeat(1, 1, shape[-1])
    coord_mat = torch.cat((x_coord, y_coord), axis=0)
    # print('coord_mat shape{}'.format(coord_mat.shape))
    return coord_mat
