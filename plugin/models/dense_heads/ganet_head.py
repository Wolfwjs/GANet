# --------------------------------------------------------
# GANet
# Copyright (c) 2022 SenseTime
# @Time    : 2022/04/23
# @Author  : Jinsheng Wang
# @Email   : jswang@stu.pku.edu.cn
# --------------------------------------------------------

import numpy as np
import torch
from torch import nn
import torch.functional as F
import torch.nn.functional as F

from mmdet.models import HEADS
from .ctnet_head import CtnetHead


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


@HEADS.register_module()
class GANetHeadFast(nn.Module):

    def __init__(self,
                 heads,
                 in_channels,
                 num_classes,
                 branch_in_channels=288,
                 hm_idx=0,  # input id for heatmap
                 joint_nums=1,
                 regression=True,
                 upsample_num=0,
                 root_thr=1,
                 train_cfg=None,
                 test_cfg=None):
        super(GANetHeadFast, self).__init__()
        self.root_thr = root_thr
        self.num_classes = num_classes
        self.hm_idx = hm_idx
        self.joint_nums = joint_nums
        if upsample_num > 0:
            self.upsample_module = nn.ModuleList([UpSampleLayer(in_ch=branch_in_channels, out_ch=branch_in_channels)
                                                  for i in range(upsample_num)])
        else:
            self.upsample_module = None

        # self.centerpts_head = CtnetHead(
        #     heads,
        #     channels_in=branch_in_channels,
        #     final_kernel=1,
        #     head_conv=branch_in_channels)

        self.keypts_head = CtnetHead(
            heads,
            channels_in=branch_in_channels,
            final_kernel=1,
            head_conv=branch_in_channels)

        self.offset_head = CtnetHead(
            heads=dict(offset_map=self.joint_nums * 2),
            channels_in=branch_in_channels,
            final_kernel=1,
            head_conv=branch_in_channels)

        self.reg_head = CtnetHead(
            heads=dict(offset_map=2),
            channels_in=branch_in_channels,
            final_kernel=1,
            head_conv=branch_in_channels)

    def ktdet_decode(self, heat, offset, error, thr=0.1):

        def _nms(heat, kernel=3):
            pad = (kernel - 1) // 2
            hmax = nn.functional.max_pool2d(heat, (1, 3), stride=(1, 1), padding=(0, 1))
            keep = (hmax == heat).float()  # false:0 true:1
            return heat * keep  # type: tensor

        def check_range(start, end, value):
            if value < start:
                # print('out range value:{}'.format(value))
                return start
            elif value > end:
                # print('out range value:{}'.format(value))
                return end
            else:
                return value

        def get_virtual_down_coord(coord, offset_map, root_i):
            x, y = coord[0], coord[1]
            x_max = offset_map.shape[1] - 1
            y_max = offset_map.shape[0] - 1
            x = check_range(0, x_max, value=x)
            y = check_range(0, y_max, value=y)
            offset_vector = offset_map[y, x]
            offset_vector = offset_vector.reshape(-1, 2)
            offset_min_idx, offset_min_value = 0, 9999
            for idx, _offset in enumerate(offset_vector):
                offset_y = _offset[1]
                if offset_y < 0:
                    continue
                if offset_y < offset_min_value:
                    offset_min_value = offset_y
                    offset_min_idx = idx
            if offset_min_value < 5 and offset_min_idx > 0:
                offset_min_idx = offset_min_idx - 1
            offset_min = offset_vector[offset_min_idx]
            virtual_down_x, virtual_down_y = x + offset_min[0] + 0.49999, y + offset_min[1] + 0.49999
            virtual_down_coord = [int(virtual_down_x), int(virtual_down_y)]
            return virtual_down_coord

        def get_vitual_root(coord, offset_map):
            virtual_down_root0 = get_virtual_down_coord(coord, offset_map, 0)
            virtual_down_root1 = get_virtual_down_coord(virtual_down_root0, offset_map, 1)
            virtual_down_root2 = get_virtual_down_coord(virtual_down_root1, offset_map, 2)
            virtual_down_root3 = get_virtual_down_coord(virtual_down_root2, offset_map, 3)
            return virtual_down_root3

        def get_vitual_root_one(coord, offset_map):
            virtual_down_root = get_virtual_down_coord(coord, offset_map, 0)
            return virtual_down_root

        def _format(heat, offset, error, inds):
            ret = []
            for y, x, c in zip(inds[0], inds[1], inds[2]):
                id_class = c + 1
                coord = [x, y]
                score = heat[y, x, c]
                if offset.shape[-1] == 2:
                    _virtual_root = get_vitual_root_one(coord, offset)
                else:
                    _virtual_root = get_vitual_root(coord, offset)
                _error = error[y, x]
                ret.append((np.int32(coord + _error), np.int32(_virtual_root)))
            return ret

        heat_nms = _nms(heat)
        heat_nms = heat_nms.squeeze(0).permute(1, 2, 0).detach().cpu().numpy()
        offset = offset.squeeze(0).permute(1, 2, 0).detach().cpu().numpy()
        error = error.squeeze(0).permute(1, 2, 0).detach().cpu().numpy()

        inds = np.where(heat_nms > thr)
        seeds = _format(heat_nms, offset, error, inds)
        return seeds

    def ktdet_decode_fast(self, heat, offset, error, thr=0.1, root_thr=1):

        def _nms(heat, kernel=3):
            hmax = nn.functional.max_pool2d(heat, (1, 3), stride=(1, 1), padding=(0, 1))
            keep = (hmax == heat).float()  # false:0 true:1
            return heat * keep  # type: tensor

        heat_nms = _nms(heat)

        # generate root centers array from offset map parallel
        offset_split = torch.split(offset, 1, dim=1)
        mask = torch.lt(offset_split[1], root_thr)  # offset < 1
        mask_nms = torch.gt(heat_nms, thr)  # key point score > 0.3
        mask_low = mask * mask_nms
        mask_low = mask_low[0, 0].transpose(1, 0).detach().cpu().numpy()
        idx = np.where(mask_low)
        root_center_arr = np.array(idx, dtype=int).transpose()

        # generate roots by coord add offset parallel
        heat_nms = heat_nms.squeeze(0).permute(1, 2, 0).detach()
        offset = offset.squeeze(0).permute(1, 2, 0).detach()
        error = error.squeeze(0).permute(1, 2, 0).detach()
        coord_mat = make_coordmat(shape=heat.shape[1:])  # 0.2ms
        coord_mat = coord_mat.permute(1, 2, 0)
        # print('\nkpt thr:', thr)
        heat_mat = heat_nms.repeat(1, 1, 2)
        root_mat = coord_mat + offset
        align_mat = coord_mat + error
        inds_mat = torch.where(heat_mat > thr)
        root_arr = root_mat[inds_mat].reshape(-1, 2).cpu().numpy()
        align_arr = align_mat[inds_mat].reshape(-1, 2).cpu().numpy()
        kpt_seeds = []
        for (align, root) in (zip(align_arr, root_arr)):
            kpt_seeds.append((align, np.array(root, dtype=float)))

        return root_center_arr, kpt_seeds

    def forward_train(self, inputs, aux_feat=None):
        x_list = list(inputs)
        f_hm = x_list[self.hm_idx]
        if self.upsample_module is not None:
            for upsample in self.upsample_module:
                f_hm = upsample(f_hm)
                if aux_feat is not None:
                    aux_feat = upsample(aux_feat)

        # z = self.centerpts_head(f_hm)
        # cpts_hm = z['hm']

        z_ = self.keypts_head(f_hm)
        kpts_hm = z_['hm']

        if aux_feat is not None:
            f_hm = aux_feat
        o = self.offset_head(f_hm)
        pts_offset = o['offset_map']

        o_ = self.reg_head(f_hm)
        int_offset = o_['offset_map']

        return [kpts_hm, pts_offset, int_offset]

    def forward_test(
            self,
            inputs,
            aux_feat=None,
            hack_seeds=None,
            hm_thr=0.3,
            kpt_thr=0.4,
            cpt_thr=0.4,
    ):

        x_list = list(inputs)
        f_hm = x_list[self.hm_idx]
        if self.upsample_module is not None:
            for upsample in self.upsample_module:
                f_hm = upsample(f_hm)
                if aux_feat is not None:
                    aux_feat = upsample(aux_feat)
        # center points hm
        # z = self.centerpts_head(f_hm)
        # hm = z['hm']
        # hm = torch.clamp(hm.sigmoid(), min=1e-4, max=1 - 1e-4)
        # cpts_hm = hm

        # key points hm
        z_ = self.keypts_head(f_hm)
        kpts_hm = z_['hm']
        kpts_hm = torch.clamp(kpts_hm.sigmoid(), min=1e-4, max=1 - 1e-4)

        # offset map
        if aux_feat is not None:
            f_hm = aux_feat
        o = self.offset_head(f_hm)
        pts_offset = o['offset_map']

        o_ = self.reg_head(f_hm)
        int_offset = o_['offset_map']

        if pts_offset.shape[1] > 2:
            def _nms(heat, kernel=3):
                hmax = nn.functional.max_pool2d(heat, (1, 3), stride=(1, 1), padding=(0, 1))
                keep = (hmax == heat).float()  # false:0 true:1
                return heat * keep  # type: tensor

            heat_nms = _nms(kpts_hm)
            offset_split = torch.split(pts_offset, 1, dim=1)
            mask = torch.lt(offset_split[1], self.root_thr)  # offset < 1
            mask_nms = torch.gt(heat_nms, kpt_thr)  # key point score > 0.3
            mask_low = mask * mask_nms
            mask_low = torch.squeeze(mask_low).permute(1, 0).detach().cpu().numpy()
            idx = np.where(mask_low)
            cpt_seeds = np.array(idx, dtype=int).transpose()
            kpt_seeds = self.ktdet_decode(kpts_hm, pts_offset, int_offset,
                                          thr=kpt_thr)  # key point position list[dict{} ]
        else:
            cpt_seeds, kpt_seeds = self.ktdet_decode_fast(kpts_hm, pts_offset, int_offset, thr=kpt_thr,
                                                          root_thr=self.root_thr)

        return [cpt_seeds, kpt_seeds]

    def inference_mask(self, pos):
        pass

    def forward(
            self,
            x_list,
            hm_thr=0.3,
            kpt_thr=0.4,
            cpt_thr=0.4,
    ):
        return self.forward_test(x_list, hm_thr, kpt_thr, cpt_thr)

    def init_weights(self):
        # ctnet_head will init weights during building
        pass
