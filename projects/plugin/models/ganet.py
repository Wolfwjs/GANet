# --------------------------------------------------------
# GANet
# Copyright (c) 2022 SenseTime
# @Time    : 2022/04/23
# @Author  : Yinchao Ma
# @Email   : imyc@mail.ustc.edu.cn
# --------------------------------------------------------

import os
import mmcv
import numpy as np
import copy
import PIL
import cv2
import torch

from mmdet.models.detectors import SingleStageDetector
from mmdet.models.builder import DETECTORS
from projects.plugin.core.post_process import PostProcessor

@DETECTORS.register_module()
class GANet(SingleStageDetector):
    def __init__(self,
                 backbone,neck,head,
                 train_cfg=None,test_cfg=None,
                 init_cfg=None,
                 post_processing=None,
            ):
        super(GANet, self).__init__(
            backbone=backbone,
            neck=neck,
            bbox_head=head,
            train_cfg=train_cfg,
            test_cfg=test_cfg,
            init_cfg=init_cfg)
        self.post_processing = PostProcessor(**post_processing)
        
    # 使用SingleStageDetector内定义的extract_feat
    def forward_train(self, img, img_metas, targets, **kwargs):
        x = self.extract_feat(img)
        losses = self.bbox_head.forward_train(x,targets)
        return losses
    
    def forward_test(self, img, img_metas, **kwargs):
        assert img.shape[0] == 1 and len(img_metas)==1
        x = self.extract_feat(img.type(torch.cuda.FloatTensor))
        pred = self.bbox_head.forward_test(x) # 里面对关键点和起始点进行解码选择出来
        result = self.post_processing(pred, img_metas[0]) # dict_keys(['final_dict_list', 'img_metas']),单图结果,保留img_metas因为后面可视化要使用其中filename
        return [result] # 表示bs=1
        
    # base_detector里train_step会调用，加上return_loss=true
    def forward(self, img, img_metas=None, targets=None, return_loss=True, **kwargs):
        img = img[0]
        img_metas = img_metas[0]
        if return_loss:
            return self.forward_train(img, img_metas, targets, **kwargs)
        else:
            return self.forward_test(img, img_metas, **kwargs)
    
    def show_result(self,
                    img,
                    result,
                    score_thr=0.3,
                    bbox_color=(72, 101, 241),
                    text_color=(72, 101, 241),
                    mask_color=None,
                    thickness=2,
                    font_size=13,
                    win_name='',
                    show=False,
                    wait_time=0,
                    out_file=None):
        img = mmcv.imread(img) # resize get
        img = img.copy()

        img_pred, img_gt, img_vc, img_cc = vis_one(img, result) 

        img1 = np.concatenate([img_pred,img_gt],axis=0) # (590, 1640, 3) (0,)
        img2 = np.concatenate([img_vc,img_cc],axis=0)
        img = np.concatenate([img1,img2],axis=1)
        
        if out_file:
            mmcv.mkdir_or_exist(os.path.dirname(out_file))
            mmcv.imwrite(img, out_file)
        else:
            return img



def vis_one(img_, output, lane_width=7): 
    # dict_keys(['decode_dict', 'group_dict_list', 'final_dict_list', 'img_metas'])
    result, virtual_center, cluster_center = output['final_dict_list']
    # result = [t['keypt'] for t in result]
    # cluster_center = [t['startpt'] for t in result]
    # virtual_center = [t['startpt_pred'] for t in result]

    img_metas = output['img_metas']
    img = mmcv.imread(img_metas['filename'])
    img_pred = copy.deepcopy(img)
    img_gt = copy.deepcopy(img)
    img_vc = copy.deepcopy(img)
    img_cc = copy.deepcopy(img)
    img_pil = PIL.Image.fromarray(img_pred)
    img_gt_pil = PIL.Image.fromarray(img_gt)

    for idx, lane in enumerate(result):
        lane_tuple = [tuple(p) for p in lane]
        PIL.ImageDraw.Draw(img_pil).line(
            xy=lane_tuple, fill=COLORS[idx + 1], width=lane_width)

    for idx, vp in enumerate(virtual_center):
        vp_tuple = [tuple(p) for p in vp]
        for _vp in vp_tuple:
            cv2.circle(img=img_cc, center=_vp, radius=3, color=COLORS[idx + 1], thickness=-1) 

    for idx, cp in enumerate(cluster_center): # [(12, 426), (140, 692), (1164, 692), (1228, 412)]
        cv2.circle(img=img_vc, center=tuple(cp), radius=10, color=COLORS[idx + 1], thickness=-1)
        cv2.circle(img=img_cc, center=tuple(cp), radius=40, color=COLORS[idx + 1], thickness=3)

    img_pred = np.array(img_pil, dtype=np.uint8)

    if 'gt_points_for_show' in img_metas:
        gt_lanes = img_metas['gt_points_for_show']
        gt_lanes = [[(lane[2*i],lane[2*i+1]) for i in range(len(lane)//2)] for lane in gt_lanes]
        for idx, lane in enumerate(gt_lanes):
            lane_tuple = [tuple(p) for p in lane]
            PIL.ImageDraw.Draw(img_gt_pil).line(
                xy=lane_tuple, fill=COLORS[idx + 1], width=lane_width)
        img_gt = np.array(img_gt_pil, dtype=np.uint8)

    return img_pred, img_gt, img_vc, img_cc

COLORS = [
    (255, 0, 0),
    (0, 255, 0),
    (0, 0, 255),
    (255, 255, 0),
    (255, 0, 255),
    (0, 255, 255),
    (128, 255, 0),
    (255, 128, 0),
    (128, 0, 255),
    (255, 0, 128),
    (0, 128, 255),
    (0, 255, 128),
    (128, 255, 255),
    (255, 128, 255),
    (255, 255, 128),
    (60, 180, 0),
    (180, 60, 0),
    (0, 60, 180),
    (0, 180, 60),
    (60, 0, 180),
    (180, 0, 60),
    (255, 0, 0),
    (0, 255, 0),
    (0, 0, 255),
    (255, 255, 0),
    (255, 0, 255),
    (0, 255, 255),
    (128, 255, 0),
    (255, 128, 0),
    (128, 0, 255),
    (255, 0, 0),
    (0, 255, 0),
    (0, 0, 255),
    (255, 255, 0),
    (255, 0, 255),
    (0, 255, 255),
    (128, 255, 0),
    (255, 128, 0),
    (128, 0, 255),
    (255, 0, 128),
    (0, 128, 255),
    (0, 255, 128),
    (128, 255, 255),
    (255, 128, 255),
    (255, 255, 128),
    (60, 180, 0),
    (180, 60, 0),
    (0, 60, 180),
    (0, 180, 60),
    (60, 0, 180),
    (180, 0, 60),
    (255, 0, 0),
    (0, 255, 0),
    (0, 0, 255),
    (255, 255, 0),
    (255, 0, 255),
    (0, 255, 255),
    (128, 255, 0),
    (255, 128, 0),
    (128, 0, 255),
]