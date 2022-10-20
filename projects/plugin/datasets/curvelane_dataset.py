import os
import json

import mmcv
from mmcv import ProgressBar
import cv2
from tqdm import tqdm
import numpy as np
import copy

from mmdet.datasets.builder import DATASETS

from tools.ganet.curvelane.curvelane_evaluate import LaneMetricCore

from .culane_dataset import CulaneDataset


@DATASETS.register_module
class CurvelaneDataset(CulaneDataset):

    def __init__(self,
                 data_root,
                 data_list,
                 pipeline,
                 test_mode=False,
                 test_suffix='png',
                 work_dir=None,
                 **kwargs
        ):
        super(CulaneDataset, self).__init__(
            data_root,
            data_list,
            pipeline,
            test_mode,
            test_suffix,
            work_dir,
            **kwargs
        )
        self.evaluator = LaneMetricCore(
                eval_width=224,
                eval_height=224,
                iou_thresh=0.5,
                lane_width=5
            )
        self.evaluate_data_list = data_list
    
    def set_ori_shape(self, idx):
        ori_filename = self.img_infos[idx]['raw_file']
        filename = os.path.join(self.img_prefix, ori_filename)
        img_tmp = cv2.imread(filename)
        ori_shape = img_tmp.shape
        self.img_infos[idx]['ori_shape']=ori_shape
        
    def set_all_scenes(self):
        pass

    def prepare_train_img(self, idx):
        ori_filename = self.img_infos[idx]['raw_file']
        filename = os.path.join(self.img_prefix, ori_filename)
        img_tmp = cv2.imread(filename)
        ori_shape = img_tmp.shape
        # 不要太高的部分
        if ori_shape == (1440, 2560, 3):
            img = np.zeros((800, 2560, 3), np.uint8)
            img[:800, :, :] = img_tmp[640:, ...]
            offset_x = 0
            offset_y = -640
        elif ori_shape == (660, 1570, 3):
            img = np.zeros((480, 1570, 3), np.uint8)
            img[:480, :, :] = img_tmp[180:, ...]
            offset_x = 0
            offset_y = -180
        elif ori_shape == (720, 1280, 3):
            img = np.zeros((352, 1280, 3), np.uint8)
            img[:352, :, :] = img_tmp[368:, ...]
            offset_x = 0
            offset_y = -368 
        else:
            assert False
            return None
        img_shape = img.shape
        kps, id_classes, id_instances = self.load_labels(
            idx, offset_x, offset_y)

        gt_points_for_show = copy.deepcopy(kps)
        for laneidx in range(len(kps)):
            for i in range(len(kps[laneidx])):
                if i%2==0:
                    gt_points_for_show[laneidx][i] -= offset_x
                else:
                    gt_points_for_show[laneidx][i] -= offset_y

        results = dict(filename=filename,
                       ori_filename=ori_filename,
                       img=img,
                       gt_points=kps,
                       gt_points_for_show=gt_points_for_show,
                       id_classes=id_classes,
                       id_instances=id_instances,
                       img_shape=img_shape,
                       ori_shape=ori_shape,
                       img_info=self.img_infos[idx],
                       offset_x = offset_x,
                       offset_y = offset_y
                  )
        self.img_infos[idx]['ori_shape']=ori_shape
        results = self.pipeline(results)
        results['img'] = [results['img']] # 为了可视化方便
        results['img_metas'] = [results['img_metas']]
        return results

    def prepare_test_img(self, idx):
        return self.prepare_train_img(idx)
    
    def evaluate(self, outputs, **eval_kwargs):
        self.evaluator.reset()
        # format和culane完全一样
        pr_dir = self.format_results(outputs) if outputs else self.work_dir + '/format'
        gt_dir = self.data_root
        for idx in tqdm(range(len(self.img_infos))):
            raw_file = self.img_infos[idx]['raw_file']
            pr_anno = os.path.join(pr_dir,raw_file) 
            gt_anno = os.path.join(gt_dir,raw_file) 
            pr = parse_anno(pr_anno) 
            gt = parse_anno(gt_anno) 
            if 'ori_shape' not in self.img_infos[idx]:
                self.set_ori_shape(idx)
            ori_shape = self.img_infos[idx]['ori_shape']
            gt_wh = dict(height=ori_shape[0], width=ori_shape[1]) # (1440, 2560, 3)
            predict_spec = dict(Lines=pr, Shape=gt_wh) # gt_wh：{'height': 1440, 'width': 2560}
            target_spec = dict(Lines=gt, Shape=gt_wh)
            self.evaluator(target_spec, predict_spec)

        metric = self.evaluator.summary()
        if self.check_or_not: 
            z_metric = self.check(pr_dir)
            metric.update(**z_metric)
        return metric


def convert_coords_formal(lanes):
    res = []
    for lane in lanes:
        lane_coords = []
        for coord in lane:
            lane_coords.append({'x': coord[0], 'y': coord[1]})
        res.append(lane_coords)
    return res

def parse_anno(filename, formal=True):
    anno_dir = filename.replace('.jpg', '.lines.txt') 
    annos = []
    with open(anno_dir, 'r') as anno_f:
        lines = anno_f.readlines()
    for line in lines: # '734.02 1439.0 815.89 1366.21 897.76 1293.43 979.62 1220.64 1061.49 1147.86 1061.86 1147.53 1178.79 1046.65 1254.7 973.04 1284.6 920.13 1307.6 890.22 1452.51 812.01 \n'
        coords = []
        numbers = line.strip().split(' ')
        coords_tmp = [float(n) for n in numbers]

        for i in range(len(coords_tmp) // 2):
            coords.append((coords_tmp[2 * i], coords_tmp[2 * i + 1]))
        annos.append(coords)
    if formal: # true
        annos = convert_coords_formal(annos)
    return annos