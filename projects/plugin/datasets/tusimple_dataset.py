import os
import json

import cv2
import numpy as np
import mmcv
from mmcv import ProgressBar

from mmdet.datasets.custom import CustomDataset
from mmdet.datasets.pipelines import Compose
from mmdet.datasets.builder import DATASETS
from mmdet.utils import get_root_logger
from shapely.geometry import Polygon, Point, LineString, MultiLineString, MultiPoint
from shapely.geometry.collection import GeometryCollection

from tools.ganet.tusimple.tusimple_evaluate import LaneEval

@DATASETS.register_module()
class TuSimpleDataset(CustomDataset):

    def __init__(self,
                 data_root,
                 data_list,
                 pipeline,
                 test_mode=False,
                 test_suffix='png',
                 work_dir=None,
                 **kwargs, # evalcore
                 ):
        self.img_prefix = data_root
        self.test_suffix = test_suffix
        self.test_mode = test_mode
        self.work_dir = work_dir
        self.data_root = data_root
        self.data_list = data_list
        self.logger = get_root_logger(log_level='INFO')

        # read image list
        self.img_infos = self.parser_datalist(data_list) # [dict_keys(['lanes', 'h_samples', 'raw_file']),...]
        # set group flag for the sampler
        if not self.test_mode: # test模式不设置group怎么办 todo
            self._set_group_flag()
        self.pipeline = Compose(pipeline) # 4, other transpose
        self.check_or_not = False
    
    def set_all_scenes(self):
        pass
    
    def parser_datalist(self, data_list):
        img_infos = []
        for anno_file in data_list:
            json_gt = [json.loads(line) for line in open(anno_file)]
            img_infos += json_gt
        return img_infos
    
    def _set_group_flag(self):
        self.flag = np.zeros(len(self), dtype=np.uint8)
        for i in range(len(self)):
            self.flag[i] = 1

    def __len__(self):
        return len(self.img_infos)

    def prepare_train_img(self, idx):
        ori_filename = self.img_infos[idx]['raw_file']
        filename = os.path.join(self.img_prefix, ori_filename)
        img = cv2.imread(filename)
        ori_shape = img.shape
        offset_x = 0
        offset_y = 0
        img_shape = img.shape
        kps, id_classes, id_instances = self.load_labels(idx, offset_x, offset_y)
        results = dict(filename=filename,
                       ori_filename=ori_filename,
                       img=img,
                       gt_points=kps,
                       gt_points_for_show=kps,
                       id_classes=id_classes,
                       id_instances=id_instances,
                       img_shape=img_shape,
                       ori_shape=ori_shape,
                       img_info=self.img_infos[idx],
                       offset_x = offset_x,
                       offset_y = offset_y
                  )
        results = self.pipeline(results)
        results['img'] = [results['img']] # 为了可视化方便
        results['img_metas'] = [results['img_metas']]
        return results
    
    def prepare_test_img(self, idx):
        return self.prepare_train_img(idx)

    def load_labels(self, idx, offset_x, offset_y):
        lanes = []
        for lane in self.img_infos[idx]['lanes']:
            coords = []
            for coord_x, coord_y in zip(lane, self.img_infos[idx]['h_samples']):
                if coord_x >= 0:
                    coord_x = float(coord_x)
                    coord_y = float(coord_y)
                    coord_x += offset_x
                    coord_y += offset_y
                    coords.append(coord_x)
                    coords.append(coord_y)
            if len(coords) > 3:
                lanes.append(coords)   # lane nums per img
        id_classes = [1 for i in range(len(lanes))]
        id_instances = [i+1 for i in range(len(lanes))]
        return lanes, id_classes, id_instances
    
    def format_results(self, outputs, **kwargs):
        save_file = self.work_dir + '/format/tusimple_format.json'
        if outputs==None:
            return save_file
        mmcv.mkdir_or_exist(os.path.dirname(save_file))
        f_pr = open(save_file,'w')
        bar = ProgressBar(len(outputs))
        for idx, output in enumerate(outputs):
            result, virtual_center, cluster_center = output['final_dict_list']
            pred = self.img_infos[idx] # 准备target的时候的info
            h_samples = pred['h_samples']
            tusimple_lanes = tusimple_convert_formal(
                result, h_samples, output['img_metas']['ori_shape'][1])
            tusimple_sample = dict(
                lanes=tusimple_lanes,
                h_samples=h_samples,
                raw_file=pred['raw_file'],
                run_time=20)
            json.dump(tusimple_sample, f_pr)
            f_pr.write('\n')
            bar.update()

        f_pr.close()
        print(f"\nwriting tusimple results to {save_file}")
        return save_file

    def evaluate(self, outputs, **eval_kwargs):
        pr_file = self.format_results(outputs)
        assert len(self.data_list) == 1
        gt_file = self.data_list[0]
        metric = LaneEval.bench_one_submit(pr_file, gt_file)
        if self.check_or_not: 
            z_metric = self.check(pr_file)
            metric.update(**z_metric)
        return metric
    
    def open_check(self):
        self.check_or_not=True
    
    def check_one_img(self,lanes):
        znum = 0
        for i,lane1 in enumerate(lanes):
            if len(lane1)<2:
                continue
            lane1 = LineString(lane1)
            for lane2 in lanes[i+1:]:
                if len(lane2)<2:
                    continue
                lane2 = LineString(lane2)
                ps = lane1.intersection(lane2)
                if not ps.is_empty:
                    if isinstance(ps,Point):
                        continue
                    elif isinstance(ps,MultiPoint):
                        znum += len(ps.geoms)
                    elif isinstance(ps,LineString):
                        znum += 1
                    elif isinstance(ps,MultiLineString):
                        znum += len(ps.geoms)
                    elif isinstance(ps,GeometryCollection):
                        znum += 1
                    else:
                        znum += 1
                        print(type(ps))
        return znum
        
    def check(self, pr_file): # outputs是一个一个图
        znum = 0
        znum_img = 0
        demo = []
        items = mmcv.list_from_file(pr_file)
        for item in items:
            pred = json.loads(item)
            path = os.path.join(self.data_root,pred['raw_file'])
            lanes = [[(x,y) for (x,y) in zip(lane,pred['h_samples']) if x>0] for lane in pred['lanes']]
            ret = self.check_one_img(lanes)
            if ret>0:
                znum+=ret
                znum_img+=1
                demo.append(path)
        ret_demo = demo if len(demo)>0 else ''
        return dict(
            znum = znum,
            znum_img = znum_img,
            demo = ret_demo,
            z_ratio = znum_img / len(items)
        )
        
def tusimple_convert_formal(lanes, h_samples, im_width, reg_x=-2):
    lane_samples = []
    for lane in lanes:
        lane_sample = []
        for h in h_samples:
            x = get_line_intersection(h, lane, im_width, reg_x)
            lane_sample.append(x)
        lane_samples.append(lane_sample)
    return lane_samples

def get_line_intersection(y, line, im_width, reg_x=-2):

    def in_line_range(val, start, end):
        s = min(start, end)
        e = max(start, end)
        if s <= val <= e and s != e:
            return True
        else:
            return False

    # reg_x = -2
    for i in range(len(line) - 1):
        point_start, point_end = line[i], line[i + 1]
        if in_line_range(y, point_start[1], point_end[1]):
            k = (point_end[0] - point_start[0]) / (point_end[1] - point_start[1])
            reg_x = int(k * (y - point_start[1]) + point_start[0] + 0.49999)
            break ## todo

    return reg_x

# /data2/hrz/datasets/tusimple/clips/0530/1492626760788443246_0/20.jpg