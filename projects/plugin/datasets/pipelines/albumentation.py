"""
    albumentation interface.
"""
import random
import collections

import albumentations as al
import numpy as np

from mmdet.datasets import PIPELINES


@PIPELINES.register_module()
class albumentation(object):

    def __init__(self, pipelines):
        assert isinstance(pipelines, collections.abc.Sequence)
        # init as None
        self.__augmentor = None
        # put transforms in a list
        self.transforms = []
        self.bbox_params = None
        self.keypoint_params = None

        for transform in pipelines:
            if isinstance(transform, dict):
                if transform['type'] == 'Compose':
                    self.get_al_params(transform['params']) # set self.keypoint_params, not real compose
                else:
                    transform = self.build_transforms(transform)
                    if transform is not None:
                        self.transforms.append(transform)
            else:
                raise TypeError('transform must be a dict')
        self.build()

    def get_al_params(self, compose): # {'bboxes': False, 'keypoints': True, 'masks': False}
        if compose['bboxes']:
            self.bbox_params = al.BboxParams(
                format='pascal_voc',
                min_area=0.0,
                min_visibility=0.0,
                label_fields=["bbox_labels"])
        if compose['keypoints']:
            self.keypoint_params = al.KeypointParams(
                format='xy', remove_invisible=False)

    def build_transforms(self, transform): # dict_items([('type', 'Crop'), ('x_min', 0), ('x_max', 1280), ('y_min', 160), ('y_max', 720), ('p', 1)])
        if transform['type'] == 'OneOf':
            transforms = transform['transforms']
            choices = []
            for t in transforms:
                parmas = {
                    key: value
                    for key, value in t.items() if key != 'type'
                }
                choice = getattr(al, t['type'])(**parmas)
                choices.append(choice)
            return getattr(al, 'OneOf')(transforms=choices, p=transform['p'])

        parmas = {
            key: value
            for key, value in transform.items() if key != 'type'
        }
        return getattr(al, transform['type'])(**parmas) # {'x_min': 0, 'x_max': 1280, 'y_min': 160, 'y_max': 720, 'p': 1}

    def build(self):
        if len(self.transforms) == 0:
            return
        self.__augmentor = al.Compose(
            self.transforms,
            bbox_params=self.bbox_params,
            keypoint_params=self.keypoint_params,
        )

    def cal_sum_list(self, itmes, index):
        sum = 0
        for i in range(index):
            sum += itmes[i]
        return sum

    def __call__(self, data):
        if self.__augmentor is None:
            return data
        img = data['img']

        if 'gt_points' in data:
            points = data["gt_points"] # culane format
            p_group_num = len(points)
            # run aug
            points_index = []
            for k in points:
                points_index.append(int(len(k) / 2)) # [23, 16, 9, 23, 20]
            points_val = []
            for pts in points:
                num = int(len(pts) / 2) # [23, 16, 9, 23, 20]
                for i in range(num):
                    points_val.append(pts[2 * i:2 * i + 2]) # xy format, all lanes
            # num_keypoints = len(points_val) // 2 # half
            keypoints_val = None
            if keypoints_val is None:
                keypoints_val = points_val
            else:
                keypoints_val = keypoints_val + points_val

        aug = self.__augmentor(
            image=img,
            keypoints=keypoints_val,
            bboxes=None,
            mask=None,
            bbox_labels=None)
        data['img'] = aug['image']
        data['img_shape'] = data['img'].shape # (320, 800, 3)

        if 'gt_points' in data:
            start_idx = 0 # 0
            points = aug['keypoints'][start_idx:]
            kp_list = [[0 for j in range(i * 2)] for i in points_index]
            for i in range(len(points_index)): # [23, 16, 9, 23, 20]
                for j in range(points_index[i]):
                    kp_list[i][2 * j] = points[self.cal_sum_list(points_index, i) + j][0]
                    kp_list[i][2 * j + 1] = points[self.cal_sum_list(points_index, i) + j][1]
            data['gt_points'] = kp_list # xy format,split lanes

        return data

    def __repr__(self):
        format_string = self.__class__.__name__ + '('
        for t in self.transforms:
            format_string += '\n'
            format_string += '    {0}'.format(t)
        format_string += '\n)'
        return format_string
