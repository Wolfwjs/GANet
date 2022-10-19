import random
import math
import copy
from functools import cmp_to_key

import PIL.ImageDraw
import numpy as np
from mmcv.parallel import DataContainer as DC

from mmdet.datasets import PIPELINES
from mmdet.datasets.pipelines import Collect, to_tensor
import scipy.interpolate as spi
from shapely.geometry import Polygon, Point, MultiPoint, LineString, MultiLineString


def gaussian2D(shape, sigma=1):
    m, n = [(ss - 1.) / 2. for ss in shape]
    y, x = np.ogrid[-m:m + 1, -n:n + 1]
    h = np.exp(-(x * x + y * y) / (2 * sigma * sigma))
    h[h < np.finfo(h.dtype).eps * h.max()] = 0
    return h


def draw_umich_gaussian(heatmap, center, radius, k=1):
    diameter = 2 * radius + 1
    gaussian = gaussian2D((diameter, diameter), sigma=diameter / 6)
    x, y = int(center[0]), int(center[1])
    height, width = heatmap.shape[0:2]
    left, right = min(x, radius), min(width - x, radius + 1)
    top, bottom = min(y, radius), min(height - y, radius + 1)
    masked_heatmap = heatmap[y - top:y + bottom, x - left:x + right]
    masked_gaussian = gaussian[radius - top:radius + bottom, radius - left:radius + right]

    if min(masked_gaussian.shape) > 0 and min(
            masked_heatmap.shape) > 0:  # TODO debug
        np.maximum(masked_heatmap, masked_gaussian * k, out=masked_heatmap)
    return heatmap


def convert_scale(p, downscale=None):
    xy = list()
    if downscale is None:
        for i in range(len(p) // 2):
            xy.append((p[2 * i], p[2 * i + 1]))
    else:
        for i in range(len(p) // 2):
            xy.append((p[2 * i] / downscale, p[2 * i + 1] / downscale))
    return xy


def ploy_fitting_cube_extend(line, h, w, sample_num=100):
    # Y->X
    line_coords = np.array(line).reshape((-1, 2))
    line_coords = np.array(sorted(line_coords, key=lambda x: x[1]))
    line_coords = line_coords[line_coords[:, 0] > 0, :]
    line_coords = line_coords[line_coords[:, 0] < w, :]  # type ndarry
    # print('line coord {}'.format(line_coords[-5:, :]))
    if line_coords.shape[0] < 2:
        return None
    line_coords_extend = extend_line(line_coords, dis=25)
    # print('extend line coord {}'.format(line_coords_extend[-5:, :]))

    X = line_coords_extend[:, 1]
    Y = line_coords_extend[:, 0]
    if len(X) < 2:
        return None

    new_x = np.linspace(max(X[0], 0), min(X[-1], h), sample_num)

    if len(X) > 3:
        ipo3 = spi.splrep(X, Y, k=3)
        iy3 = spi.splev(new_x, ipo3)
    else:
        ipo3 = spi.splrep(X, Y, k=1)
        iy3 = spi.splev(new_x, ipo3)
    return np.concatenate([iy3[:, None], new_x[:, None]], axis=1)


def ploy_fitting_cube(line, h, w, sample_num=100, use_before=True):
    line_coords = np.array(line).reshape((-1, 2))
    line_coords = np.array(sorted(line_coords, key=lambda x: x[1]))
    line_coords = line_coords[line_coords[:, 0] > 0, :]
    line_coords = line_coords[line_coords[:, 0] < w, :]

    X = line_coords[:, 1]
    Y = line_coords[:, 0]

    if len(X) < 2:
        return None
    new_x = np.linspace(max(X[0], 0), min(X[-1], h), sample_num)

    setx = set()
    nX,nY=[],[]
    for i,(x,y) in enumerate(zip(X,Y)):
        if x in setx:
            continue
        setx.add(x)
        nX.append(x)
        nY.append(y)
    if len(nX) < 2:
        return None

    if len(nX) > 3:
        ipo3 = spi.splrep(nX, nY, k=3)
        iy3 = spi.splev(new_x, ipo3)
    else:
        ipo3 = spi.splrep(nX, nY, k=1)
        iy3 = spi.splev(new_x, ipo3)
    return np.concatenate([iy3[:, None], new_x[:, None]], axis=1)


def clamp_line(line, box, min_length=0):
    left, top, right, bottom = box
    loss_box = Polygon([[left, top], [right, top], [right, bottom],
                        [left, bottom]])
    line_coords = np.array(line).reshape((-1, 2))  # -1 points nums
    if line_coords.shape[0] < 2:
        return None
    try:
        line_string = LineString(line_coords)
        I = line_string.intersection(loss_box)  # line intersection with box
        if I.is_empty:
            return None
        if I.length < min_length:
            return None
        if isinstance(I, LineString):
            pts = list(I.coords)
            # pts_ = list(line_string.coords)
            # print('pts after intersection {}'.format(pts))
            return pts
        elif isinstance(I, MultiLineString):
            pts = []
            Istrings = I.geoms
            for Istring in Istrings:
                pts += list(Istring.coords)
            return pts
    except:
        return None

def cal_dis(p1, p2):
    return math.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)

def clip_line(pts, h, w):
    pts_x = np.clip(pts[:, 0], 0, w - 1)[:, None]
    pts_y = np.clip(pts[:, 1], 0, h - 1)[:, None]
    # 这里怎么能出错呢，我真是服了自己了
    # 没有对y按照要求进行clip
    return np.concatenate([pts_x, pts_y], axis=-1)

def extend_line(line, dis=10):
    extended = copy.deepcopy(line)
    start = line[-2]
    end = line[-1]
    dx = end[0] - start[0]
    dy = end[1] - start[1]
    # print('dy ========== {}'.format(dy))
    norm = math.sqrt(dx ** 2 + dy ** 2)
    dx = dx / norm  # cos(theta)
    dy = dy / norm  # sin(theta)
    extend_point = np.array((end[0] + dx * dis, end[1] + dy * dis)).reshape(1, 2)
    extended = np.append(extended, extend_point, axis=0)
    return extended

@PIPELINES.register_module
class CollectLanePoints(Collect):
    def __init__(
            self,
            max_lane_num,
            lane_extend=False,
            keys=['img', 'targets'],
            gaussian_hyper=dict(
                radius=2,
                root_radius=4,
            ),
            dis_weights=[1, 0.4, 0.2],
            fpn_cfg=dict(
                hm_idx=None,
                fpn_down_scale=None,
                sample_per_lane=None,
                deconv_layer=None,
            ),
            interval = None,
    ):
        self.keys = keys
        self.gaussian_hyper = gaussian_hyper
        self.max_lane_num = max_lane_num

        self.fpn_down_scale = fpn_cfg.get('fpn_down_scale')
        self.hm_idx = fpn_cfg.get('hm_idx')
        self.deconv_layer = fpn_cfg.get('deconv_layer')
        self.sample_per_lane = fpn_cfg.get('sample_per_lane')

        self.dis_weights = dis_weights
        self.hm_down_scale = self.fpn_down_scale[self.hm_idx]
        self.fpn_layer_num = len(self.fpn_down_scale)
        self.lane_extend = lane_extend
        
        self.interval = interval
    
    def assign_weight(self, start_pt, pt, max_dis):
        dis = cal_dis(start_pt, pt)
        return 1-dis/(max_dis+1e-6)
    
    def get_targets(self, results):
        def assign_weight(dis, h, joints, weights=None):
            if weights is None:
                weights = [1, 0.4, 0.2]
            step = h // joints
            weight = 1
            if dis < 0:
                weight = weights[2]
            elif dis < 2 * step:
                weight = weights[0]
            else:
                weight = weights[1]
            return weight

        output_h = int(results['img_shape'][0])
        output_w = int(results['img_shape'][1])
        mask_h = int(output_h // self.hm_down_scale)
        mask_w = int(output_w // self.hm_down_scale)
        hm_h = int(output_h // self.hm_down_scale)
        hm_w = int(output_w // self.hm_down_scale)
        results['hm_shape'] = [hm_h, hm_w]
        results['mask_shape'] = [mask_h, mask_w]

        # gt init
        kpts_hm = np.zeros((1, hm_h, hm_w), np.float32)
        int_offset = np.zeros((2, hm_h, hm_w), np.float32)  # down sample offset
        pts_offset = np.zeros((2 * 1, hm_h, hm_w), np.float32)  # key points -> center points offset
        offset_mask = np.zeros((1, hm_h, hm_w), np.float32)
        offset_mask_weight = np.zeros((2 * 1, hm_h, hm_w), np.float32)

        # gt heatmap and ins of bank
        gt_points = results['gt_points']  # need some points
        end_points = []
        start_points = []
        gt_hm_lanes = {}
        for l in range(self.fpn_layer_num):
            lane_points = []
            fpn_down_scale = self.fpn_down_scale[l]
            fn_h = int(output_h // fpn_down_scale)
            fn_w = int(output_w // fpn_down_scale)
            for i, pts in enumerate(gt_points):  # per lane
                # pts shape[sample_per_lane, 2] sorted by y
                pts = convert_scale(pts, fpn_down_scale)
                pts = sorted(pts, key=cmp_to_key(lambda a, b: b[-1] - a[-1]))  # down sort by y
                if self.lane_extend:
                    pts = ploy_fitting_cube_extend(pts, fn_h, fn_w, self.sample_per_lane[l])
                else:
                    pts = ploy_fitting_cube(pts, fn_h, fn_w, self.sample_per_lane[l])
                if pts is not None:
                    pts_f = clip_line(pts, fn_h, fn_w)
                    pts = np.int32(pts_f)
                    lane_points.append(pts[None, :, ::-1])  # y, x
            lane_points_align = -1 * np.ones([self.max_lane_num, self.sample_per_lane[l], 2])
            if len(lane_points) != 0:
                lane_points_align[:len(lane_points)] = np.concatenate(lane_points, axis=0)
            else:
                gauss_mask = kpts_hm
            gt_hm_lanes[l] = DC(to_tensor(lane_points_align).float(), stack=True, pad_dims=None)

        # print(len(gt_points))
        for pts in gt_points:  # per lane
            pts = convert_scale(pts, self.hm_down_scale)
            if len(pts) < 2:
                continue

            if self.lane_extend:
                pts = ploy_fitting_cube_extend(pts, hm_h, hm_w, int(360 / self.hm_down_scale))
            else:
                pts = ploy_fitting_cube(pts, hm_h, hm_w, int(360 / self.hm_down_scale))
            if pts is None:
                continue

            pts = sorted(pts, key=cmp_to_key(lambda a, b: b[-1] - a[-1]))  # down sort by y
            pts = clamp_line(pts, box=[0, 0, hm_w - 1, hm_h - 1], min_length=1)

            if pts is not None and len(pts) > 1:
                joint_points = []
                start_point, end_point = pts[0], pts[-1]
                delta_idx = len(pts) // 1
                end_points.append(end_point)
                start_points.append(start_point)
                for i in range(1):
                    joint_points.append(pts[i * delta_idx])

                for pt in pts:
                    pt_int = (int(pt[0]), int(pt[1]))
                    draw_umich_gaussian(kpts_hm[0], pt_int, radius=self.gaussian_hyper['radius'])  # key points
                    reg_x = pt[0] - pt_int[0]
                    reg_y = pt[1] - pt_int[1]
                    int_offset[0, pt_int[1], pt_int[0]] = reg_x  # [C H W]
                    int_offset[1, pt_int[1], pt_int[0]] = reg_y  # [C H W]
                    if abs(reg_x) < 2 and abs(reg_y) < 2:
                        offset_mask[0, pt_int[1], pt_int[0]] = 1  # mask where have points

                    max_x = abs(start_point[0] - end_point[0])
                    max_y = abs(start_point[1] - end_point[1])

                    for i in range(1):
                        offset_x = joint_points[i][0] - pt[0]
                        offset_y = joint_points[i][1] - pt[1]

                        # weight mask
                        mask_value = assign_weight(offset_y, max_y, 1, self.dis_weights)
                        offset_mask_weight[2 * i, pt_int[1], pt_int[0]] = mask_value
                        offset_mask_weight[2 * i + 1, pt_int[1], pt_int[0]] = mask_value

                        pts_offset[2 * i, pt_int[1], pt_int[0]] = offset_x
                        pts_offset[2 * i + 1, pt_int[1], pt_int[0]] = offset_y

        targets = {}
        targets['gt_hm_lanes'] = gt_hm_lanes
        targets['gt_kpts_hm'] = DC(to_tensor(kpts_hm).float(), stack=True, pad_dims=None)
        targets['gt_int_offset'] = DC(to_tensor(int_offset).float(), stack=True, pad_dims=None)
        targets['gt_pts_offset'] = DC(to_tensor(pts_offset).float(), stack=True, pad_dims=None)
        targets['offset_mask'] = DC(to_tensor(offset_mask).float(), stack=True, pad_dims=None)
        targets['offset_mask_weight'] = DC(to_tensor(offset_mask_weight).float(), stack=True, pad_dims=None)

        return targets

    def __call__(self, results):
        targets=self.get_targets(results)
        results.update(targets=targets)
        #####
        data = {}
        img_meta = {}
        for key in results.keys():
            if key not in self.keys:
                img_meta[key] = results[key]
        data['img_metas'] = DC(img_meta, cpu_only=True)
        for key in self.keys:
            data[key] = results[key]
        return data

    def getct(self,circle):
        return circle.bounds[0],circle.bounds[1]+self.interval

    def get_equal_dis_points(self,line):
        line = np.array(line) # 按照y值升序

        X = line[:,1]
        Y = line[:,0]

        setx = set()
        nX,nY=[],[]
        for i,(x,y) in enumerate(zip(X,Y)):
            if x in setx:
                continue
            setx.add(x)
            nX.append(x)
            nY.append(y)

        circle = Point(line[0]).buffer(self.interval)
        linestring = LineString(line)
        pts = []
        while linestring.intersects(circle.boundary):
            pts.append(self.getct(circle))
            ps = linestring.intersection(circle.boundary)
            if isinstance(ps,Point):
                up_p = ps
            elif isinstance(ps,MultiPoint):
                up_p = Point(-1,-1)
                for p in ps.geoms:
                    if p.y > up_p.y:
                        up_p = p
            else:
                assert False
            if up_p.y < self.getct(circle)[-1]:
                break
            circle = up_p.buffer(self.interval)
        pts=np.array(pts)
        if len(pts)<2:
            pts = None
        return pts