import argparse
import os
import numpy as np
import math
import json
import cv2
import copy
import mmcv
import torch
import torch.distributed as dist
import PIL.Image
import PIL.ImageDraw
from mmcv import Config
from mmcv.parallel import MMDataParallel
from mmcv.runner import init_dist, load_checkpoint
from mmdet.datasets import build_dataloader, build_dataset
from mmdet.models import build_detector
from mmdet.utils.general_utils import mkdir, path_join
from tools.ganet.common import tusimple_convert_formal, COLORS
from tools.ganet.post_process import PostProcessor


def parse_args():
    parser = argparse.ArgumentParser(description='MMDet test detector')
    parser.add_argument('config', help='test config file path')
    parser.add_argument('checkpoint', help='seg checkpoint file')
    parser.add_argument('--local_rank', type=int, default=0)
    parser.add_argument('--hm_thr', type=float, default=0.5)
    parser.add_argument('--show', action='store_true')
    parser.add_argument(
        '--show_dst',
        default='./work_dirs/tusimple/watch',
        help='path to save visualized results.')
    parser.add_argument(
        '--result_dst',
        default='./work_dirs/tusimple/results',
        help='path to save results')
    parser.add_argument(
        '--launcher',
        choices=['none', 'pytorch', 'slurm', 'mpi'],
        default='none',
        help='job launcher')
    args = parser.parse_args()
    if 'LOCAL_RANK' not in os.environ:
        os.environ['LOCAL_RANK'] = str(args.local_rank)
    return args


def adjust_result(lanes, centers, crop_bbox, img_shape, points_thr):
    h_img, w_img = img_shape[:2]
    ratio_x = (crop_bbox[2] - crop_bbox[0]) / w_img
    ratio_y = (crop_bbox[3] - crop_bbox[1]) / h_img
    offset_x, offset_y = crop_bbox[:2]

    results = []
    virtual_centers = []
    cluster_centers = []
    if lanes is not None:
        for key in range(len(lanes)):
            pts = []
            cts = []
            for pt in lanes[key]['points']:
                pt[0] = int(pt[0] * ratio_x + offset_x)
                pt[1] = int(pt[1] * ratio_y + offset_y)
                pts.append(tuple(pt))
            for ct in lanes[key]['centers']:
                ct[0] = int(ct[0] * ratio_x + offset_x)
                ct[1] = int(ct[1] * ratio_y + offset_y)
                cts.append(tuple(ct))
            # print('lane {} ====== \npoint nums {}'.format(key, len(pts)))
            # print('lane {} ====== \n point coord {}  \nvirtual center coord {}'.format(key, pts, cts))
            if len(pts) > points_thr:
                results.append(pts)
                virtual_centers.append(cts)
        # print('lane number:{}  virtual center number:{}'.format(len(results), len(virtual_centers)))
    if centers is not None:
        for center in centers:
            center_coord = center['center']
            center_coord[0] = int(center_coord[0] * ratio_x + offset_x)
            center_coord[1] = int(center_coord[1] * ratio_y + offset_y)
            cluster_centers.append(tuple(center_coord))

    return results, virtual_centers, cluster_centers


def out_result(lanes, dst=None):
    if dst is not None:
        with open(dst, 'w') as f:
            for lane in lanes:
                for idx, p in enumerate(lane):
                    if idx == len(lane) - 1:
                        print('{:.2f} '.format(p[0]), end='', file=f)
                        print('{:.2f}'.format(p[1]), file=f)
                    else:
                        print('{:.2f} '.format(p[0]), end='', file=f)
                        print('{:.2f} '.format(p[1]), end='', file=f)


def vis_one(results, virtual_center, cluster_center, filename, img_info=None, lane_width=7):
    def parse_img_info(img_info):
        converted_lanes = []
        h_samples = img_info['h_samples']
        lanes = img_info['lanes']
        for lane in lanes:
            converted_lane = []
            for coord_x, coord_y in zip(lane, h_samples):
                if coord_x >= 0:
                    converted_lane.append((coord_x, coord_y))
            converted_lanes.append(converted_lane)
        return converted_lanes

    img = cv2.imread(filename)
    img_gt = cv2.imread(filename)
    img_vc = cv2.imread(filename)
    img_circle = cv2.imread(filename)
    img_pil = PIL.Image.fromarray(img)
    img_gt_pil = PIL.Image.fromarray(img_gt)
    # img_vc_pil = PIL.Image.fromarray(img_vc)

    for idx, lane in enumerate(results):
        lane_tuple = [tuple(p) for p in lane]
        # print('lane tuple:', lane_tuple)
        PIL.ImageDraw.Draw(img_pil).line(
            xy=lane_tuple, fill=COLORS[idx + 1], width=lane_width)

    for idx, vp in enumerate(virtual_center):
        vp_tuple = [tuple(p) for p in vp]
        # print('vp tuple:', vp_tuple)
        # for _vp in vp_tuple:
        #     cv2.circle(img=img_vc, center=_vp, radius=3, color=COLORS[idx + 1], thickness=-1)
        for _vp in vp_tuple:
            cv2.circle(img=img_circle, center=_vp, radius=3, color=COLORS[idx + 1], thickness=-1)

    for idx, cp in enumerate(cluster_center):
        print(f"{cp}###############################################################")
        cv2.circle(img=img_vc, center=cp, radius=10, color=COLORS[idx + 1], thickness=-1)
        cv2.circle(img=img_circle, center=cp, radius=40, color=COLORS[idx + 1], thickness=3)


    img = np.array(img_pil, dtype=np.uint8)
    # img_vc = np.array(img_vc_pil, dtype=np.uint8)

    if img_info is not None:
        gt_lanes = parse_img_info(img_info)
        for idx, lane in enumerate(gt_lanes):
            lane_tuple = [tuple(p) for p in lane]
            PIL.ImageDraw.Draw(img_gt_pil).line(
                xy=lane_tuple, fill=COLORS[idx + 1], width=lane_width)
        img_gt = np.array(img_gt_pil, dtype=np.uint8)

    return img, img_gt, img_vc, img_circle


def single_gpu_test(seg_model,
                    data_loader,
                    show=None,
                    hm_thr=0.3,
                    kpt_thr=0.4, # 0.3
                    cpt_thr=0.4,
                    points_thr=4, # 3, 5, 7
                    result_dst=None,
                    cluster_by_center_thr=None, # 4, 3, 5
                    group_fast=False,
                    cluster_thr=4, # 3, 4, 5
                    crop_bbox=(0, 160, 1280, 720),
                    save_param=False):
    seg_model.eval()
    dataset = data_loader.dataset
    post_processor = PostProcessor(use_offset=True,
                                           cluster_thr=cluster_thr,
                                           cluster_by_center_thr=cluster_by_center_thr,
                                           group_fast=group_fast)
    prog_bar = mmcv.ProgressBar(len(dataset))
    if save_param:
        if result_dst is not None:
            mkdir(result_dst)
            dst_dir = os.path.join(result_dst, f'test_{kpt_thr}_{points_thr}_{cluster_by_center_thr}_{cluster_thr}.json')
            f_dst = open(dst_dir, 'w')
    else:
        if result_dst is not None:
            mkdir(result_dst)
            dst_dir = os.path.join(result_dst, 'test.json')
            f_dst = open(dst_dir, 'w')
    for i, data in enumerate(data_loader):
        with torch.no_grad():
            sub_name = data['img_metas'].data[0][0]['sub_img_name']
            img_shape = data['img_metas'].data[0][0]['img_shape']
            ori_shape = data['img_metas'].data[0][0]['ori_shape']
            h_samples = data['img_metas'].data[0][0]['h_samples']
            img_info = data['img_metas'].data[0][0]['img_info']
            # seeds, _ = seg_model(
            #     return_loss=False, rescale=False, thr=hm_thr, **data)
            output = seg_model(
                return_loss=False, rescale=False, thr=hm_thr, kpt_thr=kpt_thr, cpt_thr=cpt_thr, **data)
            downscale = data['img_metas'].data[0][0]['hm_down_scale']
            lanes, cluster_centers = post_processor(output, downscale)

            result, virtual_center, cluster_center = adjust_result(
                lanes=lanes, centers=cluster_centers, crop_bbox=crop_bbox,
                img_shape=img_shape, points_thr=points_thr)
            # print('result:', result)
            if result_dst is not None:
                mkdir(result_dst)
                # dst_dir = os.path.join(result_dst, 'test.json')
                tusimple_lanes = tusimple_convert_formal(
                    result, h_samples, ori_shape[1])
                tusimple_sample = dict(
                    lanes=tusimple_lanes,
                    h_samples=h_samples,
                    raw_file=sub_name,
                    run_time=20)
                json.dump(tusimple_sample, f_dst)
                print(file=f_dst)

        if show is not None and show:
            filename = data['img_metas'].data[0][0]['filename']
            img_vis, img_gt_vis, virtual_center_vis, img_circle = vis_one(result, virtual_center, cluster_center, filename, img_info)
            save_name = sub_name.replace('/', '.')
            dst_show_dir = path_join(show, save_name)
            dst_show_gt_dir = path_join(show, save_name + '.gt.jpg')
            dst_show_vc_dir = path_join(show, save_name + '.vc.jpg')
            dst_show_cc_dir = path_join(show, save_name + '.cc.jpg')
            cv2.imwrite(dst_show_dir, img_vis)
            cv2.imwrite(dst_show_gt_dir, img_gt_vis)
            cv2.imwrite(dst_show_vc_dir, virtual_center_vis)
            cv2.imwrite(dst_show_cc_dir, img_circle)

        batch_size = data['img'].data[0].size(0)
        for _ in range(batch_size):
            prog_bar.update()
    if result_dst:
        f_dst.close()


class DateEnconding(json.JSONEncoder):

    def default(self, o):
        if isinstance(o, np.float32):
            return float(o)


def main():
    args = parse_args()

    cfg = mmcv.Config.fromfile(args.config)
    # set cudnn_benchmark
    if cfg.get('cudnn_benchmark', False):
        torch.backends.cudnn.benchmark = True
    cfg.model.pretrained = None

    # init distributed env first, since logger depends on the dist info.
    if args.launcher == 'none':
        distributed = False
    else:
        distributed = True
        init_dist(args.launcher, **cfg.dist_params)

    dataset = build_dataset(cfg.data.test)
    data_loader = build_dataloader(
        dataset,
        samples_per_gpu=1,
        workers_per_gpu=cfg.data.workers_per_gpu,
        dist=distributed,
        shuffle=False)

    # build the model and load checkpoint
    model = build_detector(cfg.model)
    load_checkpoint(model, args.checkpoint, map_location='cpu')
    model = MMDataParallel(model, device_ids=[0])
    if not args.show:
        show_dst = None
    else:
        show_dst = args.show_dst
    if args.show is not None and args.show:
        # mkdir(args.show_dst)
        if not os.path.exists(args.show_dst):
            os.makedirs(args.show_dst)
    print(args.result_dst)
    if isinstance(cfg.points_thr, list):
        for points_thr in cfg.points_thr:
            for cluster_thr in cfg.cluster_thr:
                for cluster_by_center_thr in cfg.cluster_by_center_thr:
                    single_gpu_test(
                        seg_model=model,
                        data_loader=data_loader,
                        show=show_dst,
                        hm_thr=cfg.hm_thr,
                        kpt_thr=cfg.kpt_thr,
                        cpt_thr=cfg.cpt_thr,
                        points_thr=points_thr,
                        result_dst=args.result_dst,
                        cluster_thr=cluster_thr,
                        cluster_by_center_thr=cluster_by_center_thr,
                        group_fast=cfg.group_fast,
                        crop_bbox=cfg.crop_bbox,
                        save_param=True)
    else:
        single_gpu_test(
            seg_model=model,
            data_loader=data_loader,
            show=show_dst,
            hm_thr=cfg.hm_thr,
            kpt_thr=cfg.kpt_thr,
            cpt_thr=cfg.cpt_thr,
            points_thr=cfg.points_thr,
            result_dst=args.result_dst,
            cluster_thr=cfg.cluster_thr,
            cluster_by_center_thr=cfg.cluster_by_center_thr,
            group_fast=cfg.group_fast,
            crop_bbox=cfg.crop_bbox)


if __name__ == '__main__':
    main()
