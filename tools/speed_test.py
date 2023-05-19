# Copyright (c) OpenMMLab. All rights reserved.
import sys
sys.path.insert(0,os.path.abspath("./"))
import argparse
import os
os.environ['OMP_NUM_THREADS']="1"
os.environ['MKL_NUM_THREADS']="1"
import cv2
import mmcv
import torch

from mmcv import Timer
from mmcv.runner import load_checkpoint

from mmdet.models import build_detector
from mmdet.datasets.pipelines import Normalize

from projects.plugin import *

torch.backends.cudnn.benchmark = True
torch.backends.cudnn.deterministic = True

SIZE = (800, 320)

def parse_args():
    parser = argparse.ArgumentParser(description='MMDet test detector')
    parser.add_argument('config', help='test config file path')
    parser.add_argument(
        'checkpoint', default=None, help='test config file path')
    parser.add_argument('--local_rank', type=int, default=0)
    parser.add_argument(
        '--launcher',
        choices=['none', 'pytorch', 'slurm', 'mpi'],
        default='none',
        help='job launcher')
    args = parser.parse_args()
    if 'LOCAL_RANK' not in os.environ:
        os.environ['LOCAL_RANK'] = str(args.local_rank)
    return args


def main():
    args = parse_args()

    cfg = mmcv.Config.fromfile(args.config)
    if cfg.get('cudnn_benchmark', False):
        torch.backends.cudnn.benchmark = True
    # cfg.model.pretrained = None
    cfg.model.post_processing.crop_bbox = None
    model = build_detector(cfg.model)
    if args.checkpoint is not None:
        load_checkpoint(model, args.checkpoint, map_location='cpu')

    filename = 'tools/test.jpg'
    img = cv2.imread(filename)
    img = cv2.resize(img, SIZE)

    norm = Normalize(**cfg.img_norm_cfg)
    img = norm({'img':img})['img']
    x = torch.unsqueeze(torch.from_numpy(img).permute(2, 0, 1), 0) # torch.Size([1, 3, 320, 800])
    model = model.cuda().eval()
    x = x.cuda()
    img_metas = [dict(img_shape=SIZE[::-1],filename=filename)]

    # warm up
    for i in range(1000):
        model.forward_test(x,img_metas)

    with Timer("Elapsed time in all model infernece: %f"):
        for i in range(1000):
            model.forward_test(x,img_metas)

if __name__ == '__main__':
    main()