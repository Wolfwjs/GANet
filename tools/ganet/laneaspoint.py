import os
os.chdir('..')

import argparse
import glob
import cv2
import numpy as np
import torch
import matplotlib.pyplot as plt

from tusimple.evaluate.lane import LaneEval
from tusimple.test_dataset import *

from mmcv import Timer

def generate_grid(points_map):
    b, p, h, w = points_map.shape
    y = torch.arange(h)[:, None, None].repeat(1, w, 1)
    x = torch.arange(w)[None, :, None].repeat(h, 1, 1)
    # b, h, w, p
    coods = torch.cat([y, x], dim=-1)[None, :, :, None, :].repeat(b, 1, 1, p//2, 1).float()
    # b, p, h, w
    grid = coods.reshape(b, h, w, p).permute(0, 3, 1, 2).to(points_map.device)
    return grid

parser = argparse.ArgumentParser(
        description='MMDet test (and eval) a model')
parser.add_argument('config_name', default="pointMerge_id3")
args = parser.parse_args()

config = f"configs/magiclanenet/tusimple/{args.config_name}.py"
cfg = Config.fromfile(config)
cfg.data.samples_per_gpu = 1
gt_path = '/mnt/lustre/wangjinsheng/project/lane-detection/conditional-lane-detection/datasets/tusimple/test_label.json'
pred_path = sorted(glob.glob(f'tools/output/tusimple/{args.config_name}_2021*/result/test.json'))[-1]
# p: 多检测的 n: 少检测的
criterias, _ = LaneEval.bench_one_submit(pred_path, gt_path, return_each=True)
bad_p = torch.arange(len(criterias['p']))[torch.tensor(criterias['p']) > 0]
bad_n = torch.arange(len(criterias['n']))[torch.tensor(criterias['n']) > 0]
# print(bad_p)
# print(bad_n)

dataset = build_dataset(cfg.data.test)
data_loader = build_dataloader(
    dataset,
    samples_per_gpu=1,
    workers_per_gpu=cfg.data.workers_per_gpu,
    dist=False,
    shuffle=False)
checkpoint = sorted(glob.glob(f'tools/output/tusimple/{args.config_name}_2021*/latest.pth'))[-1]
print(checkpoint)
model = build_detector(cfg.model, train_cfg=cfg.train_cfg, test_cfg=cfg.test_cfg)
model.load_state_dict(torch.load(checkpoint, map_location='cpu')['state_dict'], strict=True)
model.eval()
model = MMDataParallel(model.cuda(), device_ids=[0])

for i, data in enumerate(data_loader):
    image   = data['img'].data[0].cuda()
    thr     = 0.3
    kpt_thr = 0.3
    cpt_thr = 0.3
    b = image.shape[0]
    results = model.module.test_inference(image, thr=thr, kpt_thr=kpt_thr, cpt_thr=cpt_thr)
    nearpoints = results['deform_points'][0]
    lanepointmask = (results['kpts_hm'] > kpt_thr)[0, 0]
    lanepoints = nearpoints[:, :, lanepointmask].permute(0, 2, 1)
    b, n, p = lanepoints.shape
    p = p//2
    dists = ((lanepoints.reshape(1, 1, -1, 2) - lanepoints.reshape(1, -1, 1, 2))**2).sum(dim=-1).reshape(b, n, p, n, p).permute(0, 1, 3, 2, 4)
    mean_dist = dists.mean(dim=(3, 4))
    min_dist = dists.min(dim=3).values.min(dim=3).values
    mean_scale = p*(p-1)/2
    dist_loss = torch.exp(mean_scale-mean_dist-min_dist) * (1-torch.eye(n)[None].cuda())
    adj_matrix = dist_loss == dist_loss.max(dim=2, keepdim=True).values
    plt.imsave("debug/adj_matrix.png", adj_matrix[0].detach().cpu())
    
    print(adj_matrix.shape)
    print(adj_matrix.sum(dim=-1))
