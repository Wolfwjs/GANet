import sys,os
sys.path.insert(0,'/data1/hrz/myGANet/')
from mmcv import Config
from mmdet.datasets import build_dataset
import argparse

from projects.plugin import *

# 不重新生成预测，直接用原来生成的预测
def test_culane(metric=None):
    metric = {'cross_F1': 0.0, 'cross_precise': 0.0, 'cross_recall': -1.0, 'cross_FP': 1532.0, 'cross_FN': 0.0, 'cross_TP': 0.0, 'curve_F1': 0.750433, 'curve_precise': 0.869478, 'curve_recall': 0.660061, 'curve_FP': 130.0, 'curve_FN': 446.0, 'curve_TP': 866.0, 'hlight_F1': 0.714676, 'hlight_precise': 0.840964, 'hlight_recall': 0.621365, 'hlight_FP': 198.0, 'hlight_FN': 638.0, 'hlight_TP': 1047.0, 'shadow_F1': 0.797321, 'shadow_precise': 0.88676, 'shadow_recall': 0.72427, 'shadow_FP': 266.0, 'shadow_FN': 793.0, 'shadow_TP': 2083.0, 'arrow_F1': 0.895419, 'arrow_precise': 0.937436, 'arrow_recall': 0.857008, 'arrow_FP': 182.0, 'arrow_FN': 455.0, 'arrow_TP': 2727.0, 'noline_F1': 0.511864, 'noline_precise': 0.750069, 'noline_recall': 0.388489, 'noline_FP': 1815.0, 'noline_FN': 8574.0, 'noline_TP': 5447.0, 'night_F1': 0.72641, 'night_precise': 0.858356, 'night_recall': 0.629624, 'night_FP': 2185.0, 'night_FN': 7789.0, 'night_TP': 13241.0, 'crowd_F1': 0.771689, 'crowd_precise': 0.867407, 'crowd_recall': 0.694997, 'crowd_FP': 2975.0, 'crowd_FN': 8541.0, 'crowd_TP': 19462.0, 'normal_F1': 0.928923, 'normal_precise': 0.94832, 'normal_recall': 0.910303, 'normal_FP': 1626.0, 'normal_FN': 2940.0, 'normal_TP': 29837.0, 'test_F1': 0.784336, 'test_precise': 0.872587, 'test_recall': 0.712297, 'test_FP': 10909.0, 'test_FN': 30176.0, 'test_TP': 74710.0, 'znum': 12, 'znum_img': 3, 'demo': ['./work_dirs/culane/large/format/driver_100_30frame/05251444_0422.MP4/02610.lines.txt', './work_dirs/culane/large/format/driver_193_90frame/06051050_0624.MP4/02610.lines.txt', './work_dirs/culane/large/format/driver_193_90frame/06051059_0627.MP4/00180.lines.txt'], 'z_ratio': 8.650519031141868e-05}
    for t in "test Normal Crowd hlight Shadow Noline Arrow Curve Cross Night".lower().split():
        print(f'{metric[f"{t}_F1"]*100:.2f}',end=' ')

def test_curvelane(metric):
    # metric = {'F1': 0.6372944078947368, 'precision': 0.9030647917961466, 'recall': 0.4923860555743816, 'pr_num': 51488, 'gt_num': 94432, 'znum': 0, 'zimg_num': 0, 'demo': [], 'z_ratio': 0.0} # res18 63.73 90.31 49.24
    # metric = {'F1': 0.6385181299948929, 'precision': 0.9166567590060635, 'recall': 0.4898763131141986, 'pr_num': 50466, 'gt_num': 94432, 'znum': 0, 'zimg_num': 0, 'demo': [], 'z_ratio': 0.0} # res34 63.85 91.67 48.99
    for t in "F1 precision recall".split():
        print(f'{metric[f"{t}"]*100:.2f}',end=' ')

def test_tusimple(metric=None):
    # metric = {'Accuracy': 0.95958874653383, 'FP': 0.01904505152168707, 'FN': 0.026240115025161784, 'F1': 0.9773441746572126, 'znum': 7, 'znum_img': 3, 'demo': ['/data2/hrz/datasets/tusimple/clips/0531/1492638762056680028/20.jpg', '/data2/hrz/datasets/tusimple/clips/0601/1495492750556508768/20.jpg', '/data2/hrz/datasets/tusimple/clips/0601/1494452455562021371/20.jpg'], 'z_ratio': 0.0010783608914450035} # res18
    # metric = {'Accuracy': 0.958885886309942, 'FP': 0.020393002635993328, 'FN': 0.026569614186436642, 'F1': 0.9765089246148977, 'znum': 0, 'znum_img': 0, 'demo': [], 'z_ratio': 0.0} # res34
    # metric = {'Accuracy': 0.9654421707575855, 'FP': 0.028182020471740225, 'FN': 0.021836808051761348, 'F1': 0.9749802621204513, 'znum': 2, 'znum_img': 1, 'demo': ['/data2/hrz/datasets/tusimple/clips/0601/1494453625548430229/20.jpg'], 'z_ratio': 0.0003594536304816679} # res101
    metric = {'Accuracy': 0.9589634469891398, 'FP': 0.02236400670980114, 'FN': 0.027288521447399985, 'F1': 0.9751675188637338, 'znum': 2, 'znum_img': 1, 'demo': ['/data2/hrz/datasets/tusimple/clips/0601/1495492552642880511/20.jpg'], 'z_ratio': 0.0003594536304816679}
    for k in ['F1','Accuracy','FP','FN']:
        print(f"{metric[k]*100:.2f}",end=' ')


def do_eval(cfg):
    # 3个数据集都可以
    cfg = Config.fromfile(cfg)
    dataset = build_dataset(cfg.data.test)
    # dataset.check_or_not = True
    dataset.open_check()
    dataset.set_all_scenes()
    metric = dataset.evaluate(outputs=None)
    print(metric)
    return metric

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('config', help='test config file path')
    args = parser.parse_args()
    metric = do_eval(args.config)
    if 'culane' in args.config:
        test_culane(metric)
    elif 'curvelane' in args.config:
        test_curvelane(metric)
    elif 'tusimple' in args.config:
        test_tusimple(metric)
# # test_tusimple()
# # OMP_NUM_THREADS
# test_culane()
# # 16,891s