import sys,os
sys.path.insert(0,'/data1/hrz/myGANet/')
from mmcv import Config
from mmdet.datasets import build_dataset
import argparse

from projects.plugin import *

# 不重新生成预测，直接用原来生成的预测
def test_culane(metric=None):
    metric = {'cross_F1': 0.0, 'cross_precise': 0.0, 'cross_recall': -1.0, 'cross_FP': 1450.0, 'cross_FN': 0.0, 'cross_TP': 0.0, 'curve_F1': 0.780026, 'curve_precise': 0.896142, 'curve_recall': 0.690549, 'curve_FP': 105.0, 'curve_FN': 406.0, 'curve_TP': 906.0, 'hlight_F1': 0.730211, 'hlight_precise': 0.821826, 'hlight_recall': 0.656973, 'hlight_FP': 240.0, 'hlight_FN': 578.0, 'hlight_TP': 1107.0, 'shadow_F1': 0.737683, 'shadow_precise': 0.814364, 'shadow_recall': 0.6742, 'shadow_FP': 442.0, 'shadow_FN': 937.0, 'shadow_TP': 1939.0, 'arrow_F1': 0.901693, 'arrow_precise': 0.935179, 'arrow_recall': 0.870522, 'arrow_FP': 192.0, 'arrow_FN': 412.0, 'arrow_TP': 2770.0, 'noline_F1': 0.526034, 'noline_precise': 0.703452, 'noline_recall': 0.420084, 'noline_FP': 2483.0, 'noline_FN': 8131.0, 'noline_TP': 5890.0, 'night_F1': 0.743549, 'night_precise': 0.833038, 'night_recall': 0.671422, 'night_FP': 2830.0, 'night_FN': 6910.0, 'night_TP': 14120.0, 'crowd_F1': 0.782913, 'crowd_precise': 0.843105, 'crowd_recall': 0.730743, 'crowd_FP': 3808.0, 'crowd_FN': 7540.0, 'crowd_TP': 20463.0, 'normal_F1': 0.93521, 'normal_precise': 0.94817, 'normal_recall': 0.922598, 'normal_FP': 1653.0, 'normal_FN': 2537.0, 'normal_TP': 30240.0, 'test_F1': 0.792077, 'test_precise': 0.854333, 'test_recall': 0.738278, 'test_FP': 13203.0, 'test_FN': 27451.0, 'test_TP': 77435.0, 'znum': 0, 'znum_img': 0, 'demo': '', 'z_ratio': 0.0}
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

# if __name__ == '__main__':
#     parser = argparse.ArgumentParser()
#     parser.add_argument('config', help='test config file path')
#     args = parser.parse_args()
#     metric = do_eval(args.config)
#     if 'culane' in args.config:
#         test_culane(metric)
#     elif 'curvelane' in args.config:
#         test_curvelane(metric)
#     elif 'tusimple' in args.config:
#         test_tusimple(metric)
# test_tusimple()
# OMP_NUM_THREADS
test_culane()
# 16,891s