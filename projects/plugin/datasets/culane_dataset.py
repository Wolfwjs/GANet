import os
from time import sleep
import numpy as np
import mmcv
from mmcv import ProgressBar
import subprocess
import re
import datetime
import time
import threading

from mmdet.datasets.builder import DATASETS
from .tusimple_dataset import TuSimpleDataset

@DATASETS.register_module()
class CulaneDataset(TuSimpleDataset):

    def __init__(self,
                 data_root,
                 data_list,
                 evaluate_data_list_1,
                 evaluate_data_list_s,
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
        self.evaluate_data_list_1=evaluate_data_list_1
        self.evaluate_data_list_s=evaluate_data_list_s
        self.evaluate_data_list = self.evaluate_data_list_1
    
    def set_all_scenes(self):
        self.evaluate_data_list = self.evaluate_data_list_s

    # 重载函数
    def parser_datalist(self, data_list):
        img_infos = []
        for anno_list in data_list:
            with open(anno_list) as f:
                lines = f.readlines()
                for line in lines:
                    raw_file = line.strip()
                    raw_file = raw_file[1:] if raw_file[0]=='/' else raw_file
                    img_info = dict(raw_file=raw_file)
                    # if self.test_mode == False: # 不论模式都加载anno
                    raw_anno_file = raw_file.replace('.jpg', '.lines.txt')
                    img_info.update(dict(anno_file=raw_anno_file))
                    img_infos.append(img_info)
        return img_infos

    def _set_group_flag(self):
        self.flag = np.zeros(len(self), dtype=np.uint8) # 0
        for i in range(len(self)):
            self.flag[i] = 1

    def __len__(self):
        return len(self.img_infos)

    def load_labels(self, idx, offset_x, offset_y):
        anno_file = self.img_prefix + "/" + self.img_infos[idx]['anno_file']
        lanes = []
        with open(anno_file, 'r') as anno_f:
            lines = anno_f.readlines() # [],会有空的
            for line in lines:
                coords = []
                coords_str = line.strip().split(' ')
                for i in range(len(coords_str) // 2):
                    coord_x = float(coords_str[2 * i]) + offset_x ## todo
                    coord_y = float(coords_str[2 * i + 1]) + offset_y
                    coords.append(coord_x)
                    coords.append(coord_y)
                if len(coords) > 3:
                    lanes.append(coords)
        id_classes = [1 for i in range(len(lanes))]
        id_instances = [i + 1 for i in range(len(lanes))]
        return lanes, id_classes, id_instances

    def format_results(self, outputs, **kwargs):
        save_dir = self.work_dir + '/format'
        bar = ProgressBar(len(outputs))
        for idx, output in enumerate(outputs):
            result, virtual_center, cluster_center = output['final_dict_list']
            culane_lanes = culane_convert_formal(result)
            save_file = save_dir + '/' + output['img_metas']['ori_filename'].replace('.jpg', '.lines.txt')
            mmcv.mkdir_or_exist(os.path.dirname(save_file))
            f_pr = open(save_file,'w')
            for lane in culane_lanes:
                for v in lane:
                    f_pr.write(str(v)+' ')
                f_pr.write('\n')
            bar.update()

        f_pr.close()
        print(f"\nwriting culane results to {save_dir}")
        return save_dir

    # def evaluate_p(self, outputs, **eval_kwargs):
    #     pr_dir = self.format_results(outputs) if outputs else self.work_dir + '/format'
    #     gt_dir = self.data_root
    #     pr_dir = pr_dir+'/' if pr_dir[-1]!='/' else pr_dir
    #     gt_dir = gt_dir+'/' if gt_dir[-1]!='/' else gt_dir
    #     w_lane=30;
    #     iou=0.5  # Set iou to 0.3 or 0.5
    #     im_w=1640
    #     im_h=590
    #     frame=1
    #     F1_pa = re.compile("Fmeasure: (.*)")
    #     PR_pa = re.compile("precision: (.*)")
    #     RE_pa = re.compile("recall: (.*)")
    #     FP_pa = re.compile("fp: (\d*)")
    #     FN_pa = re.compile("fn: (\d*)")
    #     TP_pa = re.compile("tp: (\d*)")
    #     metric_pa = {"F1":F1_pa,
    #                 "precise":PR_pa,
    #                 "recall":RE_pa,
    #                 "FP":FP_pa,
    #                 "FN":FN_pa,
    #                 "TP":TP_pa,
    #                 }
    #     ret = {}
    #     starttime = datetime.datetime.now()
    #     for data_list in self.evaluate_data_list:
    #         print(f"evaluating {data_list}...")
    #         name = os.path.basename(data_list).split('.')[0].split('_')[-1]
    #         p = subprocess.Popen(f"tools/ganet/culane/culane_evaluate/evaluate -a {gt_dir} -d {pr_dir} -i {gt_dir} -l {data_list} -w {w_lane} -t {iou} -c {im_w} -r {im_h} -f {frame}", stdout=subprocess.PIPE, shell=True)
    #         p.wait()
    #         output = p.stdout.read().decode('utf-8')
    #         # print(output)
    #         for k in metric_pa:
    #             val = re.findall(metric_pa[k],output)[0]
    #             ret[f"{name}_{k}"] = float(val)
    #     endtime = datetime.datetime.now()
    #     print(f"{(endtime-starttime).seconds}s elapse...")
    #     if self.check_or_not: 
    #         z_metric = self.check(pr_dir)
    #         ret.update(**z_metric)
    #     return ret


    def evaluate(self, outputs, **eval_kwargs):
        pr_dir = self.format_results(outputs) if outputs else self.work_dir + '/format'
        gt_dir = self.data_root
        print(f"pr:{pr_dir}\ngt:{gt_dir}")
        pr_dir = pr_dir+'/' if pr_dir[-1]!='/' else pr_dir
        gt_dir = gt_dir+'/' if gt_dir[-1]!='/' else gt_dir
        ret = {}
        starttime = datetime.datetime.now()
        t_list = []
        for data_list in self.evaluate_data_list:
            t = threading.Thread(target=eval_one, args=(data_list,gt_dir,pr_dir,ret))
            # t.setDaemon(True)
            t.start()
            t_list.append(t)
        for t in t_list:
            t.join()
        endtime = datetime.datetime.now()
        print(f"{(endtime-starttime).seconds}s elapse...")
        if self.check_or_not: 
            z_metric = self.check(pr_dir)
            ret.update(**z_metric)
        return ret

    def check(self, pr_dir):
        znum = 0
        znum_img = 0
        demo = []
        self.img_info = self.parser_datalist(self.evaluate_data_list) # 避免重新测试的时候没有img_info
        for img_info in self.img_infos:
            path = os.path.join(pr_dir,img_info['anno_file'])
            annos = mmcv.list_from_file(path)
            lanes = []
            for anno in annos:
                str_list = anno.split()
                lanes.append([(float(str_list[2*i]),float(str_list[2*i+1])) for i in range(len(str_list)//2)])
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
            z_ratio = znum_img / len(self.img_infos)
        )

def culane_convert_formal(lanes):
    res = []
    for lane in lanes:
        lane_coords = []
        sety = set()
        for coord in lane:
            x = round(coord[0])
            y = round(coord[1])
            if y not in sety:
                lane_coords.append(x)
                lane_coords.append(y)
                sety.add(y)
        res.append(lane_coords)
    return res

def parse_anno(filename):
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
    return annos

def eval_one(data_list,gt_dir,pr_dir,ret):
    F1_pa = re.compile("Fmeasure: (.*)")
    PR_pa = re.compile("precision: (.*)")
    RE_pa = re.compile("recall: (.*)")
    FP_pa = re.compile("fp: (\d*)")
    FN_pa = re.compile("fn: (\d*)")
    TP_pa = re.compile("tp: (\d*)")
    metric_pa = {"F1":F1_pa,
                "precise":PR_pa,
                "recall":RE_pa,
                "FP":FP_pa,
                "FN":FN_pa,
                "TP":TP_pa,
                }
    w_lane=30;
    iou=0.5  # Set iou to 0.3 or 0.5
    im_w=1640
    im_h=590
    frame=1
    print(f"evaluating {data_list}...")
    name = os.path.basename(data_list).split('.')[0].split('_')[-1]
    p = subprocess.Popen(f"tools/ganet/culane/culane_evaluate/evaluate -a {gt_dir} -d {pr_dir} -i {gt_dir} -l {data_list} -w {w_lane} -t {iou} -c {im_w} -r {im_h} -f {frame}", stdout=subprocess.PIPE, shell=True)
    p.wait()
    output = p.stdout.read().decode('utf-8')
    # print(output)
    for k in metric_pa:
        val = re.findall(metric_pa[k],output)[0]
        ret[f"{name}_{k}"] = float(val)