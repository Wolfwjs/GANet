import numpy as np
import ujson as json
from sklearn.linear_model import LinearRegression

class LaneEval(object):
    lr = LinearRegression()
    pixel_thresh = 20
    pt_thresh = 0.85

    @staticmethod
    def get_angle(xs, y_samples):
        xs, ys = xs[xs >= 0], y_samples[xs >= 0]
        if len(xs) > 1:
            LaneEval.lr.fit(ys[:, None], xs)
            k = LaneEval.lr.coef_[0]
            theta = np.arctan(k)
        else:
            theta = 0
        return theta

    @staticmethod
    def line_accuracy(pred, gt, thresh):
        pred = np.array([p if p >= 0 else -100 for p in pred])
        gt = np.array([g if g >= 0 else -100 for g in gt])
        # 都是固定数量的xs，用-2填充
        return np.sum(np.where(np.abs(pred - gt) < thresh, 1., 0.)) / len(gt)

    @staticmethod
    def distances(pred, gt):
        return np.abs(pred - gt)

    @staticmethod
    def bench(pred, gt, y_samples, running_time, get_matches=False):
        '''gt里面可能是有全-2的，pred和gt的length都是h_sample的
        '''
        if any(len(p) != len(y_samples) for p in pred):
            raise Exception('Format of lanes error.')
        if running_time > 20000 or len(gt) + 2 < len(pred):
            if get_matches:
                return 0., 0., 1., [False] * len(pred), [0] * len(pred), [None] * len(pred)
            return 0., 0., 1.,
        angles = [LaneEval.get_angle(np.array(x_gts), np.array(y_samples)) for x_gts in gt] # 4,[-0.75, 0.92, 1.29, 1.37]
        threshs = [LaneEval.pixel_thresh / np.cos(angle) for angle in angles] # 4,[27.48, 33.06, 72.65, 103.83]
        line_accs = []
        fp, fn = 0., 0.
        matched = 0.
        my_matches = [False] * len(pred)
        my_accs = [0] * len(pred)
        my_dists = [None] * len(pred)
        for x_gts, thresh in zip(gt, threshs):
            accs = [LaneEval.line_accuracy(np.array(x_preds), np.array(x_gts), thresh) for x_preds in pred] # [0.98, 0.17, 0.17, 0.16],对应其中有一个的概率是非常大的，水平上的差值阈值是要cos的
            my_accs = np.maximum(my_accs, accs) # 在4个位置都取最大值
            max_acc = np.max(accs) if len(accs) > 0 else 0.
            my_dist = [LaneEval.distances(np.array(x_preds), np.array(x_gts)) for x_preds in pred] # 4个(56,)的list
            if len(accs) > 0:
                my_dists[np.argmax(accs)] = {
                    'y_gts': list(np.array(y_samples)[np.array(x_gts) >= 0].astype(int)), # x_gts大于等于0，这个后面还不知道怎么用
                    'dists': list(my_dist[np.argmax(accs)])
                }

            if max_acc < LaneEval.pt_thresh: # 0.85
                fn += 1
            else:
                my_matches[np.argmax(accs)] = True
                matched += 1
            line_accs.append(max_acc) # 是统计每个预测的lane与几个gt之间的精确度和距离，得到精度最高的
        fp = len(pred) - matched # 有几个没预测出来

        if len(gt) > 4 and fn > 0: # 若gt数量大于4（最多是5个车道),有负的预测为正的情况下，todo
            fn -= 1

        s = sum(line_accs) # 3.875

        if len(gt) > 4: # 5条车道，todo不知道和vil100精度差有无关系
            s -= min(line_accs)
            
        if get_matches: 
            # # 这样会让结果值特别高
            # return s / len(gt) if len(gt)>0 else s, \
            #     fp / len(pred) if len(pred)>0 else fp, \
            #     fn / len(gt) if len(gt)>0 else fn, \
            #     my_matches, \
            #     my_accs, \
            #     my_dists ## todo
            return s / max(min(4.0, len(gt)), 1.), \
                fp / len(pred) if len(pred) > 0 else 0., \
                fn / max(min(len(gt), 4.), 1.), \
                my_matches, \
                my_accs, \
                my_dists
        return s / max(min(4.0, len(gt)), 1.), \
                fp / len(pred) if len(pred) > 0 else 0., \
                fn / max(min(len(gt), 4.), 1.)

    @staticmethod
    def bench_one_submit(pred_file, gt_file):
        try:
            json_pred = [json.loads(line) for line in open(pred_file).readlines()]
        except BaseException as e:
            raise Exception('Fail to load json file of the prediction.')
        json_gt = [json.loads(line) for line in open(gt_file).readlines()]
        if len(json_gt) != len(json_pred):
            raise Exception('We do not get the predictions of all the test tasks')
        gts = {img['raw_file']: img for img in json_gt}
        accuracy, fp, fn = 0., 0., 0.
        run_times = []
        for pred in json_pred:
            if 'raw_file' not in pred or 'lanes' not in pred or 'run_time' not in pred:
                raise Exception('raw_file or lanes or run_time not in some predictions.')
            raw_file = pred['raw_file']
            pred_lanes = pred['lanes']
            run_time = pred['run_time']
            run_times.append(run_time)
            if raw_file not in gts:
                import pdb;pdb.set_trace()
                raise Exception('Some raw_file from your predictions do not exist in the test tasks.')
            gt = gts[raw_file]
            gt_lanes = gt['lanes']
            y_samples = gt['h_samples']
            try:
                a, p, n = LaneEval.bench(pred_lanes, gt_lanes, y_samples, run_time)
            except BaseException as e:
                raise Exception('Format of lanes error.')
            accuracy += a
            fp += p
            fn += n
        num = len(gts)
        
        pr = 1 - fp / num
        re = 1 - fn / num
        f1 = 2 * pr * re / (pr + re)

        return dict(
            Accuracy=accuracy / num,
            FP=fp / num,
            FN=fn / num,
            F1=f1,
        )

if __name__ == '__main__':
    import sys
    pr = sys.argv[1]
    gt = sys.argv[2]
    print(LaneEval.bench_one_submit(pr,gt))