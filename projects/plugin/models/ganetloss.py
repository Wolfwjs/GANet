import torch
import torch.nn as nn
import torch.nn.functional as F
from mmdet.models.builder import LOSSES, build_loss
from mmdet.core import build_assigner

def _neg_loss(pred, gt, channel_weights=None):
    ''' Modified focal loss. Exactly the same as CornerNet.
      Runs faster and costs a little bit more memory
      Arguments:
      pred (batch x c x h x w)
      gt_regr (batch x c x h x w)
    '''
    # neg and pos indexes
    pos_inds = gt.eq(1).float()
    neg_inds = gt.lt(1).float()

    # for neg_index, 1-gt = pt, y = 4
    neg_weights = torch.pow(1 - gt, 4)

    loss = 0

    # for pos, y = 2
    pos_loss = torch.log(pred) * torch.pow(1 - pred, 2) * pos_inds

    # for net, y = 2, need mul neg_weights
    neg_loss = torch.log(1 - pred) * torch.pow(pred, 2) * neg_weights * neg_inds

    num_pos = pos_inds.float().sum()
    if channel_weights is None:
        pos_loss = pos_loss.sum()
        neg_loss = neg_loss.sum()
    else:
        pos_loss_sum = 0
        neg_loss_sum = 0
        for i in range(len(channel_weights)):
            p = pos_loss[:, i, :, :].sum() * channel_weights[i]
            n = neg_loss[:, i, :, :].sum() * channel_weights[i]
            pos_loss_sum += p
            neg_loss_sum += n
        pos_loss = pos_loss_sum
        neg_loss = neg_loss_sum
    if num_pos > 2:
        loss = loss - (pos_loss + neg_loss) / num_pos
    else:
        # loss = loss - (pos_loss + neg_loss) / 256
        loss = torch.tensor(0, dtype=torch.float32).to(pred.device)
    return loss


class FocalLoss(nn.Module):
    '''nn.Module warpper for focal loss'''

    def __init__(self):
        super(FocalLoss, self).__init__()
        self.neg_loss = _neg_loss

    def forward(self, out, target, weights_list=None):
        return self.neg_loss(out, target, weights_list)


class RegL1KpLoss(nn.Module):

    def __init__(self):
        super(RegL1KpLoss, self).__init__()

    def forward(self, output, target, mask):
        loss = F.l1_loss(output * mask, target * mask, reduction='sum')
        mask = mask.bool().float()
        loss = loss / (mask.sum() + 1e-4)
        return loss


def compute_locations(shape, device):
    pos = torch.arange(
        0, shape[-1], step=1, dtype=torch.float32, device=device)
    pos = pos.reshape((1, 1, 1, -1))
    pos = pos.repeat(shape[0], shape[1], shape[2], 1)
    return pos

@LOSSES.register_module()
class LaneLossAggress(torch.nn.Module):
    def __init__(self,
                 loss_weights,
                 use_smooth,
                 deconv_layer,
                 assigner_cfg,
                 sample_gt_points,
                 point_scale
                ):
        super(LaneLossAggress, self).__init__()
        self.loss_weights=loss_weights
        self.use_smooth=use_smooth
        self.deconv_layer=deconv_layer
        self.sample_gt_points=sample_gt_points
        self.assigner=build_assigner(assigner_cfg)
        self.point_scale=point_scale
        loss_reg=dict(type='SmoothL1Loss', 
                      beta=1.0 / 9.0, 
                      loss_weight=1.0)
        self.focalloss = FocalLoss() 
        self.smoothl1loss = build_loss(loss_reg) 
        self.regl1kploss = RegL1KpLoss()
        
    def get_loss_item(self, pred, target, deform_points):
        kpts_hm = pred['kpts_hm']
        # kpts_hm = torch.clamp(torch.sigmoid(kpts_hm), min=1e-4, max=1 - 1e-4) 已经做过
        # 为什么要进行sigmoid，再限制在0~1呢，是要一个他是否为关键点的分数
        pts_offset = pred['pts_offset']
        int_offset = pred['int_offset']
        # pts_offset和int_offset的监督，只有关键点才有
        
        gt_kpts_hm = target['gt_kpts_hm']
        gt_int_offset = target['gt_int_offset']
        gt_pts_offset= target['gt_pts_offset']
        gt_hm_lanes = target['gt_hm_lanes']

        loss_items = [
            dict( # 关键点损失
                type="focalloss",
                gt = gt_kpts_hm,
                pred = kpts_hm,
                weight=self.loss_weights["point"]
            ),
            dict( # int_offset损失
                type="regl1kploss" if not self.use_smooth else "smoothl1loss",
                gt = gt_int_offset,
                pred = int_offset,
                mask = target['offset_mask'],
                weight = self.loss_weights["error"],
            ),
            dict( # pts_offset损失
                type="regl1kploss" if not self.use_smooth else "smoothl1loss",
                gt = gt_pts_offset,
                pred = pts_offset,
                mask = target['offset_mask_weight'], 
                weight = self.loss_weights["offset"]
            )
        ]
        
        if self.loss_weights["aux"] != 0:
            for i,has_dconv in enumerate(self.deconv_layer):
                if has_dconv==False:
                    continue
                assert gt_hm_lanes[i]!=None
                assert deform_points[i]!=None
                gt_matched_points, pred_matched_points = self.assigner.assign(
                    deform_points[i], 
                    gt_hm_lanes[i], 
                    self.sample_gt_points[i],
                )
                scale = 2 ** (3 - i) if self.point_scale else 1
                # scale表示的是下采样尺度
                loss_items.append(
                    dict(
                        type = "smoothl1loss", 
                        gt = gt_matched_points / scale,
                        pred = pred_matched_points / scale, 
                        weight = self.loss_weights["aux"]
                    )
                )
        return loss_items

    def forward(self, pred, target, deform_points):
        outputs = self.get_loss_item(pred, target, deform_points)
        loss_result = {}
        for i, loss_item in enumerate(outputs):
            loss_func = getattr(self, loss_item['type'])
            if "mask" in loss_item:
                loss_result[f"{i}_{loss_item['type']}"] = loss_func(
                    loss_item['pred'], 
                    loss_item['gt'], 
                    loss_item['mask'])*loss_item.get('weight', 1.0)
            else:
                loss_result[f"{i}_{loss_item['type']}"] = loss_func(
                    loss_item['pred'], 
                    loss_item['gt'])*loss_item.get('weight', 1.0)
        return loss_result