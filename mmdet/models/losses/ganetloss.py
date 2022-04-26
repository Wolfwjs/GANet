import math

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from mmdet.models.builder import LOSSES, build_loss
from torch.utils.tensorboard import SummaryWriter
from .smooth_l1_loss import SmoothL1Loss

class CrossEntropyLoss(nn.Module):

    def __init__(self):
        super(CrossEntropyLoss, self).__init__()
        self.crit = nn.BCEWithLogitsLoss()

    def forward(self, logit, target, mask=None):
        if mask is not None:
            logit = logit[mask]
            target = target[mask]
        loss = self.crit(logit, target)
        return loss


def _neg_loss(pred, gt, channel_weights=None):
    ''' Modified focal loss. Exactly the same as CornerNet.
      Runs faster and costs a little bit more memory
      Arguments:
      pred (batch x c x h x w)
      gt_regr (batch x c x h x w)
    '''
    pos_inds = gt.eq(1).float()
    neg_inds = gt.lt(1).float()

    neg_weights = torch.pow(1 - gt, 4)

    loss = 0

    pos_loss = torch.log(pred) * torch.pow(1 - pred, 2) * pos_inds
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
        loss = loss - (pos_loss + neg_loss) / 256
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
        loss = F.l1_loss(output * mask, target * mask, size_average=False)
        mask = mask.bool().float()
        loss = loss / (mask.sum() + 1e-4)
        return loss


def compute_locations(shape, device):
    pos = torch.arange(
        0, shape[-1], step=1, dtype=torch.float32, device=device)
    pos = pos.reshape((1, 1, 1, -1))
    pos = pos.repeat(shape[0], shape[1], shape[2], 1)
    return pos

@LOSSES.register_module
class LaneLossAggress(torch.nn.Module):
    def __init__(self,
                 loss_reg=dict(
                     type='SmoothL1Loss', beta=1.0 / 9.0, loss_weight=1.0)):
        super(LaneLossAggress, self).__init__()
        self.focalloss = FocalLoss()
        self.smoothl1loss = build_loss(loss_reg)
        self.regl1kploss = RegL1KpLoss()

    def forward(self, outputs, weight=1.0):
        loss_result = {}
        for i, loss_item in enumerate(outputs):
            loss_func = getattr(self, loss_item['type'])
            if "mask" in loss_item:
                loss_result[f"{i}_{loss_item['type']}"] = loss_func(loss_item['pred'], loss_item['gt'], loss_item['mask'])*loss_item.get('weight', 1.0)
            else:
                loss_result[f"{i}_{loss_item['type']}"] = loss_func(loss_item['pred'], loss_item['gt'])*loss_item.get('weight', 1.0)
        return loss_result