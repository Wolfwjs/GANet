import torch
from scipy.optimize import linear_sum_assignment

from ..builder import BBOX_ASSIGNERS
from .assign_result import AssignResult
from .base_assigner import BaseAssigner


@BBOX_ASSIGNERS.register_module()
class LaneAssigner():
    """Assign a corresponding gt bbox or background to each point.

    Each proposals will be assigned with `0`, or a positive integer
    indicating the ground truth index.

    - 0: negative sample, no assigned gt
    - positive integer: positive sample, index (1-based) of assigned gt

    """

    def __init__(self):
        pass

    def sample_idx(self, num, s_num, device="cpu"):
        assert num %(s_num-1)==1
        gap = num // (s_num-1)
        return torch.arange(0, num, gap).to(device)

    def generate_grid(self, points_map):
        b, p, h, w = points_map.shape
        y = torch.arange(h)[:, None, None].repeat(1, w, 1)
        x = torch.arange(w)[None, :, None].repeat(h, 1, 1)
        coods = torch.cat([y, x], dim=-1)[None, :, :, :].repeat(b, 1, 1, p//2).float()
        grid = coods.reshape(b, h, w, p).permute(0, 3, 1, 2).to(points_map.device)
        return grid

    def assign(self, points_map, gt_points, sample_gt_points=None):
        """
            points -> b, p*2, h, w
            gt_points -> b, l, 50, 2 float
        """
        # device = gt_points.device y,x
        b, p, h, w = points_map.shape
        b, l, g, _ = gt_points.shape
        if sample_gt_points is not None:
            sample_idx = self.sample_idx(g, sample_gt_points, device=points_map.device)
        # generate cood grid
        grid = self.generate_grid(points_map)
        # get the abs position in feature map, according to gt_points
        points_map = points_map.contiguous() + grid.contiguous()
        gt_points = gt_points.to(points_map.device).contiguous()
        gt_points_int = gt_points.long()
        # filter the vaild lane, case the lane num is align to six
        lane_valid_mask = (gt_points_int[:, :, 0, 0] > 0).long()[:, :, None, None, None]
        gt_points_int = gt_points_int.reshape(b, -1, 2).contiguous()
        assert p % 2 == 0
        p = p // 2
        points_map = points_map.reshape(b, p, 2, h, w).contiguous()
        points_cat = []
        # get the point on each lane
        for i in range(b):
            gt_points_int_y = gt_points_int[i, :, 0] # l*g
            gt_points_int_x = gt_points_int[i, :, 1]
            points_cat.append(points_map[i, :, :, gt_points_int_y, gt_points_int_x].reshape([1, p, 2, l, g]).permute(0, 3, 1, 2, 4).contiguous())

        points = torch.cat(points_cat, dim=0) # b, l, p, 2, k
        points_ = points[:, :, :, None, ...].contiguous() # b, l, p, 1, 2, k
        gt_points  = gt_points[:, :, sample_idx, :]
        gt_points_ = gt_points[:, :, None, :, ..., None].contiguous() # b, l, 1, g, 2, 1
        # compute the distance cost
        cost = ((gt_points_-points_)**2).sum(-2).detach().cpu() # compute the distance bt pred and gt [b, l, p, g, k]
        # print(cost.shape)
        # bimatch
        indices = [[[linear_sum_assignment(cost[b_, l_, ..., g_])[1] for g_ in range(g)] for l_ in range(l)] for b_ in range(b)] # b, l, k, (x, y)
        # align the point bt gt and pred
        gt_points_match = torch.cat(
                            [torch.cat(
                                [gt_points[b_, l_, torch.tensor(indices[b_][l_])].unsqueeze(0) for l_ in range(l)], 
                                dim=0).unsqueeze(0) for b_ in range(b)], 
                            dim=0)
        points_match = points.contiguous().permute(0, 1, 4, 2, 3) # b, l, k, p, g
        return gt_points_match*lane_valid_mask, points_match*lane_valid_mask