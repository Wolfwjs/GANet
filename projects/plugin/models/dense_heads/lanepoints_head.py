import numpy as np
import torch
import torch.nn as nn
from mmcv.cnn import ConvModule, bias_init_with_prob, normal_init

from mmdet.core import (PointGenerator, build_assigner, build_sampler,
                        images_to_levels, multi_apply, multiclass_nms, unmap)
from ...ops.dcn import DeformConv, DeformConv1D
from mmdet.models import HEADS, build_loss


@HEADS.register_module()
class LanePointsHead(nn.Module):
    """RepPoint head.

    Args:
        in_channels (int): Number of channels in the input feature map.
        feat_channels (int): Number of channels of the feature map.
        point_feat_channels (int): Number of channels of points features.
        stacked_convs (int): How many conv layers are used.
        gradient_mul (float): The multiplier to gradients from
            points refinement and recognition.
        point_strides (Iterable): points strides.
        point_base_scale (int): bbox scale for assigning labels.
        background_label (int | None): Label ID of background, set as 0 for
            RPN and num_classes for other heads. It will automatically set as
            num_classes if None is given.
        loss_cls (dict): Config of classification loss.
        loss_bbox_init (dict): Config of initial points loss.
        loss_bbox_refine (dict): Config of points loss in refinement.
        use_grid_points (bool): If we use bounding box representation, the
        reppoints is represented as grid points on the bounding box.
        center_init (bool): Whether to use center point assignment.
        transform_method (str): The methods to transform RepPoints to bbox.
    """  # noqa: W605

    def __init__(self,
                 num_classes,
                 in_channels,
                 feat_channels=256,
                 point_feat_channels=256,
                 stacked_convs=3,  # repeat conv
                 num_points=9,
                 gradient_mul=0.1,  # reduce gradient
                 conv_cfg=None,
                 norm_cfg=None,
                 center_init=True,
                 assigner_cfg=None,
                 train_cfg=None,
                 test_cfg=None):
        super(LanePointsHead, self).__init__()
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.feat_channels = feat_channels
        self.point_feat_channels = point_feat_channels
        self.stacked_convs = stacked_convs
        self.num_points = num_points
        self.gradient_mul = gradient_mul
        self.conv_cfg = conv_cfg
        self.norm_cfg = norm_cfg
        self.train_cfg = train_cfg
        self.assigner_cfg = assigner_cfg
        self.cls_out_channels = self.num_classes
        self.assigner_cfg = dict(
            init=dict(
                assigner=dict(type='LaneAssigner')),
            refine=dict(
                assigner=dict(type='LaneAssigner'))
        )
        if self.assigner_cfg:
            self.init_assigner = build_assigner(self.assigner_cfg['init']['assigner'])
            self.refine_assigner = build_assigner(
                self.assigner_cfg['refine']['assigner'])
        self.center_init = center_init
        # we use deformable conv to extract points features
        self.dcn_kernel = num_points
        self.dcn_pad = int((self.dcn_kernel - 1) / 2)
        assert self.dcn_kernel % 2 == 1, \
            'The points number should be an odd number.'
        # kernel position [-1, 0, 1]
        dcn_base = np.arange(-self.dcn_pad,
                             self.dcn_pad + 1).astype(np.float64)
        # 
        dcn_base_y = np.repeat(0, self.dcn_kernel)  # [0, 0, 0]
        dcn_base_x = np.tile(dcn_base, 1)  # [-1, 0, 1]
        dcn_base_offset = np.stack([dcn_base_y, dcn_base_x], axis=1).reshape(
            (-1))  # [0,-1, 0,0, 0,1]
        self.dcn_base_offset = torch.tensor(dcn_base_offset).reshape(1, -1, 1, 1)
        self._init_layers()

    def _init_layers(self):
        self.relu = nn.ReLU(inplace=True)
        self.cls_convs = nn.ModuleList()
        self.reg_convs = nn.ModuleList()
        for i in range(self.stacked_convs):
            chn = self.in_channels if i == 0 else self.feat_channels
            self.cls_convs.append(
                ConvModule(
                    chn,
                    self.feat_channels,
                    3,
                    stride=1,
                    padding=1,
                    conv_cfg=self.conv_cfg,
                    norm_cfg=self.norm_cfg))
            self.reg_convs.append(
                ConvModule(
                    chn,
                    self.feat_channels,
                    3,
                    stride=1,
                    padding=1,
                    conv_cfg=self.conv_cfg,
                    norm_cfg=self.norm_cfg))
        pts_out_dim = 2 * self.num_points
        self.reppoints_cls_conv = DeformConv1D(self.feat_channels,
                                               self.point_feat_channels,
                                               self.dcn_kernel, 1, self.dcn_pad)
        self.reppoints_cls_out = nn.Conv2d(self.point_feat_channels,
                                           self.cls_out_channels, 1, 1, 0)
        self.reppoints_pts_init_conv = nn.Conv2d(self.feat_channels,
                                                 self.point_feat_channels, 3,
                                                 1, 1)
        self.reppoints_pts_init_out = nn.Conv2d(self.point_feat_channels,
                                                pts_out_dim, 1, 1, 0)
        self.reppoints_pts_refine_conv = DeformConv1D(self.feat_channels,
                                                      self.point_feat_channels,
                                                      self.dcn_kernel, 1,
                                                      self.dcn_pad)
        self.reppoints_pts_refine_out = nn.Conv2d(self.point_feat_channels,
                                                  pts_out_dim, 1, 1, 0)

    def init_weights(self):
        for m in self.cls_convs:
            normal_init(m.conv, std=0.01)
        for m in self.reg_convs:
            normal_init(m.conv, std=0.01)
        bias_cls = bias_init_with_prob(0.01)
        normal_init(self.reppoints_cls_conv, std=0.01)
        normal_init(self.reppoints_cls_out, std=0.01, bias=bias_cls)
        normal_init(self.reppoints_pts_init_conv, std=0.01)
        normal_init(self.reppoints_pts_init_out, std=0.01)
        normal_init(self.reppoints_pts_refine_conv, std=0.01)
        normal_init(self.reppoints_pts_refine_out, std=0.01)

    def forward_single(self, x):
        # x -> single layer feature
        # [-1,-1, -1,1, ..., 1,0, 1,1], shape([1, -1, 1, 1])
        dcn_base_offset = self.dcn_base_offset.type_as(x)
        # reg from kernel grid
        points_init = 0  # reg from center
        cls_feat = x
        pts_feat = x
        for cls_conv in self.cls_convs:
            cls_feat = cls_conv(cls_feat)
        for reg_conv in self.reg_convs:
            pts_feat = reg_conv(pts_feat)
        # initialize reppoints
        pts_out_init = self.reppoints_pts_init_out(
            self.relu(self.reppoints_pts_init_conv(pts_feat)))
        # regress from center
        # get kernel position from center
        pts_out_init = pts_out_init + points_init
        # refine and classify reppoints
        # reduce the gradient for pts_out_init
        pts_out_init_grad_mul = (1 - self.gradient_mul) * pts_out_init.detach(
        ) + self.gradient_mul * pts_out_init
        # dcn_base_offset [-1,-1, -1,1, ..., 1,0, 1,1], shape([1, -1, 1, 1])
        # diff between real position and init position, as the input of deformable convolution
        dcn_offset = pts_out_init_grad_mul.contiguous() - dcn_base_offset.contiguous()
        # deformable convolution, feature aggretation from given points
        cls_out = self.reppoints_cls_out(
            self.relu(self.reppoints_cls_conv(cls_feat, dcn_offset.contiguous())))
        # deformable convolution
        pts_out_refine = self.reppoints_pts_refine_out(
            self.relu(self.reppoints_pts_refine_conv(pts_feat, dcn_offset.contiguous())))
        pts_out_refine = pts_out_refine + pts_out_init.detach()
        # channel number -> 1, p*2, p*2
        return cls_out, pts_out_init, pts_out_refine  # score, stage1_point, stage2_point

    def forward(self, feats):
        return multi_apply(self.forward_single, feats)

    def match_target(self, cls_out, pts_out_init, pts_out_refine, gt_cls, gt_points):
        pts_out_init_match, gt_points_init_match = self.init_assigner.assign(pts_out_init, gt_points)
        pts_out_refine_match, gt_points_refine_match = self.refine_assigner.assign(pts_out_refine, gt_points)
        results = {
            'cls_pred': cls_out,
            'cls_gt': gt_cls,
            'pts_init_pred': pts_out_init_match,
            'pts_init_gt': gt_points_init_match,
            'pts_refine_pred': pts_out_refine_match,
            'pts_gt_refine': gt_points_refine_match,
        }
        return results
