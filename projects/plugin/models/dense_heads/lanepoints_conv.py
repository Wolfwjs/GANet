import numpy as np
import torch
import torch.nn as nn
from mmcv.cnn import ConvModule, bias_init_with_prob, normal_init

from ...ops.dcn import DeformConv, DeformConv1D
from mmdet.models import HEADS, build_loss
from mmcv import Timer

@HEADS.register_module()
class LanePointsConv(nn.Module):
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
                 in_channels,
                 feat_channels=256,
                 point_feat_channels=256,
                 stacked_convs=3, # repeat conv
                 num_points=9,
                 gradient_mul=0.1, # reduce gradient
                 use_latern=False,
                 conv_cfg=None,
                 norm_cfg=None,
                 train_cfg=None,
                 test_cfg=None):
        super(LanePointsConv, self).__init__()
        self.in_channels = in_channels # 64
        self.feat_channels = feat_channels # 256
        self.point_feat_channels = point_feat_channels # 64
        self.stacked_convs = stacked_convs # 3
        self.num_points = num_points # 7
        self.gradient_mul = gradient_mul # 0.1
        self.use_latern = use_latern # f
        self.conv_cfg = conv_cfg # n
        self.norm_cfg = norm_cfg # n
        self.train_cfg = train_cfg # n
        self.test_cfg = test_cfg # n
        # we use deformable conv to extract points features
        self.dcn_kernel = num_points # 7
        self.dcn_pad = int((self.dcn_kernel - 1) / 2) # 3
        assert self.dcn_kernel % 2 == 1, \
            'The points number should be an odd number.'
        # kernel position [-1, 0, 1]
        dcn_base = np.arange(-self.dcn_pad,
                             self.dcn_pad + 1).astype(np.float64) # 7,
        # 
        dcn_base_y = np.repeat(0, self.dcn_kernel) # [0, 0, 0] (7,)
        dcn_base_x = np.tile(dcn_base, 1)   # [-1, 0, 1] (7,)
        dcn_base_offset = np.stack([dcn_base_y, dcn_base_x], axis=1).reshape(
            (-1)) # (14,) [0.0, -3.0, 0.0, -2.0, 0.0, -1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 2.0, 0.0, 3.0]
        self.dcn_base_offset = torch.tensor(dcn_base_offset).reshape(1, -1, 1, 1) # torch.Size([1, 14, 1, 1])
        self._init_layers()

    def _init_layers(self):
        self.relu = nn.ReLU(inplace=False)
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
        for i in range(3):
            chn = self.in_channels if i == 0 else self.feat_channels
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
        if self.use_latern:
            self.implicit_cls_add = nn.Parameter(torch.zeros(1, self.in_channels, 1, 1))
            nn.init.normal_(self.implicit_cls_add, std=.02)
            self.implicit_pts_add = nn.Parameter(torch.zeros(1, self.in_channels, 1, 1))
            nn.init.normal_(self.implicit_pts_add, std=.02)
            self.implicit_cls_mul = nn.Parameter(torch.ones(1 , self.in_channels, 1, 1))
            nn.init.normal_(self.implicit_cls_mul, mean=1., std=.02)
            self.implicit_pts_mul = nn.Parameter(torch.ones(1 , self.in_channels, 1, 1))
            nn.init.normal_(self.implicit_pts_mul, mean=1., std=.02)

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

    def forward_single(self, x): # torch.Size([32, 64, 40, 100])
        # x -> single layer feature
        # [-1,-1, -1,1, ..., 1,0, 1,1], shape([1, -1, 1, 1])
        dcn_base_offset = self.dcn_base_offset.type_as(x) # torch.Size([1, 14, 1, 1])
        # reg from kernel grid
        points_init = 0 # reg from center
        if self.use_latern: # false
            cls_feat = x*self.implicit_cls_mul+self.implicit_cls_add # no such attri
            pts_feat = x*self.implicit_pts_mul+self.implicit_pts_add
        else:
            cls_feat = x
            pts_feat = x
            
        for cls_conv in self.cls_convs: # len=3, torch.Size([32, 64, 40, 100])-> torch.Size([32, 256, 40, 100])-> torch.Size([32, 256, 40, 100])-> torch.Size([32, 256, 40, 100])
            cls_feat = cls_conv(cls_feat)
        for reg_conv in self.reg_convs: # same as above
            pts_feat = reg_conv(pts_feat)
        # initialize reppoints
        pts_out_init = self.reppoints_pts_init_out(
            self.relu(self.reppoints_pts_init_conv(pts_feat))) # feat_channels-> point_feat_channels -> pts_out_dim(2 * self.num_points), namely 256-> 64-> 14, torch.Size([32, 14, 40, 100])
        # regress from center
        # get kernel position from center
        pts_out_init = pts_out_init + points_init # 0
        # refine and classify reppoints
        # reduce the gradient for pts_out_init
        pts_out_init_grad_mul = (1 - self.gradient_mul) * pts_out_init.detach(
        ) + self.gradient_mul * pts_out_init # torch.Size([32, 14, 40, 100]), do mix
        # dcn_base_offset [-1,-1, -1,1, ..., 1,0, 1,1], shape([1, -1, 1, 1])
        # diff between real position and init position, as the input of deformable convolution
        dcn_offset = pts_out_init_grad_mul.contiguous() - dcn_base_offset.contiguous()
        # deformable convolution, feature aggretation from given points
        feature_out = self.relu(self.reppoints_cls_conv(cls_feat, dcn_offset.contiguous())) # feature_out
        
        return feature_out, pts_out_init

    def forward(self, feats):
        return self.forward_single(feats)
        