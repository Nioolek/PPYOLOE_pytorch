import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from ppyoloe.bbox.utils import batch_distance2bbox, batch_distance2bbox_cxcywh
from ..assigner import ATSSAssigner, TaskAlignedAssigner
from .network_blocks import ConvBNLayer, get_activation
from .losses import VarifocalLoss, FocalLoss, BboxLoss


class ESEAttn(nn.Module):
    def __init__(self, feat_channels, act='swish'):
        super(ESEAttn, self).__init__()
        self.fc = nn.Conv2d(feat_channels, feat_channels, 1)
        self.sig = nn.Sigmoid()
        self.conv = ConvBNLayer(feat_channels, feat_channels, 1, act=act)

        self._init_weights()

    def _init_weights(self):
        nn.init.normal_(self.fc.weight, mean=0, std=0.001)

    def forward(self, feat, avg_feat):
        weight = self.sig(self.fc(avg_feat))
        return self.conv(feat * weight)


class PPYOLOEHead(nn.Module):
    def __init__(self,
                 in_channels=[1024, 512, 256],
                 width_mult=1.0,
                 num_classes=80,
                 act='swish',
                 fpn_strides=(32, 16, 8),
                 grid_cell_scale=5.0,
                 grid_cell_offset=0.5,
                 reg_max=16,
                 static_assigner_epoch=4,
                 use_varifocal_loss=True,
                 static_assigner=ATSSAssigner(9, num_classes=80),
                 assigner=TaskAlignedAssigner(topk=13, alpha=1.0, beta=6.0),
                 eval_input_size=[],
                 loss_weight={
                     'class': 1.0,
                     'iou': 2.5,
                     'dfl': 0.5,
                 },
                 atss_topk=9):
        super(PPYOLOEHead, self).__init__()
        assert len(in_channels) > 0, "len(in_channels) should > 0"
        in_channels = [max(round(c * width_mult), 1) for c in in_channels]
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.fpn_strides = fpn_strides
        self.grid_cell_scale = grid_cell_scale
        self.grid_cell_offset = grid_cell_offset
        self.reg_max = reg_max
        self.loss_weight = loss_weight
        self.use_varifocal_loss = use_varifocal_loss
        self.varifocal_loss = VarifocalLoss().cuda()
        self.focal_loss = FocalLoss().cuda()
        self.bbox_loss = BboxLoss(self.num_classes, self.reg_max).cuda()
        self.eval_input_size = eval_input_size
        self.static_assigner_epoch = static_assigner_epoch
        # stem
        self.stem_cls = nn.ModuleList()
        self.stem_reg = nn.ModuleList()
        act = get_activation(act) if act is None or isinstance(act,
                                                               (str, dict)) else act

        for in_c in self.in_channels:
            self.stem_cls.append(ESEAttn(in_c, act=act))
            self.stem_reg.append(ESEAttn(in_c, act=act))
        # pred head
        self.pred_cls = nn.ModuleList()
        self.pred_reg = nn.ModuleList()
        for in_c in self.in_channels:
            self.pred_cls.append(
                nn.Conv2d(
                    in_c, self.num_classes, 3, padding=1))
            self.pred_reg.append(
                nn.Conv2d(
                    in_c, 4 * (self.reg_max + 1), 3, padding=1))
        # projection conv
        self.proj_conv = nn.Conv2d(self.reg_max + 1, 1, 1, bias=False)
        self._init_weights()

        # ATSS para
        self.atss_topk = atss_topk
        self.atss_assign = static_assigner
        self.assigner = assigner

    def _init_weights(self, prior_prob=0.01):
        for conv in self.pred_cls:
            b = conv.bias.view(-1, )
            b.data.fill_(-math.log((1 - prior_prob) / prior_prob))
            conv.bias = torch.nn.Parameter(b.view(-1), requires_grad=True)
            w = conv.weight
            w.data.fill_(0.)
            conv.weight = torch.nn.Parameter(w, requires_grad=True)

        for conv in self.pred_reg:
            b = conv.bias.view(-1, )
            b.data.fill_(1.0)
            conv.bias = torch.nn.Parameter(b.view(-1), requires_grad=True)
            w = conv.weight
            w.data.fill_(0.)
            conv.weight = torch.nn.Parameter(w, requires_grad=True)

        self.proj = nn.Parameter(torch.linspace(0, self.reg_max, self.reg_max + 1), requires_grad=False)
        self.proj_conv.weight = torch.nn.Parameter(self.proj.view([1, self.reg_max + 1, 1, 1]).clone().detach(),
                                                   requires_grad=False)

        if self.eval_input_size:
            anchor_points, stride_tensor = self._generate_anchors()
            self.register_buffer('anchor_points', anchor_points)
            self.register_buffer('stride_tensor', stride_tensor)

    def forward_train(self, feats, targets, extra_info):
        anchors, anchor_points, num_anchors_list, stride_tensor = \
            self.generate_anchors_for_grid_cell(
                feats, self.fpn_strides, self.grid_cell_scale,
                self.grid_cell_offset, device=targets.device)

        cls_score_list, reg_distri_list = [], []
        for i, feat in enumerate(feats):
            avg_feat = F.adaptive_avg_pool2d(feat, (1, 1))
            cls_logit = self.pred_cls[i](self.stem_cls[i](feat, avg_feat) +
                                         feat)
            reg_distri = self.pred_reg[i](self.stem_reg[i](feat, avg_feat))
            # cls and reg
            cls_score = F.sigmoid(cls_logit)
            cls_score_list.append(cls_score.flatten(2).permute((0, 2, 1)))
            reg_distri_list.append(reg_distri.flatten(2).permute((0, 2, 1)))
        cls_score_list = torch.cat(cls_score_list, axis=1)
        reg_distri_list = torch.cat(reg_distri_list, axis=1)

        return self.get_loss([
            cls_score_list, reg_distri_list, anchors, anchor_points,
            num_anchors_list, stride_tensor
        ], targets, extra_info)

    def forward_eval(self, feats):
        if self.eval_input_size:
            anchor_points, stride_tensor = self.anchor_points, self.stride_tensor
        else:
            anchor_points, stride_tensor = self._generate_anchors(feats, device=feats[0].device)
        cls_score_list, reg_dist_list = [], []
        for i, feat in enumerate(feats):
            b, _, h, w = feat.shape
            l = h * w
            avg_feat = F.adaptive_avg_pool2d(feat, (1, 1))
            cls_logit = self.pred_cls[i](self.stem_cls[i](feat, avg_feat) +
                                         feat)
            reg_dist = self.pred_reg[i](self.stem_reg[i](feat, avg_feat))
            reg_dist = reg_dist.reshape([-1, 4, self.reg_max + 1, l]).permute(
                0, 2, 1, 3)
            reg_dist = self.proj_conv(F.softmax(reg_dist, dim=1))
            # cls and reg
            cls_score = F.sigmoid(cls_logit)
            cls_score_list.append(cls_score.reshape([b, self.num_classes, l]))
            reg_dist_list.append(reg_dist.reshape([b, 4, l]))

        cls_score_list = torch.cat(cls_score_list, axis=-1)
        reg_dist_list = torch.cat(reg_dist_list, axis=-1)
        # torch.Size([1, 80, 8400]) torch.Size([1, 4, 8400]) torch.Size([8400, 2]) torch.Size([8400, 1])
        # print(cls_score_list.shape, reg_dist_list.shape, anchor_points.shape, stride_tensor.shape)

        # decode_outputs
        # [1,4,8400] x1y1x2y2
        # 为配合yolox输出是cxcywh，这里将输出转化
        # print(anchor_points.device, reg_dist_list.device)
        pred_bboxes = batch_distance2bbox_cxcywh(anchor_points,
                                          reg_dist_list.permute(0, 2, 1))
        pred_bboxes *= stride_tensor
        return torch.cat(  # 目标是(1,8400,85)
            [
                pred_bboxes,    #
                torch.ones((b, pred_bboxes.shape[1], 1), device=pred_bboxes.device, dtype=pred_bboxes.dtype),
                cls_score_list.permute(0, 2, 1)
            ],
            axis=-1)

    def _generate_anchors(self, feats=None, device='cuda:0'):
        # just use in eval time
        anchor_points = []
        stride_tensor = []
        for i, stride in enumerate(self.fpn_strides):
            if feats is not None:
                _, _, h, w = feats[i].shape
            else:
                h = int(self.eval_input_size[0] / stride)
                w = int(self.eval_input_size[1] / stride)
            shift_x = torch.arange(end=w, device=device) + self.grid_cell_offset
            shift_y = torch.arange(end=h, device=device) + self.grid_cell_offset
            shift_y, shift_x = torch.meshgrid(shift_y, shift_x)
            anchor_point = torch.stack(
                    [shift_x, shift_y], axis=-1).to(torch.float)
            # anchor_point = paddle.cast(
            #     paddle.stack(
            #         [shift_x, shift_y], axis=-1), dtype='float32')
            anchor_points.append(anchor_point.reshape([-1, 2]))
            stride_tensor.append(
                torch.full(
                    (h * w, 1), stride, dtype=torch.float, device=device))
        anchor_points = torch.cat(anchor_points)
        stride_tensor = torch.cat(stride_tensor)
        return anchor_points, stride_tensor

    def forward(self, feats, targets=None, extra_info=None):
        assert len(feats) == len(self.fpn_strides), \
            "The size of feats is not equal to size of fpn_strides"

        if self.training:
            return self.forward_train(feats, targets, extra_info)
        else:
            return self.forward_eval(feats)

    def bbox_decode(self, anchor_points, pred_dist):
        batch_size, n_anchors, _ = pred_dist.shape
        pred_dist = F.softmax(pred_dist.view(batch_size, n_anchors, 4, self.reg_max + 1), dim=-1).matmul(self.proj)
        return batch_distance2bbox(anchor_points, pred_dist)

    def get_loss(self, head_outs, targets, extra_info):
        pred_scores, pred_distri, anchors, \
        anchor_points, num_anchors_list, stride_tensor = head_outs
        # 要确定一下targets的输入方式    xyxy

        # pred_bboxes:[batch_size, n_anchors, 4]
        # pred_scores:[batch_size, n_anchors, num_classes]
        # anchors:[n_anchors, 4]
        # anchor_points:[n_anchors, 2]
        # num_anchors_list:list [169,676,2704]先是检测大物体的特征图
        # stride_tensor:[n_anchors, 1]

        anchor_points_s = anchor_points / stride_tensor
        # x1y1x2y2
        pred_bboxes = self.bbox_decode(anchor_points_s, pred_distri)
        gt_labels = targets[:, :, :1]
        # xywh
        gt_bboxes = targets[:, :, 1:]
        pad_gt_mask = (gt_bboxes.sum(-1, keepdim=True) > 0).float()

        if extra_info['epoch'] < self.static_assigner_epoch:
            assigned_labels, assigned_bboxes, assigned_scores = self.atss_assign(
                anchors,
                num_anchors_list,
                gt_labels,
                gt_bboxes,
                pad_gt_mask,
                bg_index=self.num_classes,
                pred_bboxes=pred_bboxes.detach() * stride_tensor)
            alpha_l = 0.25
        else:
            assigned_labels, assigned_bboxes, assigned_scores = \
                self.assigner(
                    pred_scores.detach(),
                    pred_bboxes.detach() * stride_tensor,
                    anchor_points,
                    num_anchors_list,
                    gt_labels,
                    gt_bboxes,
                    pad_gt_mask,
                    bg_index=self.num_classes)
            alpha_l = -1

        # rescale bbox
        assigned_bboxes /= stride_tensor
        # cls loss
        if self.use_varifocal_loss:
            one_hot_label = F.one_hot(assigned_labels, self.num_classes + 1)[..., :-1]
            loss_cls = self.varifocal_loss(pred_scores, assigned_scores,
                                            one_hot_label)
        else:
            loss_cls = self.focal_loss(pred_scores, assigned_scores, alpha_l)

        assigned_scores_sum = assigned_scores.sum()
        loss_cls /= assigned_scores_sum

        loss_l1, loss_iou, loss_dfl = self.bbox_loss(pred_distri, pred_bboxes, anchor_points_s,
                                                     assigned_labels, assigned_bboxes, assigned_scores,
                                                     assigned_scores_sum)

        loss = self.loss_weight['class'] * loss_cls + \
               self.loss_weight['iou'] * loss_iou + \
               self.loss_weight['dfl'] * loss_dfl
        out_dict = {
            'total_loss': loss,
            'loss_cls': loss_cls,
            'loss_iou': loss_iou,
            'loss_dfl': loss_dfl,
            'loss_l1': loss_l1,
        }
        return out_dict

    def generate_anchors_for_grid_cell(self, feats, fpn_strides, grid_cell_size=5.0, grid_cell_offset=0.5,
                                       device='cpu'):
        r"""
        Like ATSS, generate anchors based on grid size.
        Args:
            feats (List[Tensor]): shape[s, (b, c, h, w)]
            fpn_strides (tuple|list): shape[s], stride for each scale feature
            grid_cell_size (float): anchor size
            grid_cell_offset (float): The range is between 0 and 1.
        Returns:
            anchors (Tensor): shape[l, 4], "xmin, ymin, xmax, ymax" format.
            anchor_points (Tensor): shape[l, 2], "x, y" format.
            num_anchors_list (List[int]): shape[s], contains [s_1, s_2, ...].
            stride_tensor (Tensor): shape[l, 1], contains the stride for each scale.
        """
        assert len(feats) == len(fpn_strides)
        anchors = []
        anchor_points = []
        num_anchors_list = []
        stride_tensor = []
        for feat, stride in zip(feats, fpn_strides):
            _, _, h, w = feat.shape
            cell_half_size = grid_cell_size * stride * 0.5
            shift_x = (torch.arange(end=w, device=device) + grid_cell_offset) * stride
            shift_y = (torch.arange(end=h, device=device) + grid_cell_offset) * stride
            shift_y, shift_x = torch.meshgrid(shift_y, shift_x)
            anchor = torch.stack(
                [
                    shift_x - cell_half_size, shift_y - cell_half_size,
                    shift_x + cell_half_size, shift_y + cell_half_size
                ],
                axis=-1).clone().to(feat.dtype)
            anchor_point = torch.stack(
                [shift_x, shift_y], axis=-1).clone().to(feat.dtype)

            anchors.append(anchor.reshape([-1, 4]))
            anchor_points.append(anchor_point.reshape([-1, 2]))
            num_anchors_list.append(len(anchors[-1]))
            stride_tensor.append(
                torch.full(
                    [num_anchors_list[-1], 1], stride, dtype=feat.dtype))
        anchors = torch.cat(anchors)
        anchor_points = torch.cat(anchor_points).cuda()
        stride_tensor = torch.cat(stride_tensor).cuda()
        return anchors, anchor_points, num_anchors_list, stride_tensor
