#!/usr/bin/env python
# -*- encoding: utf-8 -*-

import torch.nn as nn
from .yolo_head import YOLOXHead
from .yolo_pafpn import YOLOPAFPN
from .sim_gen import FeatureTransformer, SimMapGen, PositionalEncoding
import torch.nn.functional as F
import torch

import numpy as np
import timm


class RegionalTokenModel(nn.Module):

    def __init__(self, args, feat_transformer_config, backbone=None, head=None):
        super().__init__()
        if backbone is None:
            backbone = YOLOPAFPN()
        if head is None:
            head = YOLOXHead(1)

        self.args = args

        self.backbone = backbone
        self.head = head

        self.keep_ratio = feat_transformer_config['keep_ratio']
        self.bcewithlog_loss = nn.BCEWithLogitsLoss(reduction="none")
        self.unsupervised_weight = feat_transformer_config['unsupervised_weight']

    def forward(self, feat1, feat2, mask1=None, mask2=None, targets=None, id_=None, input_size=(640, 640), global_loss=True, vis=False):
        if len(feat1.size()) == 3:
            global_pred = torch.from_numpy(np.zeros([4,1])).cuda()
            global_loss = False

            feat1_ = torch.nn.functional.normalize(feat1, dim=2, p=2)
            feat2_ = torch.nn.functional.normalize(feat2, dim=2, p=2)
            sim_matrix = torch.bmm(feat1_, torch.transpose(feat2_, 1, 2))
        elif len(feat1.size()) == 4:
            global_pred = torch.from_numpy(np.zeros([4, 1])).cuda()
            global_loss = False

            B, L, n_crops, C = feat1.size()
            feat1_ = feat1.view(B, L * n_crops, C)
            feat2_ = feat2.view(B, L * n_crops, C)

            feat1_ = torch.nn.functional.normalize(feat1_, dim=2, p=2)
            feat2_ = torch.nn.functional.normalize(feat2_, dim=2, p=2)

            sim_matrix = torch.bmm(feat1_, torch.transpose(feat2_, 1, 2))
            sim_matrix = F.max_pool2d(sim_matrix, kernel_size=n_crops, stride=n_crops)

        x, global_label = [], []
        for idx, m in enumerate(sim_matrix):
            m = m[:mask1[idx].sum().int(), :mask2[idx].sum().int()][..., None]
            if self.training:
                if targets[idx].max() < 1e-6:
                    global_label.append(torch.tensor([0.], device=m.device))
                else:
                    global_label.append(torch.tensor([1.], device=m.device))
            m = m.repeat(1, 1, 1, 3).permute(0, 3, 1, 2)
            m = F.interpolate(m, size=input_size, mode='bicubic')
            x.append(m)
        x = torch.cat(x, dim=0)

        if global_loss and self.training:
            weight = id_.clone()
            weight[id_ < self.sum_index] = 1.
            weight[id_ >= self.sum_index] = self.unsupervised_weight
            weight = weight.float()
            global_label = torch.cat(global_label, dim=0).view(-1, 1)
            # global_pred = torch.cat(global_pred, dim=0).view(-1, 1)
            global_loss = torch.squeeze(self.bcewithlog_loss(global_pred, global_label))
            global_loss = torch.matmul(global_loss, weight)
        else:
            global_loss = 0

        fpn_outs = self.backbone(x)

        if self.training:
            assert targets is not None
            if sum(id_ < self.sum_index) > 0:
                fpn_outs_supervised = tuple([fpn_out[id_ < self.sum_index, ...] for fpn_out in fpn_outs])
                targets_supervised = targets[id_ < self.sum_index, ...]
                x_supervised = x[id_ < self.sum_index, ...]
                (
                    loss_supervised,
                    iou_loss_supervised,
                    conf_loss_supervised,
                    cls_loss_supervised,
                    l1_loss_supervised,
                    num_fg_supervised,
                ) = self.head(
                    fpn_outs_supervised,
                    targets_supervised,
                    x_supervised,
                )
            else:
                (
                    loss_supervised,
                    iou_loss_supervised,
                    conf_loss_supervised,
                    cls_loss_supervised,
                    l1_loss_supervised,
                    num_fg_supervised,
                ) = (0, 0, 0, 0, 0, 0)
            if sum(id_ >= self.sum_index) > 0:
                fpn_outs_unsupervised = tuple([fpn_out[id_ >= self.sum_index, ...] for fpn_out in fpn_outs])
                targets_unsupervised = targets[id_ >= self.sum_index, ...]
                x_unsupervised = x[id_ >= self.sum_index, ...]
                (
                    loss_unsupervised,
                    iou_loss_unsupervised,
                    conf_loss_unsupervised,
                    cls_loss_unsupervised,
                    l1_loss_unsupervised,
                    num_fg_unsupervised,
                ) = self.head(
                    fpn_outs_unsupervised,
                    targets_unsupervised,
                    x_unsupervised,
                )
            else:
                (
                    loss_unsupervised,
                    iou_loss_unsupervised,
                    conf_loss_unsupervised,
                    cls_loss_unsupervised,
                    l1_loss_unsupervised,
                    num_fg_unsupervised,
                ) = (0, 0, 0, 0, 0, 0)
            outputs = {
                "total_loss": loss_supervised + self.unsupervised_weight * loss_unsupervised + 0.01 * global_loss,
                "iou_loss": loss_supervised + self.unsupervised_weight * loss_unsupervised,
                "l1_loss": l1_loss_supervised + self.unsupervised_weight * l1_loss_unsupervised,
                "conf_loss": conf_loss_supervised + self.unsupervised_weight * conf_loss_unsupervised,
                "cls_loss": cls_loss_supervised + self.unsupervised_weight * cls_loss_unsupervised,
                "global_loss": global_loss,
                "num_fg": num_fg_supervised + self.unsupervised_weight * num_fg_unsupervised,
            }
        else:
            outputs = self.head(fpn_outs)
            outputs = [global_pred, outputs]

        return outputs
