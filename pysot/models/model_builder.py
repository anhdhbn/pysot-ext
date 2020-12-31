# Copyright (c) SenseTime. All Rights Reserved.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import torch.nn as nn
import torch.nn.functional as F
import torch
from pysot.core.config import cfg
from pysot.models.loss import select_cross_entropy_loss, weight_l1_loss
from pysot.models.backbone import get_backbone
from pysot.models.head import get_rpn_head, get_mask_head, get_refine_head, get_tr_head
from pysot.models.neck import get_neck

class ModelBuilder(nn.Module):
    def __init__(self):
        super(ModelBuilder, self).__init__()

        # build backbone
        self.backbone = get_backbone(cfg.BACKBONE.TYPE,
                                     **cfg.BACKBONE.KWARGS)

        # build adjust layer
        if cfg.ADJUST.ADJUST:
            self.neck = get_neck(cfg.ADJUST.TYPE,
                                 **cfg.ADJUST.KWARGS)

        # build rpn head
        if cfg.TRANSFORMER.TRANSFORMER:
            assert(cfg.TRANSFORMER.KWARGS.hidden_dims == cfg.ADJUST.KWARGS.out_channels, "AdjustLayer out_channels = hidden_dims")
            self.tr_head = get_tr_head(cfg.TRANSFORMER.TYPE,
                                     **cfg.TRANSFORMER.KWARGS)
        else:
            self.rpn_head = get_rpn_head(cfg.RPN.TYPE,
                                     **cfg.RPN.KWARGS)
            # build mask head
            if cfg.MASK.MASK:
                self.mask_head = get_mask_head(cfg.MASK.TYPE,
                                            **cfg.MASK.KWARGS)

                if cfg.REFINE.REFINE:
                    self.refine_head = get_refine_head(cfg.REFINE.TYPE)

    def template(self, z):
        if cfg.TRANSFORMER.TRANSFORMER:
            pass
        else:
            zf = self.backbone(z)
            if cfg.MASK.MASK:
                zf = zf[-1]
            if cfg.ADJUST.ADJUST:
                zf = self.neck(zf)
            self.zf = zf

    def track(self, x):
        if cfg.TRANSFORMER.TRANSFORMER:
            xf = self.backbone(x)
            if cfg.ADJUST.ADJUST:
                xf = self.neck(xf)
            # pass
            cls, loc = self.tr_head(self.zf, xf)
            return {
                    'cls': cls,
                    'loc': loc
                }
        else:
            xf = self.backbone(x)
            if cfg.MASK.MASK:
                self.xf = xf[:-1]
                xf = xf[-1]
            if cfg.ADJUST.ADJUST:
                xf = self.neck(xf)
            cls, loc = self.rpn_head(self.zf, xf)
            if cfg.MASK.MASK:
                mask, self.mask_corr_feature = self.mask_head(self.zf, xf)
            return {
                    'cls': cls,
                    'loc': loc,
                    'mask': mask if cfg.MASK.MASK else None
                }

    def mask_refine(self, pos):
        if cfg.TRANSFORMER.TRANSFORMER:
            pass
        else:
            return self.refine_head(self.xf, self.mask_corr_feature, pos)

    def log_softmax(self, cls):
        if cfg.TRANSFORMER.TRANSFORMER:
            cls = F.log_softmax(cls, dim=-1)
            return cls
        else:
            b, a2, h, w = cls.size()
            cls = cls.view(b, 2, a2//2, h, w)
            cls = cls.permute(0, 2, 3, 4, 1).contiguous()
            cls = F.log_softmax(cls, dim=4)
            return cls

    def forward(self, data):
        """ only used in training
        """
        template = data['template'].cuda() # 28, 2, 127, 127: batch 28, color 3, size 127x127
        search = data['search'].cuda()
        label_cls = data['label_cls'].cuda() # torch.Size([28, 5, 25, 25])
        label_loc = data['label_loc'].cuda()
        
        # get feature
        zf = self.backbone(template)
        xf = self.backbone(search)
        
        if cfg.TRANSFORMER.TRANSFORMER:
            if cfg.ADJUST.ADJUST:
                zf = self.neck(zf)
                xf = self.neck(xf)
            cls, loc = self.tr_head(zf, xf)
            _, query, _ = loc.shape
            # print("loc", loc.shape)
            # print("loc_t", label_loc.shape)
            loc_loss = F.mse_loss(loc, label_loc.unsqueeze(1).repeat(1, query, 1))
            outputs = {}
            outputs['total_loss'] = cfg.TRAIN.LOC_WEIGHT * loc_loss
            # print(cls.view(-1).shape, label_cls.view(-1).shape)
            # loss = F.nll_loss(cls.view(-1), label_cls.view(-1))
            # print(loss.shape, loss)
            return outputs
        else:
            label_loc_weight = data['label_loc_weight'].cuda()
            if cfg.MASK.MASK:
                zf = zf[-1]
                self.xf_refine = xf[:-1]
                xf = xf[-1]
            if cfg.ADJUST.ADJUST:
                zf = self.neck(zf)
                xf = self.neck(xf)
            cls, loc = self.rpn_head(zf, xf)
            # loc torch.Size([28, 20, 25, 25])
            # label_loc torch.Size([28, 4, 5, 25, 25])
            # label_loc_weight torch.Size([28, 5, 25, 25])
            # get loss
            cls = self.log_softmax(cls) # torch.Size([28, 5, 25, 25, 2])
            cls_loss = select_cross_entropy_loss(cls, label_cls) # cls_loss torch.Size([])
            loc_loss = weight_l1_loss(loc, label_loc, label_loc_weight) 
            outputs = {}
            outputs['total_loss'] = cfg.TRAIN.CLS_WEIGHT * cls_loss + \
                cfg.TRAIN.LOC_WEIGHT * loc_loss
            outputs['cls_loss'] = cls_loss
            outputs['loc_loss'] = loc_loss

            if cfg.MASK.MASK:
                # TODO
                mask, self.mask_corr_feature = self.mask_head(zf, xf)
                mask_loss = None
                outputs['total_loss'] += cfg.TRAIN.MASK_WEIGHT * mask_loss
                outputs['mask_loss'] = mask_loss
            return outputs
