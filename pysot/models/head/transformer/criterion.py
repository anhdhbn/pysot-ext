from typing import Dict, List, Tuple

import torch
from torch import nn, Tensor
from pysot.models.head.transformer.boxOps import boxCxcywh2Xyxy, gIoU, boxIoU, boxIoU_batch
import torch.nn.functional as F

class SiamTrCriterion(nn.Module):
    def __init__(self, cls_weight = 1, loc_weight = 1.2, giou_weight = 1.4):
        super(SiamTrCriterion, self).__init__()
        self.loc_weight = loc_weight
        self.cls_weight = cls_weight
        self.giou_weight = giou_weight

        self.tr_cls_loss = nn.BCELoss()

    def forward(self, x: Tuple[Tensor, Tensor], y: Tuple[str, Tensor]) -> Dict[str, Tensor]:
        """
        :param x and y: a tuple containing:
            a tensor of shape [batchSize , 1]
            a tensor of shape [batchSize , 4]

        :return: a dictionary containing classification loss, bbox loss, and gIoU loss
        """
        outputs = {}
        cls, loc = x
        label_cls, label_loc = y
        N, _ = label_loc.shape

        # class loss
        cls_loss = self.tr_cls_loss(cls, label_cls)
        
        # ignore negative labels
        mask = label_cls != torch.tensor([0], dtype=torch.float).cuda()
        mask = torch.cat((mask, mask, mask, mask), 1)
        loc_mask = loc[mask].view(-1, 4)
        label_loc_mask = label_loc[mask].view(-1, 4)
        # loc loss
        loc_loss = F.mse_loss(loc_mask, label_loc_mask)

        # giou loss
        # giou_loss = 1 - torch.diag(gIoU(loc, label_loc))
        # giou_loss = giou_loss.sum() / (N + 1e-6)

        # iou loss
        iou = boxIoU_batch(loc_mask, label_loc_mask)
        iou_loss = torch.mean(1 - iou)

        outputs['loc_loss'] = loc_loss
        outputs['cls_loss'] = cls_loss
        # outputs['giou_loss'] = giou_loss
        outputs['iou_loss'] = iou_loss
        outputs['iou_var'] = torch.std(iou)
        # outputs['total_loss'] = self.cls_weight * cls_loss + self.loc_weight * loc_loss + self.giou_weight * giou_loss
        outputs['total_loss'] = self.cls_weight * cls_loss + self.loc_weight * loc_loss + self.giou_weight * iou_loss
        return outputs