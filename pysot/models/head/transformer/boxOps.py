from typing import Tuple

import torch
from torch import Tensor



def boxIoU_batch(bboxes1, bboxes2):
    x11, y11, x12, y12 = torch.chunk(bboxes1, 4, dim=1)
    x21, y21, x22, y22 = torch.chunk(bboxes2, 4, dim=1)
    # determine the (x, y)-coordinates of the intersection rectangle
    xA = torch.max(x11, x21.transpose(0, 1))
    yA = torch.max(y11, y21.transpose(0, 1))
    xB = torch.min(x12, x22.transpose(0, 1))
    yB = torch.min(y12, y22.transpose(0, 1))

    # compute the area of intersection rectangle
    x_ = (xB - xA + 1)
    y_ = (yB - yA + 1)
    inter_area = torch.max(x_, torch.zeros_like(x_)) * torch.max(y_, torch.zeros_like(y_))

    # compute the area of both the prediction and ground-truth rectangles
    boxAArea = (x12 - x11 + 1) * (y12 - y11 + 1)
    boxBArea = (x22 - x21 + 1) * (y22 - y21 + 1)

    iou = inter_area / (boxAArea + boxBArea.transpose(0, 1) - inter_area)
    iou = torch.diagonal(iou)
    return iou

@torch.no_grad()
def boxIoU(boxes1: Tensor, boxes2: Tensor) -> Tuple[Tensor, Tensor]:
    boxes1Area = (boxes1[..., 2] - boxes1[..., 0]) * (boxes1[..., 3] - boxes1[..., 1])
    boxes2Area = (boxes2[..., 2] - boxes2[..., 0]) * (boxes2[..., 3] - boxes2[..., 1])

    lt = torch.max(boxes1[:, None, :2], boxes2[:, :2])
    rb = torch.min(boxes1[:, None, 2:], boxes2[:, 2:])

    wh = (rb - lt).clamp(min=0)
    intersectArea = wh[:, :, 0] * wh[:, :, 1]

    unionArea = boxes1Area[:, None] + boxes2Area - intersectArea

    iou = intersectArea / unionArea
    return iou, unionArea

@torch.no_grad()
def gIoU(boxes1: Tensor, boxes2: Tensor) -> Tensor:
    iou, unionArea = boxIoU(boxes1, boxes2)
    lt = torch.min(boxes1[:, None, :2], boxes2[:, :2])
    rb = torch.max(boxes1[:, None, 2:], boxes2[:, 2:])
    wh = (rb - lt).clamp(min=0)

    enclosingArea = wh[:, :, 0] * wh[:, :, 1]

    return iou - (enclosingArea - unionArea) / enclosingArea

@torch.no_grad()
def boxCxcywh2Xyxy(box: Tensor) -> Tensor:
    cx, cy, w, h = box.unbind(-1)

    x1 = cx - w / 2
    y1 = cy - h / 2
    x2 = cx + w / 2
    y2 = cy + h / 2

    return torch.stack([x1, y1, x2, y2], -1)