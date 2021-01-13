from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import numpy as np
import torch.nn.functional as F

from pysot.core.config import cfg
from pysot.utils.anchor import Anchors
from pysot.tracker.base_tracker import SiameseTracker
import cv2
import torch

class SiamTrTracker(SiameseTracker):
    def __init__(self, model):
        super(SiamTrTracker, self).__init__()
        self.model = model
        self.model.eval()

    def transform(self, img):
        img = img.transpose(2, 0, 1)
        img = img[np.newaxis, :, :, :]
        img = img.astype(np.float32)
        img = torch.from_numpy(img)
        return img

    def init_(self, img):
        """
        args:
            img(np.ndarray): BGR image
        """
        # get crop
        z_crop = self.transform(img)
        self.model.template(z_crop)
    
    def init(self, img, roi):
        (x, y, w, h) = roi
        z_crop = img[int(y):int(y+h), int(x):int(x+w)]
        self.init_(z_crop)

    def track(self, img):
        shape = img.shape[:2]
        x_crop = self.transform(img)
        output = self.model.track(x_crop)

        cls = self._convert_score(output['cls'])[0]
        bbox = self._convert_bbox(output['loc'][0], shape)
        return (cls, bbox)

    def _convert_score(self, score):
        return F.softmax(score, dim=1).data[:, 0].cpu().numpy()

    def _convert_bbox(self, delta, shape):
        delta = delta.data.cpu().numpy()
        w, h = shape
        x1, y1, x2, y2 = delta
        x1 = x1 * w
        y1 = y1 * h
        x2 = x2 * w
        y2 = y2 * h
        return int(x1), int(y1), int(x2), int(y2)