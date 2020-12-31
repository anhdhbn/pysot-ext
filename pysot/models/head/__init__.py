# Copyright (c) SenseTime. All Rights Reserved.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

from pysot.models.head.mask import MaskCorr, Refine
from pysot.models.head.rpn import UPChannelRPN, DepthwiseRPN, MultiRPN
from pysot.models.head.transformer import SiamTr

RPNS = {
        'UPChannelRPN': UPChannelRPN,
        'DepthwiseRPN': DepthwiseRPN,
        'MultiRPN': MultiRPN
       }

MASKS = {
         'MaskCorr': MaskCorr,
        }

REFINE = {
          'Refine': Refine,
         }

TRANSFORMER = {
    'v1': SiamTr
}

def get_rpn_head(name, **kwargs):
    return RPNS[name](**kwargs)


def get_mask_head(name, **kwargs):
    return MASKS[name](**kwargs)


def get_refine_head(name):
    return REFINE[name]()

def get_tr_head(name, **kwargs):
    return TRANSFORMER[name](**kwargs)
