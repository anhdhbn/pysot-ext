from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import nn, Tensor
from typing import Tuple, List
from .embedding import PositionEmbeddingSine
from pysot.core.xcorr import xcorr_slow, xcorr_fast, xcorr_depthwise
from .transformer import Transformer

class Extractor(nn.Module):
    def __init__(self, hidden_dims=256):
        super(Extractor, self).__init__()
        self.position_embedding = PositionEmbeddingSine(hidden_dims // 2)

    def forward(self, template: Tensor, search: Tensor)-> Tuple[Tensor, Tuple[Tensor, Tensor]]:     
        score = xcorr_depthwise(search, template)
        return score, self.position_embedding(score)

class MLP(nn.Module):
    def __init__(self, inputDim: int, hiddenDim: int, outputDim: int, numLayers: int):
        super().__init__()
        self.numLayers = numLayers

        h = [hiddenDim] * (numLayers - 1)
        self.layers = nn.ModuleList(nn.Linear(n, k) for n, k in zip([inputDim] + h, h + [outputDim]))

    def forward(self, x: Tensor) -> Tensor:
        for i, layer in enumerate(self.layers):
            x = nn.functional.relu(layer(x)) if i < self.numLayers - 1 else layer(x)
        return x

class SiamTr(nn.Module):
    def __init__(self, hidden_dims=512, num_head = 8, num_encoder_layer=6, 
    num_decoder_layer=6, dim_feed_forward=2048, dropout=.1, num_query=10):
        super(SiamTr, self).__init__()
        self.extractor = Extractor(hidden_dims=hidden_dims)
        self.transformer = Transformer(hidden_dims, num_head, num_encoder_layer, num_decoder_layer,
                                       dim_feed_forward, dropout)

        self.query_embed = nn.Embedding(num_query, hidden_dims)
        self.class_embed = nn.Linear(num_decoder_layer * num_decoder_layer * hidden_dims, 2)
        self.bbox_embed = nn.Linear(num_decoder_layer * num_decoder_layer * hidden_dims, 4)
        # self.bbox_embed = MLP(hidden_dims, hidden_dims, 4, 3)

    def forward(self, template: Tensor, search: Tensor):
        features, (pos, mask) = self.extractor(template, search)

        out = self.transformer(features, mask, self.query_embed.weight, pos)
        N = out.shape[0]
        out = out.view((N, -1))

        outputs_class = self.class_embed(out).sigmoid()
        outputs_coord = self.bbox_embed(out)
        # print(outputs_class.shape, outputs_coord.shape)
        # print(outputs_class[0], outputs_coord[0])
        # exit(0)
        return outputs_class, outputs_coord

        # return {'class': outputs_class[-1],
        #         'bbox': outputs_coord[-1],
        #         'aux': [{'class': oc, 'bbox': ob} for oc, ob in zip(outputs_class[:-1], outputs_coord[:-1])]}
