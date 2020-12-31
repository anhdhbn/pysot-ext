import math
from typing import Tuple

import torch
from torch import nn, Tensor


class PositionEmbeddingSine(nn.Module):
    def __init__(self, num_position_features: int = 64, temperature: int = 10000, normalize: bool = True,
                 scale: float = None):
        super(PositionEmbeddingSine, self).__init__()

        self.num_position_features = num_position_features
        self.temperature = temperature
        self.normalize = normalize

        if scale is not None and normalize is False:
            raise ValueError("normalize should be True if scale is passed")
        if scale is None:
            scale = 2 * math.pi
        self.scale = scale

    def forward(self, x: Tensor) -> Tuple[Tensor, Tensor]:
        N, _, H, W = x.shape

        mask = torch.zeros(N, H, W, dtype=torch.bool, device=x.device)
        not_mask = ~mask

        y_embed = not_mask.cumsum(1)
        x_embed = not_mask.cumsum(2)

        if self.normalize:
            epsilon = 1e-6
            y_embed = y_embed / (y_embed[:, -1:, :] + epsilon) * self.scale
            x_embed = x_embed / (x_embed[:, :, -1:] + epsilon) * self.scale

        dimT = torch.arange(self.num_position_features, dtype=torch.float32, device=x.device)
        dimT = self.temperature ** (2 * (dimT // 2) / self.num_position_features)

        posX = x_embed.unsqueeze(-1) / dimT
        posY = y_embed.unsqueeze(-1) / dimT

        posX = torch.stack((posX[:, :, :, 0::2].sin(), posX[:, :, :, 1::2].cos()), -1).flatten(3)
        posY = torch.stack((posY[:, :, :, 0::2].sin(), posY[:, :, :, 1::2].cos()), -1).flatten(3)

        return torch.cat((posY, posX), 3).permute(0, 3, 1, 2), mask
