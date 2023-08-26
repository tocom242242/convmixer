# ref : https://arxiv.org/pdf/2201.09792.pdf

import torch.nn as nn


class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x):
        return self.fn(x) + x


class Block(nn.Module):
    def __init__(self, dim, kernel_size):
        super().__init__()
        self.depthwise_block = Residual(
            nn.Sequential(
                nn.Conv2d(dim, dim, kernel_size, groups=dim, padding="same"),
                nn.GELU(),
                nn.BatchNorm2d(dim),
            )
        )
        self.pointwise_block = nn.Sequential(
            nn.Conv2d(dim, dim, kernel_size=1), nn.GELU(), nn.BatchNorm2d(dim)
        )

    def forward(self, x):
        x = self.depthwise_block(x)
        return self.pointwise_block(x)


class ConvMixer(nn.Module):
    def __init__(self, dim, depth, kernel_size=9, patch_size=7, n_classes=10):
        super().__init__()
        self.patch_emb = nn.Sequential(
            nn.Conv2d(3, dim, kernel_size=patch_size, stride=patch_size),
            nn.GELU(),
            nn.BatchNorm2d(dim),
        )
        self.convmixer_block = nn.Sequential(
            *[Block(dim, kernel_size) for _ in range(depth)]
        )
        self.last_block = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)), nn.Flatten(), nn.Linear(dim, n_classes)
        )

    def forward(self, x):
        x = self.patch_emb(x)
        x = self.convmixer_block(x)
        return self.last_block(x)
