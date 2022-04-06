from typing import List

import torch
import torch.nn as nn

class Block(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, stride: int) -> None:
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=4,
                stride=stride,
                padding=1,
                bias=True,
                padding_mode='reflect'
            ),
            nn.InstanceNorm2d(out_channels),
            nn.LeakyReLU(0.2)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)

class Discriminator(nn.Module):
    def __init__(self, in_channels: int=3, features: List[int]=[64, 128, 256, 512]) -> None:
        super().__init__()
        self.initial = nn.Sequential(
            nn.Conv2d(
                in_channels,
                features[0],
                kernel_size=4,
                stride=2,
                padding=1,
                padding_mode='reflect'
            ),
            nn.LeakyReLU(0.2)
        )

        layers = []
        in_channels = features[0]
        for feature in features[1:]:
            layers.append(Block(in_channels, feature, stride=1 if feature == features[-1] else 2))
            in_channels = feature
        layers.append(nn.Conv2d(in_channels, 1, kernel_size=4, stride=1, padding=1, padding_mode='reflect'))
        self.model = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.initial(x)
        x = self.model(x)
        x = torch.sigmoid(x)
        return x

if __name__ == '__main__':
    x = torch.randn((10, 3, 256, 256))
    model = Discriminator(3)
    preds = model(x)
    print(preds.shape)