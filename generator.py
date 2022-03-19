from turtle import forward
import torch
import torch.nn as nn

class ConvBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, down: bool=True, use_act: bool=True, **kwargs) -> None:
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, padding_mode='reflect', **kwargs)
            if down
            else nn.ConvTranspose2d(in_channels, out_channels, **kwargs),
            nn.InstanceNorm2d(out_channels),
            nn.ReLU(inplace=True) if use_act else nn.Identity()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)

class ResidualBlock(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.block = nn.Sequential(
            ConvBlock(channels, channels, kernel_size=3, padding=1),
            ConvBlock(channels, channels, use_act=False, kernel_size=3, padding=1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.block(x)

class Generator(nn.Module):
    def __init__(self, img_channels: int, num_features: int=64, num_residuals: int=9) -> None:
        super().__init__()
        self.initial = nn.Sequential(
            nn.Conv2d(img_channels, num_features, kernel_size=7, stride=1, padding=3, padding_mode='reflect'),
            nn.ReLU(inplace=True)
        )
        self.down_blocks = nn.ModuleList(
            [
                ConvBlock(num_features, num_features*2, kernel_size=3, stride=2, padding=1),
                ConvBlock(num_features*2, num_features*4, kernel_size=3, stride=2, padding=1),
            ]
        )
        self.residual_blocks = nn.Sequential(
            *[ResidualBlock(num_features*4) for _ in range(num_residuals)]
        )
        self.up_blocks = nn.ModuleList(
            [
                ConvBlock(num_features*4, num_features*2, kernel_size=3, stride=2, padding=1, down=False, output_padding=1),
                ConvBlock(num_features*2, num_features, kernel_size=3, stride=2, padding=1, down=False, output_padding=1)
            ]
        )

        self.last = nn.Conv2d(num_features, img_channels, kernel_size=7, stride=1, padding=3, padding_mode='reflect')
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.initial(x)
        for layer in self.down_blocks:
            x = layer(x)
        x = self.residual_blocks(x)
        for layer in self.up_blocks:
            x = layer(x)
        x = self.last(x)
        x = torch.tanh(x)
        return x

if __name__ == '__main__':
    img_channels = 3
    img_size = 256
    x = torch.randn((2, img_channels, img_size, img_size))
    gen = Generator(img_channels, 9)
    print(gen(x).shape)