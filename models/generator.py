import functools

import torch
import torch.nn as nn

PADDING_TYPES_MAPPING = {
    'reflect': nn.ReflectionPad2d,
    'replicate': nn.ReplicationPad2d,
    'zero': nn.ZeroPad2d
}

class ResnetBlock(nn.Module):

    def __init__(self, dim: int, padding_type: str, norm_layer: nn.modules.batchnorm._NormBase, use_dropout: bool, use_bias: bool) -> None:
        super().__init__()

        conv_block = []
        padding = PADDING_TYPES_MAPPING.get(padding_type, 0)
        if not padding:
            raise NotImplementedError('padding_type must be in {}'.format(PADDING_TYPES_MAPPING.keys()))
        conv_block += [padding(1)]

        conv_block += [
            nn.Conv2d(in_channels=dim, out_channels=dim, kernel_size=3, bias=use_bias),
            norm_layer(dim),
            nn.ReLU(inplace=True)
        ]
        if use_dropout:
            conv_block += [nn.Dropout(p=0.5)]

        conv_block += [padding(1)]

        conv_block += [
            nn.Conv2d(in_channels=dim, out_channels=dim, kernel_size=3, bias=use_bias),
            norm_layer(dim)
        ]

        self.conv_block = nn.Sequential(*conv_block)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = x + self.conv_block(x)
        return out


class Generator(nn.Module):
    def __init__(self, input_nc: int, output_nc: int, ngf:int = 64, norm_layer: nn.modules.batchnorm._NormBase = nn.BatchNorm2d, use_dropout: bool = False, n_blocks: int = 9, padding_type: str = 'reflect') -> None:
        super().__init__()

        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        model = [
            nn.ReflectionPad2d(3),
            nn.Conv2d(in_channels=input_nc, out_channels=ngf, kernel_size=7, padding=0, bias=use_bias),
            norm_layer(ngf),
            nn.ReLU(inplace=True)
        ]

        n_downsampling = 2
        for i in range(n_downsampling):
            mult = 2 ** i
            model += [
                nn.Conv2d(in_channels=ngf * mult, out_channels=ngf * mult * 2, kernel_size=3, stride=2, padding=1, bias=use_bias),
                norm_layer(ngf * mult * 2),
                nn.ReLU(inplace=True)
            ]

        mult = 2 ** n_downsampling
        for i in range(n_blocks):
            model += [ResnetBlock(dim=ngf * mult, padding_type=padding_type, norm_layer=norm_layer, use_dropout=use_dropout, use_bias=use_bias)]

        for i in range(n_downsampling):
            mult = 2 ** (n_downsampling - i)
            model += [
                nn.ConvTranspose2d(in_channels=ngf * mult, out_channels=int(ngf * mult / 2), kernel_size=3, stride=2, padding=1, output_padding=1, bias=use_bias),
                norm_layer(int(ngf * mult / 2)),
                nn.ReLU(inplace=True)
            ]
        model += [nn.ReflectionPad2d(3)]
        model += [nn.Conv2d(in_channels=ngf, out_channels=output_nc, kernel_size=7)]
        model += [nn.Tanh()]

        self.model = nn.Sequential(*model)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return self.model(input)


if __name__ == '__main__':
    img_channels = 3
    img_size = 256
    x = torch.randn((2, img_channels, img_size, img_size))
    gen = Generator(img_channels, 9)
    print(gen(x).shape)
