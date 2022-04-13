#!/usr/bin/env python3

import traceback
import argparse

import torch
import numpy as np
from PIL import Image
import torch.nn as nn
import torchvision.transforms as transforms
import matplotlib.pyplot as plt

from models.generator import Generator
import models.config as config

class Options:
    def __init__(self) -> None:
        self.is_initialized = False
    
    def initialize(self, parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
        parser.add_argument('--image', '-i', type=str, required=True, help='input image path')
        parser.add_argument('--artist', '-a', type=str, nargs='+', required=True, help='artist name [{}]'.format(', '.join(config.ARTISTS)))
        parser.add_argument('--output', '-o', type=str, help='output image path')
        self.is_initialized = True
        return parser

    def parse(self) -> argparse.Namespace:
        parser = argparse.ArgumentParser()
        self.initialize(parser)

        return parser.parse_args()

def transform_image(img_path: str, artist: str) -> torch.Tensor:
    gen = Generator(input_nc=3, output_nc=3, norm_layer=nn.InstanceNorm2d)
    gen_path = "checkpoints/style_{}_pretrained/latest_net_G.pth".format(artist.lower().replace(' ', ''))

    gen_dict = gen.state_dict()
    state_dict = torch.load(gen_path, map_location=config.DEVICE)
    
    pretrained_dict = {k: v for k, v in state_dict.items() if k in gen_dict}
    gen_dict.update(pretrained_dict) 
    gen.load_state_dict(pretrained_dict)

    img = np.array(Image.open(img_path).convert('RGB'))
    height, width = img.shape[:2]
    feature = config.PROD_TRANSFORM(image=img)['image']
    output = gen(feature)

    output = output * 0.5 + 0.5
    output = transforms.Resize((height, width))(output)
    
    output = output.detach().numpy().transpose(1, 2, 0)

    return output

def main():
    opt = Options().parse()
    output = transform_image(opt.image, ' '.join(opt.artist))
    plt.imshow(output)
    if (opt.output):
        plt.imsave(opt.output, output)

    plt.show()

if __name__ == '__main__':
    try:
        main()
    except SystemExit:
        pass
    except:
        traceback.print_exc()
