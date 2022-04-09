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
    def __init__(self):
        self.is_initialized = False
    
    def initialize(self, parser):
        parser.add_argument('--image', '-i', type=str, required=True, help='input image path')
        parser.add_argument('--artist', '-a', type=str, nargs='+', required=True, help='artist name [{}]'.format(', '.join(config.ARTISTS)))
        parser.add_argument('--output', '-o', type=str, help='output image path')
        self.is_initialized = True
        return parser

    def parse(self):
        parser = argparse.ArgumentParser()
        self.initialize(parser)

        return parser.parse_args()

def transform_image(img_path: str, artist: str) -> torch.Tensor:
    gen_path = f"checkpoints/style_{artist.lower().replace(' ', '')}_pretrained/latest_net_G.pth"
    generator = Generator(input_nc=3, output_nc=3, norm_layer=nn.InstanceNorm2d)
    model_dict = generator.state_dict()
    state_dict = torch.load(gen_path, map_location=torch.device('cpu'))
    
    pretrained_dict = {k: v for k, v in state_dict.items() if k in model_dict}
    model_dict.update(pretrained_dict) 
    generator.load_state_dict(pretrained_dict)

    img = Image.open(img_path).convert('RGB')
    img = np.asarray(img)
    width, height = img.shape[1], img.shape[0]
    feature = config.PROD_TRANSFORM(image=img)['image']
    output = generator(feature)

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
