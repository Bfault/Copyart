#!/usr/bin/env python3

import sys
import traceback

import torch
import numpy as np
from PIL import Image
import torch.nn as nn
import torchvision.transforms as transforms
import matplotlib.pyplot as plt

from models.generator import Generator
import models.config as config

def transform_image(img_path: str, artist: str) -> torch.Tensor:
    gen_path = f"checkpoints/style_{artist.lower()}_pretrained/latest_net_G.pth"
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

def main(img_path, gen_path):
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
    plt.imshow(output)

    plt.show()

if __name__ == '__main__':
    args = sys.argv[1:]
    try:
        main(*args)
    except:
        traceback.print_exc()
