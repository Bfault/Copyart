import sys

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision.utils import save_image
from tqdm import tqdm

from dataset import ArtImageDataset
from discriminator import Discriminator
from generator import Generator
from utils import save_checkpoint
import config

torch.cuda.empty_cache()

def train(generator_I: Generator, generator_A: Generator, discriminator_I: Discriminator, discriminator_A: Discriminator, loader: DataLoader, gen_opt: optim.Optimizer, disc_opt: optim.Optimizer, L1: nn.L1Loss, mse: nn.MSELoss, gen_scaler: torch.cuda.amp.GradScaler, disc_scaler: torch.cuda.amp.GradScaler) -> None:
    image_real = 0
    image_fake = 0
    loop = tqdm(loader, leave=True)

    for idx, (art, image) in enumerate(loop):
        art = art.to(config.DEVICE)
        image = image.to(config.DEVICE)

        with torch.cuda.amp.autocast():
            fake_image = generator_I(art)
            D_image_real = discriminator_I(image)
            D_image_fake = discriminator_I(fake_image.detach())
            image_real += D_image_real.mean().item()
            image_fake += D_image_fake.mean().item()
            D_image_real_loss = mse(D_image_real, torch.ones_like(D_image_real))
            D_image_fake_loss = mse(D_image_fake, torch.zeros_like(D_image_fake))
            D_image_loss = D_image_real_loss + D_image_fake_loss

            fake_art = generator_A(image)
            D_art_real = discriminator_A(art)
            D_art_fake = discriminator_A(fake_art.detach())
            D_art_real_loss = mse(D_art_real, torch.ones_like(D_art_real))
            D_art_fake_loss = mse(D_art_fake, torch.zeros_like(D_art_fake))
            D_art_loss = D_art_real_loss + D_art_fake_loss

            D_loss = (D_image_loss + D_art_loss)/2
        
        disc_opt.zero_grad()
        disc_scaler.scale(D_loss).backward()
        disc_scaler.step(disc_opt)
        disc_scaler.update()

        with torch.cuda.amp.autocast():
            # Adversarial loss
            D_image_fake = discriminator_I(fake_image)
            D_art_fake = discriminator_A(fake_art)
            G_image_loss = mse(D_image_fake, torch.ones_like(D_image_fake))
            G_art_loss = mse(D_art_fake, torch.zeros_like(D_art_fake))

            # Cycle loss
            cycle_art = generator_A(fake_image)
            cycle_image = generator_I(fake_art)
            cycle_image_loss = L1(image, cycle_image)
            cycle_art_loss = L1(art, cycle_art)

            # Identity loss
            identity_image = generator_I(image)
            identity_art = generator_A(art)
            if config.LAMBDA_IDENTITY > 0:
                identity_image_loss = L1(identity_image, image)
                identity_art_loss = L1(identity_art, art)
            else:
                identity_image_loss = 0
                identity_art_loss = 0

            G_loss = (
                G_image_loss +
                G_art_loss +
                cycle_image_loss * config.LAMBDA_CYCLE +
                cycle_art_loss * config.LAMBDA_CYCLE +
                identity_image_loss * config.LAMBDA_IDENTITY +
                identity_art_loss * config.LAMBDA_IDENTITY
            )
        
        gen_opt.zero_grad()
        gen_scaler.scale(G_loss).backward()
        gen_scaler.step(gen_opt)
        gen_scaler.update()

        if idx % config.SAVE_INTERVAL == 0:
            save_image(fake_art*0.5+0.5, f'{config.OUTPUT_PATH}/{idx}_fake_art.png')
            save_image(image*0.5+0.5, f'{config.OUTPUT_PATH}/{idx}_real_image.png')
        

def main():
    generator_I = Generator(input_nc=3, output_nc=3, norm_layer=nn.InstanceNorm2d).to(config.DEVICE)
    discriminator_I = Discriminator(in_channels=3).to(config.DEVICE)

    generator_A = Generator(input_nc=3, output_nc=3, norm_layer=nn.InstanceNorm2d).to(config.DEVICE)
    discriminator_A = Discriminator(in_channels=3).to(config.DEVICE)

    discriminator_optimizer = optim.Adam(
        list(discriminator_I.parameters()) + list(discriminator_A.parameters()),
        lr=config.LEARNING_RATE,
        betas=(0.5, 0.999)
    )

    generator_optimizer = optim.Adam(
        list(generator_I.parameters()) + list(generator_A.parameters()),
        lr=config.LEARNING_RATE,
        betas=(0.5, 0.999)
    )

    L1 = nn.L1Loss()
    mse = nn.MSELoss()

    dataset = ArtImageDataset(root_art=config.ART_IMAGE_PATH, root_image=config.IMAGE_PATH, transform=config.TRANSFORM)
    loader = DataLoader(dataset, batch_size=config.BATCH_SIZE, shuffle=True, num_workers=config.NUM_WORKERS, pin_memory=True)

    generator_scaler = torch.cuda.amp.GradScaler()
    discriminator_scaler = torch.cuda.amp.GradScaler()

    for epoch in range(config.NUM_EPOCHS):
        train(
            generator_I=generator_I,
            generator_A=generator_A,
            discriminator_I=discriminator_I,
            discriminator_A=discriminator_A,
            loader=loader,
            gen_opt=generator_optimizer,
            disc_opt=discriminator_optimizer,
            L1=L1,
            mse=mse,
            gen_scaler=generator_scaler,
            disc_scaler=discriminator_scaler,
        )

        save_checkpoint(generator_I, generator_optimizer, filename=config.CHECKPOINT_GENERATOR_I.format(epoch))
        save_checkpoint(generator_A, generator_optimizer, filename=config.CHECKPOINT_GENERATOR_A.format(epoch))
        save_checkpoint(discriminator_I, discriminator_optimizer, filename=config.CHECKPOINT_DISCRIMINATOR_I.format(epoch))
        save_checkpoint(discriminator_A, discriminator_optimizer, filename=config.CHECKPOINT_DISCRIMINATOR_A.format(epoch))

if __name__ == '__main__':
    main()