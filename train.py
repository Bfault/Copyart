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
# from utils import save_checkpoint, load_checkpoint
import config

def train(gen_dom1: Generator, gen_dom2: Generator, disc_dom1: Discriminator, disc_dom2: Discriminator, loader: DataLoader, gen_opt: optim.Optimizer, disc_opt: optim.Optimizer, L1: nn.L1Loss, mse: nn.MSELoss, gen_scaler: torch.cuda.amp.GradScaler, disc_scaler: torch.cuda.amp.GradScaler) -> None:
    loop = tqdm(loader, leave=True)

    for idx, (art, image) in enumerate(loop):
        art = art.to(config.DEVICE)
        image = image.to(config.DEVICE)

        with torch.cuda.amp.autocast():
            fake_image = gen_dom1(art)
            D_image_real = disc_dom1(image)
            D_image_fake = disc_dom1(fake_image.detach())
            D_image_real_loss = mse(D_image_real, torch.ones_like(D_image_real))
            D_image_fake_loss = mse(D_image_fake, torch.zeros_like(D_image_fake))
            D_image_loss = D_image_real_loss + D_image_fake_loss

            fake_art = gen_dom2(art)
            D_art_real = disc_dom2(art)
            D_art_fake = disc_dom2(fake_art.detach())
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
            D_image_fake = disc_dom1(fake_image)
            D_art_fake = disc_dom2(fake_art)
            G_image_loss = mse(D_image_fake, torch.ones_like(D_image_fake))
            G_art_loss = mse(D_art_fake, torch.ones_like(D_art_fake))

            # Cycle loss
            cycle_image = gen_dom2(fake_art)
            cycle_art = gen_dom1(fake_image)
            cycle_image_loss = L1(cycle_image, image)
            cycle_art_loss = L1(cycle_art, art)

            # Identity loss
            identity_image = gen_dom1(image)
            identity_art = gen_dom2(art)
            identity_image_loss = L1(identity_image, image)
            identity_art_loss = L1(identity_art, art)

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
            save_image(fake_image*0.5+0.5, f'{config.OUTPUT_PATH}/fake_image_{idx}.png')
            save_image(fake_art*0.5+0.5, f'{config.OUTPUT_PATH}/fake_art_{idx}.png')


def main():
    generator_I = Generator(img_channels=3, num_residuals=9).to(config.DEVICE)
    discriminator_I = Discriminator(in_channels=3).to(config.DEVICE)

    generator_A = Generator(img_channels=3, num_residuals=9).to(config.DEVICE)
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
            gen_dom1=generator_I,
            gen_dom2=generator_A,
            disc_dom1=discriminator_I,
            disc_dom2=discriminator_A,
            loader=loader,
            gen_opt=generator_optimizer,
            disc_opt=discriminator_optimizer,
            L1=L1,
            mse=mse,
            gen_scaler=generator_scaler,
            disc_scaler=discriminator_scaler,
        )

        # if config.SAVE_MODEL:
        #     save_checkpoint(generator_I, generator_optimizer, filename=config.CHECKPOINT_GENERATOR_I)
        #     save_checkpoint(generator_A, generator_optimizer, filename=config.CHECKPOINT_GENERATOR_A)
        #     save_checkpoint(discriminator_I, discriminator_optimizer, filename=config.CHECKPOINT_DISCRIMINATOR_I)
        #     save_checkpoint(discriminator_A, discriminator_optimizer, filename=config.CHECKPOINT_DISCRIMINATOR_A)

if __name__ == '__main__':
    main()