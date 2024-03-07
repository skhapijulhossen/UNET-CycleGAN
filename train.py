"""
Training for CycleGAN

Programmed by Aladdin Persson <aladdin.persson at hotmail dot com>
* 2020-11-05: Initial coding
* 2022-12-21: Small revision of code, checked that it works with latest PyTorch version
"""

import torch
from dataset import HorseZebraDataset
import sys
from utils import save_checkpoint, load_checkpoint
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim
import config
from tqdm import tqdm
from torchvision.utils import save_image
from discriminator import Discriminator
from generator import Generator


def train_fn(
    discriminatorX, discriminatorY, generatorX, generatorY, loader, opt_disc, opt_gen, l1, mse, d_scaler, g_scaler
):
    H_reals = 0
    H_fakes = 0
    loop = tqdm(loader, leave=True)

    for idx, (zebra, horse) in enumerate(loop):
        zebra = zebra.to(config.DEVICE)
        horse = horse.to(config.DEVICE)

        # Train Discriminators H and Z
        with torch.cuda.amp.autocast():
            fake_horse = generatorY(zebra)
            D_H_real = discriminatorX(horse)
            D_H_fake = discriminatorX(fake_horse.detach())
            H_reals += D_H_real.mean().item()
            H_fakes += D_H_fake.mean().item()
            D_H_real_loss = mse(D_H_real, torch.ones_like(D_H_real))
            D_H_fake_loss = mse(D_H_fake, torch.zeros_like(D_H_fake))
            D_H_loss = D_H_real_loss + D_H_fake_loss

            fake_zebra = generatorX(horse)
            D_Z_real = discriminatorY(zebra)
            D_Z_fake = discriminatorY(fake_zebra.detach())
            D_Z_real_loss = mse(D_Z_real, torch.ones_like(D_Z_real))
            D_Z_fake_loss = mse(D_Z_fake, torch.zeros_like(D_Z_fake))
            D_Z_loss = D_Z_real_loss + D_Z_fake_loss

            # put it togethor
            D_loss = (D_H_loss + D_Z_loss) / 2

        opt_disc.zero_grad()
        d_scaler.scale(D_loss).backward()
        d_scaler.step(opt_disc)
        d_scaler.update()

        # Train Generators H and Z
        with torch.cuda.amp.autocast():
            # adversarial loss for both generators
            D_H_fake = discriminatorX(fake_horse)
            D_Z_fake = discriminatorY(fake_zebra)
            loss_G_H = mse(D_H_fake, torch.ones_like(D_H_fake))
            loss_G_Z = mse(D_Z_fake, torch.ones_like(D_Z_fake))

            # cycle loss
            cycle_zebra = generatorX(fake_horse)
            cycle_horse = generatorY(fake_zebra)
            cycle_zebra_loss = l1(zebra, cycle_zebra)
            cycle_horse_loss = l1(horse, cycle_horse)

            # identity loss (remove these for efficiency if you set lambda_identity=0)
            identity_zebra = generatorX(zebra)
            identity_horse = generatorY(horse)
            identity_zebra_loss = l1(zebra, identity_zebra)
            identity_horse_loss = l1(horse, identity_horse)

            # add all togethor
            G_loss = (
                loss_G_Z
                + loss_G_H
                + cycle_zebra_loss * config.LAMBDA_CYCLE
                + cycle_horse_loss * config.LAMBDA_CYCLE
                + identity_horse_loss * config.LAMBDA_IDENTITY
                + identity_zebra_loss * config.LAMBDA_IDENTITY
            )

        opt_gen.zero_grad()
        g_scaler.scale(G_loss).backward()
        g_scaler.step(opt_gen)
        g_scaler.update()

        if idx % 200 == 0:
            save_image(fake_horse * 0.5 + 0.5, f"work_examples/horse_{idx}.png")
            save_image(fake_zebra * 0.5 + 0.5, f"work_examples/zebra_{idx}.png")

        loop.set_postfix(H_real=H_reals / (idx + 1),
                         H_fake=H_fakes / (idx + 1))


def main():
    discriminatorX = Discriminator(in_channels=3).to(config.DEVICE)
    discriminatorY = Discriminator(in_channels=3).to(config.DEVICE)
    generatorX = Generator(img_channels=3, num_residuals=9).to(config.DEVICE)
    generatorY = Generator(img_channels=3, num_residuals=9).to(config.DEVICE)
    opt_disc = optim.Adam(
        list(discriminatorX.parameters()) + list(discriminatorY.parameters()),
        lr=config.LEARNING_RATE,
        betas=(0.5, 0.999),
    )

    opt_gen = optim.Adam(
        list(generatorX.parameters()) + list(generatorY.parameters()),
        lr=config.LEARNING_RATE,
        betas=(0.5, 0.999),
    )

    L1 = nn.L1Loss()
    mse = nn.MSELoss()

    if config.LOAD_MODEL:
        load_checkpoint(
            config.CHECKPOINT_generatorY,
            generatorY,
            opt_gen,
            config.LEARNING_RATE,
        )
        load_checkpoint(
            config.CHECKPOINT_generatorX,
            generatorX,
            opt_gen,
            config.LEARNING_RATE,
        )
        load_checkpoint(
            config.CHECKPOINT_CRITIC_H,
            discriminatorX,
            opt_disc,
            config.LEARNING_RATE,
        )
        load_checkpoint(
            config.CHECKPOINT_CRITIC_Z,
            discriminatorY,
            opt_disc,
            config.LEARNING_RATE,
        )

    dataset = HorseZebraDataset(
        root_horse=config.TRAIN_DIR + "/horses",
        root_zebra=config.TRAIN_DIR + "/zebras",
        transform=config.transforms,
    )
    val_dataset = HorseZebraDataset(
        root_horse=config.VAL_DIR + "/horses",
        root_zebra=config.VAL_DIR + "/zebras",
        transform=config.transforms,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=1,
        shuffle=False,
        pin_memory=True,
    )
    loader = DataLoader(
        dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=True,
        num_workers=config.NUM_WORKERS,
        pin_memory=True,
    )
    g_scaler = torch.cuda.amp.GradScaler(enabled=False)
    d_scaler = torch.cuda.amp.GradScaler(enabled=False)

    for epoch in range(config.NUM_EPOCHS):
        train_fn(
            discriminatorX,
            discriminatorY,
            generatorX,
            generatorY,
            loader,
            opt_disc,
            opt_gen,
            L1,
            mse,
            d_scaler,
            g_scaler,
        )

        if config.SAVE_MODEL:
            save_checkpoint(generatorY, opt_gen, filename=config.CHECKPOINT_generatorY)
            save_checkpoint(generatorX, opt_gen, filename=config.CHECKPOINT_generatorX)
            save_checkpoint(discriminatorX, opt_disc,
                            filename=config.CHECKPOINT_CRITIC_H)
            save_checkpoint(discriminatorY, opt_disc,
                            filename=config.CHECKPOINT_CRITIC_Z)


if __name__ == "__main__":
    main()
