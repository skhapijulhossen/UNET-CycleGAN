# -*- coding: utf-8 -*-
"""RS-UNET-ViT.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/151i_OIBP69AxtPVRYGAZHOivtMk4fId8

#installs
"""

# ! pip install -r requirements.txt

# ! pip install fastapi kaleido python-multipart uvicorn numpy==1.24.1

# from google.colab import drive
# drive.mount('/content/drive')

import pandas as pd

"""#imports"""

import torch
import albumentations as alb
from albumentations.pytorch import ToTensorV2
import torch.nn as nn

from PIL import Image
import os
from torch.utils.data import Dataset
import numpy as np

import multiprocessing

"""#Configuration"""

cores = multiprocessing.cpu_count() # Count the number of cores in a computer
cores

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
TRAIN_DIR = "/content/drive/MyDrive/Colab Notebooks/Steganography/data/train"
VAL_DIR = "/content/drive/MyDrive/Colab Notebooks/Steganography/data/test"
BATCH_SIZE = 8
LEARNING_RATE = 1e-5
LAMBDA_IDENTITY = 0.0
LAMBDA_CYCLE = 10
NUM_WORKERS = 2
NUM_EPOCHS = 10
LOAD_MODEL = False
SAVE_MODEL = True
CHECKPOINT_GEN_X= "genX.pth.tar"
CHECKPOINT_GEN_Y = "genY.pth.tar"
CHECKPOINT_DISC_X = "discX.pth.tar"
CHECKPOINT_DISC_Y = "discY.pth.tar"
SAVE_PATH = '/content/drive/MyDrive/Colab Notebooks/Steganography/models'

### Transformations before feeding
transforms = alb.Compose(
    [
        alb.Resize(width=256, height=256),
        alb.HorizontalFlip(p=0.5),
        alb.Normalize(mean=[0.5, 0.5, 0.5], std=[
                      0.5, 0.5, 0.5], max_pixel_value=255),
        ToTensorV2(),
    ],
    additional_targets={"image0": "image"},
)







"""# Dataset Pre-process"""



class HorseZebraDataset(Dataset):
    def __init__(self, root_y, root_x, transform=None):
        self.root_y = root_y
        self.root_x = root_x
        self.transform = transform

        self.y_images = os.listdir(root_y)
        self.x_images = os.listdir(root_x)
        self.length_dataset = max(len(self.y_images), len(self.x_images)) # 1000, 1500
        self.y_len = len(self.y_images)
        self.x_len = len(self.x_images)

    def __len__(self):
        return self.length_dataset

    def __getitem__(self, index):
        y_img = self.y_images[index % self.y_len]
        x_img = self.x_images[index % self.x_len]

        zebra_path = os.path.join(self.root_y, y_img)
        horse_path = os.path.join(self.root_x, x_img)

        y_img = np.array(Image.open(zebra_path).convert("RGB"))
        x_img = np.array(Image.open(horse_path).convert("RGB"))

        if self.transform:
            augmentations = self.transform(image=y_img, image0=x_img)
            y_img = augmentations["image"]
            x_img = augmentations["image0"]

        return y_img, x_img







"""# Architecture
<img src="https://imgs.search.brave.com/ycwQQQ4PY2s_Ai8M_pGgFYCJckpOr46RCAz7A8vBWqo/rs:fit:860:0:0/g:ce/aHR0cHM6Ly93d3cu/c2NhbGVyLmNvbS90/b3BpY3MvaW1hZ2Vz/L2FyY2hpdGVjdHVy/ZS1pbi1jeWNsZWdh/bi53ZWJw">

# Generator

<img src="https://imgs.search.brave.com/dfxQmMVRDzPdxSJThHcvoLUdMd874cHLbd7YbfbaUqs/rs:fit:860:0:0/g:ce/aHR0cHM6Ly9uZXVy/b2hpdmUuaW8vd3At/Y29udGVudC91cGxv/YWRzLzIwMTgvMTEv/VS1uZXQtbmV1cmFs/LW5ldHdvcmstbWVk/aWNpbmUucG5n">
"""

### For UNET
class DoubleConv(nn. Module):
    def __init__(self, in_channels, out_channels):
        super (DoubleConv, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, 1, 1, bias=False),
            nn.BatchNorm2d (out_channels),
            nn.ReLU(inplace=True),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, 3, 1, 1, bias=False),
            nn.BatchNorm2d (out_channels),
            nn.ReLU(inplace=True),
        )


    def forward (self, x):
        """Residual CONV"""
        x = self.conv1(x)
        return  x + self.conv2(x)

class UNET(nn.Module):
    """
    UNET Module
    """
    def __init__(self, in_channels=3, out_channels=3, features=[64, 128, 256, 512]):
        super (UNET, self).__init__()
        self.upsample_steps = nn.ModuleList()
        self.downsample_steps = nn.ModuleList()
        self.pool = nn. MaxPool2d (kernel_size=2, stride=2)

        # DownsamplingBlock of UNET
        for feature in features:
            self.downsample_steps.append (DoubleConv(in_channels, feature))
            in_channels = feature

        # ##
        self.bottleneck = DoubleConv(features[-1], features [-1]*2)

        # UpsamplingBlock of UNET
        for feature in reversed(features):
            self.upsample_steps.append(nn.ConvTranspose2d(feature*2, feature, kernel_size=2, stride=2))

            self.upsample_steps.append(DoubleConv(feature*2, feature))

        self.final_conv = nn.Conv2d(features[0], out_channels, kernel_size=1)

    def forward(self, x):
        skip_connections = []

        # downsample
        for downsample in self.downsample_steps:
            x = downsample(x)
            skip_connections.append(x)
            x = self.pool(x)

        # bottleneck - before upsample
        x = self.bottleneck(x)
        skip_connections = skip_connections[::-1]

        # upsample
        for index in range(0, len(self.upsample_steps), 2):
            x = self.upsample_steps[index](x)
            skip_connection = skip_connections[index//2]
            concat_skip = torch.cat((skip_connection, x), dim=1)
            x = self.upsample_steps[index+1](concat_skip)
        return self.final_conv(x)



def test():
    img_channels = 3
    img_size = 256
    x = torch.randn((2, img_channels, img_size, img_size))
    gen = UNET(img_channels)
    print(gen(x).shape)

# test()







"""# Discriminator

<img src="https://media.geeksforgeeks.org/wp-content/uploads/20200605220731/Discriminator.jpg">
"""

def patchify(images, n_patches):
    n, c, h, w = images.shape

    assert h == w, "Patchify method is implemented for square images only"

    patches = torch.zeros(n, n_patches ** 2, h * w * c // n_patches ** 2)
    patch_size = h // n_patches

    for idx, image in enumerate(images):
        for i in range(n_patches):
            for j in range(n_patches):
                patch = image[:, i * patch_size: (i + 1) * patch_size, j * patch_size: (j + 1) * patch_size]
                patches[idx, i * n_patches + j] = patch.flatten()
    return patches

def get_positional_embeddings(sequence_length, d):
    result = torch.ones(sequence_length, d)
    for i in range(sequence_length):
        for j in range(d):
            result[i][j] = np.sin(i / (10000 ** (j / d))) if j % 2 == 0 else np.cos(i / (10000 ** ((j - 1) / d)))
    return result

class ViTBlock(nn.Module):
    def __init__(self, hidden_d, n_heads, mlp_ratio=4):
        super(ViTBlock, self).__init__()
        self.hidden_d = hidden_d
        self.n_heads = n_heads

        self.norm1 = nn.LayerNorm(hidden_d)
        self.mhsa = MSA(hidden_d, n_heads)
        self.norm2 = nn.LayerNorm(hidden_d)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_d, mlp_ratio * hidden_d),
            nn.GELU(),
            nn.Linear(mlp_ratio * hidden_d, hidden_d)
        )

    def forward(self, x):
        out = x + self.mhsa(self.norm1(x))
        out = out + self.mlp(self.norm2(out))
        return out

class MSA(nn.Module):
    def __init__(self, d, n_heads=2):
        super(MSA, self).__init__()
        self.d = d
        self.n_heads = n_heads

        assert d % n_heads == 0, f"Can't divide dimension {d} into {n_heads} heads"

        d_head = int(d / n_heads)
        self.q_mappings = nn.ModuleList([nn.Linear(d_head, d_head) for _ in range(self.n_heads)])
        self.k_mappings = nn.ModuleList([nn.Linear(d_head, d_head) for _ in range(self.n_heads)])
        self.v_mappings = nn.ModuleList([nn.Linear(d_head, d_head) for _ in range(self.n_heads)])
        self.d_head = d_head
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, sequences):
        # Sequences has shape (N, seq_length, token_dim)
        # We go into shape    (N, seq_length, n_heads, token_dim / n_heads)
        # And come back to    (N, seq_length, item_dim)  (through concatenation)
        result = []
        for sequence in sequences:
            seq_result = []
            for head in range(self.n_heads):
                q_mapping = self.q_mappings[head]
                k_mapping = self.k_mappings[head]
                v_mapping = self.v_mappings[head]

                seq = sequence[:, head * self.d_head: (head + 1) * self.d_head]
                q, k, v = q_mapping(seq), k_mapping(seq), v_mapping(seq)

                attention = self.softmax(q @ k.T / (self.d_head ** 0.5))
                seq_result.append(attention @ v)
            result.append(torch.hstack(seq_result))
        return torch.cat([torch.unsqueeze(r, dim=0) for r in result])

class ViT(nn.Module):
    def __init__(self, chw, n_patches=7, n_blocks=2, hidden_d=8, n_heads=2, out_d=1):
        # Super constructor
        super(ViT, self).__init__()

        # Attributes
        self.chw = chw # ( C , H , W )
        self.n_patches = n_patches
        self.n_blocks = n_blocks
        self.n_heads = n_heads
        self.hidden_d = hidden_d

        # Input and patches sizes
        assert chw[1] % n_patches == 0, "Input shape not entirely divisible by number of patches"
        assert chw[2] % n_patches == 0, "Input shape not entirely divisible by number of patches"
        self.patch_size = (chw[1] / n_patches, chw[2] / n_patches)

        # 1) Linear mapper
        self.input_d = int(chw[0] * self.patch_size[0] * self.patch_size[1])
        self.linear_mapper = nn.Linear(self.input_d, self.hidden_d)

        # 2) Learnable classification token
        self.class_token = nn.Parameter(torch.rand(1, self.hidden_d))

        # 3) Positional embedding
        self.register_buffer('positional_embeddings', get_positional_embeddings(n_patches ** 2 + 1, hidden_d), persistent=False)

        # 4) Transformer encoder blocks
        self.blocks = nn.ModuleList([ViTBlock(hidden_d, n_heads) for _ in range(n_blocks)])

        # 5) Classification FCN
        self.mlp = nn.Sequential(
            nn.Linear(self.hidden_d, out_d),
            nn.Sigmoid()
        )

    def forward(self, images):
        # Dividing images into patches
        n, c, h, w = images.shape
        patches = patchify(images, self.n_patches).to(self.positional_embeddings.device)

        # Running linear layer tokenization
        # Map the vector corresponding to each patch to the hidden size dimension
        tokens = self.linear_mapper(patches)

        # Adding classification token to the tokens
        tokens = torch.cat((self.class_token.expand(n, 1, -1), tokens), dim=1)

        # Adding positional embedding
        out = tokens + self.positional_embeddings.repeat(n, 1, 1)

        # Transformer Blocks
        for block in self.blocks:
            out = block(out)
        return out
        # Getting the classification token only
        out = out[:, 0]

        return self.mlp(out)







class Block(nn.Module):
    """
    Basic building block for the discriminator called Block.
    This block consists of a convolutional layer with instance normalization and leaky ReLU activation.
    """
    def __init__(self, in_channels, out_channels, stride):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(
                in_channels,
                out_channels,
                4,
                stride,
                1,
                bias=True,
                padding_mode="reflect",
            ),
            nn.InstanceNorm2d(out_channels),
            nn.LeakyReLU(0.2, inplace=True),
        )

    ### Will perform Forward Prop
    def forward(self, x):
        return self.conv(x)





class Discriminator(nn.Module):
    """
    Intregated Discriminator Network
    """
    def __init__(self, in_channels=3, features=[64, 128, 256, 512]):
        super().__init__()
        self.initial = nn.Sequential(
            nn.Conv2d(
                in_channels,
                features[0],
                kernel_size=4,
                stride=2,
                padding=1,
                padding_mode="reflect",
            ),
            nn.LeakyReLU(0.2, inplace=True),
        )

        ### Layers for Discriminator
        layers = []
        in_channels = features[0]
        for feature in features[1:]:
            layers.append(
                Block(in_channels, feature,
                      stride=1 if feature == features[-1] else 2)
            )
            in_channels = feature

        ### Layer before output
        layers.append(
            nn.Conv2d(
                in_channels,
                1,
                kernel_size=4,
                stride=1,
                padding=1,
                padding_mode="reflect",
            )
        )
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        x = self.initial(x)
        return torch.sigmoid(self.model(x))

### Test Discriminator
def test():
    x = torch.randn((5, 3, 256, 256))
    # model = Discriminator(in_channels=3)
    model = ViT((3, 256, 256), n_patches=16, n_blocks=2, hidden_d=8, n_heads=2, out_d=1)
    preds = model(x)
    print(preds.shape)

# test()



# print(Discriminator(in_channels=3))

"""# Utilities"""

import os, numpy as np
import copy

def save_checkpoint(model, optimizer, filename="my_checkpoint.pth.tar"):
    print("=> Saving checkpoint")
    checkpoint = {
        "state_dict": model.state_dict(),
        "optimizer": optimizer.state_dict(),
    }
    torch.save(checkpoint, filename)


def load_checkpoint(checkpoint_file, model, optimizer, lr):
    print("=> Loading checkpoint")
    checkpoint = torch.load(checkpoint_file, map_location=DEVICE)
    model.load_state_dict(checkpoint["state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer"])

    # If we don't do this then it will just have learning rate of old checkpoint
    # and it will lead to many hours of debugging \:
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr


def seed_everything(seed=33):
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False













"""# Training"""

import sys
from torch.utils.data import DataLoader
import torch.optim as optim
from tqdm import tqdm
from torchvision.utils import save_image

def train_fn(disc_X, disc_Y, gen_Y, gen_X, loader, opt_disc, opt_gen, l1, mse, d_scaler, g_scaler):
    X_reals = 0
    X_fakes = 0
    loop = tqdm(loader, leave=True)

    for idx, (Y, X) in enumerate(loop):
        Y = Y.to(DEVICE)
        X = X.to(DEVICE)

        # Train Discriminators X and Y
        with torch.cuda.amp.autocast():
            fake_X = gen_X(Y)
            D_X_real = disc_X(X)
            D_X_fake = disc_X(fake_X.detach())
            X_reals += D_X_real.mean().item()
            X_fakes += D_X_fake.mean().item()
            D_X_real_loss = mse(D_X_real, torch.ones_like(D_X_real))
            D_X_fake_loss = mse(D_X_fake, torch.zeros_like(D_X_fake))
            D_X_loss = D_X_real_loss + D_X_fake_loss

            fake_Y = gen_Y(X)
            D_Y_real = disc_Y(Y)
            D_Y_fake = disc_Y(fake_Y.detach())
            D_Y_real_loss = mse(D_Y_real, torch.ones_like(D_Y_real))
            D_Y_fake_loss = mse(D_Y_fake, torch.zeros_like(D_Y_fake))
            D_Y_loss = D_Y_real_loss + D_Y_fake_loss

            # put it togethor
            D_loss = (D_X_loss + D_Y_loss) / 2

        opt_disc.zero_grad()
        d_scaler.scale(D_loss).backward()
        d_scaler.step(opt_disc)
        d_scaler.update()

        # Train Generators H and Z
        with torch.cuda.amp.autocast():
            # adversarial loss for both generators
            D_X_fake = disc_X(fake_X)
            D_Y_fake = disc_Y(fake_Y)
            loss_G_X = mse(D_X_fake, torch.ones_like(D_X_fake))
            loss_G_Y = mse(D_Y_fake, torch.ones_like(D_Y_fake))

            # cycle loss
            cycle_X = gen_Y(fake_X)
            cycle_Y = gen_X(fake_Y)
            cycle_Y_loss = l1(X, cycle_Y)
            cycle_X_loss = l1(Y, cycle_X)

            # identity loss (remove these for efficiency if you set lambda_identity=0)
            identity_Y = gen_Y(Y)
            identity_X = gen_X(X)
            identity_Y_loss = l1(Y, identity_Y)
            identity_X_loss = l1(X, identity_X)

            # add all togethor
            G_loss = (
                loss_G_Y
                + loss_G_X
                + cycle_Y_loss * LAMBDA_CYCLE
                + cycle_X_loss * LAMBDA_CYCLE
                + identity_X_loss * LAMBDA_IDENTITY
                + identity_Y_loss * LAMBDA_IDENTITY
            )

        opt_gen.zero_grad()
        g_scaler.scale(G_loss).backward()
        g_scaler.step(opt_gen)
        g_scaler.update()

        if idx % 50 == 0:
            save_image(X * 0.5 + 0.5, f"/content/drive/MyDrive/Colab Notebooks/Steganography/outputs_ViT/X_ORG_{idx}.png")
            save_image(Y * 0.5 + 0.5, f"/content/drive/MyDrive/Colab Notebooks/Steganography/outputs_ViT/Y_ORG_{idx}.png")
            save_image(fake_X * 0.5 + 0.5, f"/content/drive/MyDrive/Colab Notebooks/Steganography/outputs_ViT/X_{idx}.png")
            save_image(fake_Y * 0.5 + 0.5, f"/content/drive/MyDrive/Colab Notebooks/Steganography/outputs_ViT/Y_{idx}.png")

        loop.set_postfix(
            X_real=X_reals / (idx + 1),X_fake=X_fakes / (idx + 1),
            Generator_Loss=G_loss.item(), Discriminator_Loss=D_loss.item()
            )
    return G_loss.item(), D_loss.item()

def main():
    global metrics
    # disc_X = Discriminator(in_channels=3).to(DEVICE)
    disc_X = ViT((3, 256, 256), n_patches=16, n_blocks=2, hidden_d=8, n_heads=2, out_d=1)
    # disc_Y = Discriminator(in_channels=3).to(DEVICE)
    disc_Y = ViT((3, 256, 256), n_patches=16, n_blocks=2, hidden_d=8, n_heads=2, out_d=1)
    gen_Y = UNET(in_channels=3).to(DEVICE)
    gen_X = UNET(in_channels=3).to(DEVICE)
    opt_disc = optim.Adam(
        list(disc_X.parameters()) + list(disc_Y.parameters()),
        lr=LEARNING_RATE,
        betas=(0.5, 0.999),
    )

    opt_gen = optim.Adam(
        list(gen_Y.parameters()) + list(gen_X.parameters()),
        lr=LEARNING_RATE,
        betas=(0.5, 0.999),
    )

    L1 = nn.L1Loss()
    mse = nn.MSELoss()

    if LOAD_MODEL:
        load_checkpoint(
            CHECKPOINT_GEN_X,
            gen_X,
            opt_gen,
            LEARNING_RATE,
        )
        load_checkpoint(
            CHECKPOINT_GEN_Y,
            gen_Y,
            opt_gen,
            LEARNING_RATE,
        )
        load_checkpoint(
            CHECKPOINT_DISC_X,
            disc_X,
            opt_disc,
            LEARNING_RATE,
        )
        load_checkpoint(
            CHECKPOINT_DISC_Y,
            disc_Y,
            opt_disc,
            LEARNING_RATE,
        )

    dataset = HorseZebraDataset(
        root_x=TRAIN_DIR + "/horses",
        root_y=TRAIN_DIR + "/zebras",
        transform=transforms,
    )
    val_dataset = HorseZebraDataset(
        root_x=VAL_DIR + "/horses",
        root_y=VAL_DIR + "/zebras",
        transform=transforms,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=2,
        shuffle=False,
        pin_memory=True,
    )
    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=True,
    )
    g_scaler = torch.cuda.amp.GradScaler(enabled=False)
    d_scaler = torch.cuda.amp.GradScaler(enabled=False)

    for epoch in range(NUM_EPOCHS):
        g, d = train_fn(disc_X, disc_Y, gen_Y, gen_X, loader, opt_disc, opt_gen, L1, mse, d_scaler, g_scaler)
        metrics = pd.concat([metrics, pd.DataFrame({'discriminator':[d,], 'generator':[g,]})], ignore_index=True, axis=0)
        if SAVE_MODEL and (epoch%5 == 0):
            # save_checkpoint(gen_X, opt_gen, filename=CHECKPOINT_GEN_X)
            # save_checkpoint(gen_Y, opt_gen, filename=CHECKPOINT_GEN_Y)
            # save_checkpoint(disc_X, opt_disc,
            #                 filename=CHECKPOINT_DISC_X)
            # save_checkpoint(disc_Y, opt_disc,
            #                 filename=CHECKPOINT_DISC_Y)

            torch.save(gen_X.state_dict(), SAVE_PATH+'/GenX.pt')
            torch.save(gen_Y.state_dict(), SAVE_PATH+'/GenY.pt')
            torch.save(disc_X.state_dict(), SAVE_PATH+'/DiscX.pt')
            torch.save(disc_Y.state_dict(), SAVE_PATH+'/DiscY.pt')

metrics = pd.DataFrame({'discriminator':[0,], 'generator':[0,]})

seed_everything()
main()

"""# SAVE MODELS"""



metrics.to_csv("/content/drive/MyDrive/Colab Notebooks/Steganography/ViTMetrics.csv", index=False)

