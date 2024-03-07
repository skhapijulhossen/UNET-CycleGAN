import torch
import torch.nn as nn
import torch.nn.functional as F
from torchmetrics.image.inception import InceptionScore
from torchmetrics.image.fid import FrechetInceptionDistance
from torchmetrics.image import PeakSignalNoiseRatio
from torch.utils.data import DataLoader
from dataset import HorseZebraDataset


def evaluate(real_images, fake_images):
    """
    Calculates the FID of a batch of real and fake images.

    Args:
        real_images: A PyTorch tensor of real images.
        fake_images: A PyTorch tensor of fake images.

    Returns:
        Tuple Scores.
    """

    # Calculate the mean squared error between the real and fake images.
    real_images = real_images.type(torch.FloatTensor)
    fake_images = fake_images.type(torch.FloatTensor)
    mse_loss = nn.MSELoss()(real_images, fake_images)

    # Example plotting a single value

    metric = PeakSignalNoiseRatio()
    psnr = metric(fake_images, real_images)
    # fig_, ax_ = metric.plot(psnr)

    # Computes the inception score of the generated images.
    real_images = real_images.type(torch.uint8)
    fake_images = fake_images.type(torch.uint8)
    inception = InceptionScore()
    inception.update(real_images)
    inc_score = inception.compute()
    inception.reset()

    # Calculates the FID of a batch of real and fake images.
    fid = FrechetInceptionDistance(feature=64)
    fid.update(real_images, real=True)
    fid.update(fake_images, real=False)
    fid_score = fid.compute()


    return dict(MSELoss=mse_loss.item(), PeakSignalNoiseRatio=psnr.item(), InceptionScore_mean=inc_score[0].item(), InceptionScore_std=inc_score[1].item(), FrechetInceptionDistance=fid_score.item())


