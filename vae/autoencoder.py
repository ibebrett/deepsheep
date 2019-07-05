from itertools import islice

import argparse

from tqdm import tqdm
import numpy as np

import torch
import torch.utils.data

import torchvision
from torch import nn, optim
from torch.nn import functional as F

from torchvision import datasets, transforms
from torchvision.utils import save_image

from models import VAE


def get_data_loader(path, resize=(200, 200), batch_size=32, shuffle=True):
    ds = datasets.ImageFolder(
        path,
        transforms.Compose([
            transforms.Resize(resize),
            transforms.ToTensor(),
        ]))

    return torch.utils.data.DataLoader(ds,
                                       batch_size=32,
                                       num_workers=8,
                                       pin_memory=True,
                                       shuffle=shuffle)


# NOTE: Took this from:
# https://github.com/pytorch/examples/blob/master/vae/main.py
# Some modifications:
# 1. Clamped the inputs to binary_cross_entropy in attempt to avoid a bug.
def loss_function(recon_x, x, mean, logvar):
    BCE = F.binary_cross_entropy(torch.clamp(recon_x, 0.000001, 0.999999),
                                 torch.clamp(x.view(-1, 3 * 200 * 200),
                                             0.000001, 0.999999),
                                 reduction='sum')

    # see Appendix B from VAE paper:
    # Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
    # https://arxiv.org/abs/1312.6114
    # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    KLD = -0.5 * torch.sum(1 + logvar - mean.pow(2) - logvar.exp())

    return BCE + KLD


def train(epoch, device, model, optimizer, train_loader):
    model.train()

    for i, (data, _) in tqdm(enumerate(train_loader)):
        optimizer.zero_grad()
        X = data.to(device)

        y, mean, logvar, z = model(X)

        loss = loss_function(y, X, mean, logvar)

        loss.backward()
        optimizer.step()

        if (i / len(train_loader)) > 0.2:
            # leave early due to weird problems. shuffling helps, save more.
            break


def validate(epoch, device, model, validate_data):
    X = validate_data.to(device)
    y, mean, logvar, z = model(X)

    unstacked_y = y.view(X.shape).cpu()
    stacked_image = torch.cat((X.cpu(), unstacked_y), dim=0)
    torchvision.utils.save_image(torchvision.utils.make_grid(stacked_image),
                                 f'validate{epoch:04}.png')


def load_state(path, model, optimizer):
    try:
        checkpoint = torch.load(path)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        epoch = checkpoint['epoch']
    except FileNotFoundError:
        epoch = 0

    return epoch


def save_model(path, model, optimizer, epoch):
    torch.save(
        {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
        }, path)


def main():
    parser = argparse.ArgumentParser(description='Train a VAE of sheep')
    parser.add_argument('model_path')
    parser.add_argument('data_path')

    args = parser.parse_args()

    device = torch.device("cuda")

    model = VAE().to(device)

    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    epoch = load_state(args.model_path, model, optimizer)

    data_loader = get_data_loader(args.data_path)

    # steal some fixed data for validation.
    validate_data, _ = next(iter(data_loader))

    while epoch < 100:
        print('epoch', epoch)
        epoch += 1
        train(epoch, device, model, optimizer, data_loader)
        save_model(args.model_path, model, optimizer, epoch)

        validate(epoch, device, model, validate_data)


if __name__ == '__main__':
    main()
