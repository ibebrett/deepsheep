import argparse

from tqdm import tqdm
import numpy as np
import torch
import torch.utils.data

import torchvision
from torch import nn, optim
from torch.utils.data.sampler import SubsetRandomSampler, SequentialSampler
from torch.nn import functional as F

from torchvision import datasets, transforms

from torchvision.utils import save_image


class VAE(nn.Module):
    '''Simple VAE network, based on the example VAE found in
    the pytorch examples.
    '''

    def __init__(self, image_w=200, image_h=200, latent_size=30):
        super(VAE, self).__init__()

        self._image_w = image_w
        self._image_h = image_h
        self._latent_size = latent_size

        # Encoder layer into latent space.
        self._encoder = nn.Sequential(
            nn.Linear(3 * self._image_w * self._image_h, 400), nn.ReLU())

        # Latent space to prob space
        self._mean = nn.Linear(400, self._latent_size)
        self._std = nn.Linear(400, self._latent_size)

        # Latent space to decoded space.
        self._decoder = nn.Sequential(
            nn.Linear(self._latent_size, 400), nn.ReLU(),
            nn.Linear(400, 3 * self._image_w * self._image_h), nn.Sigmoid())

    def encode(self, x):
        h1 = self._encoder(x)
        return self._mean(h1), self._std(h1)

    def reparameterize(self, mean, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mean + eps * std

    def decode(self, z):
        return self._decoder(z)

    def forward(self, x):
        mean, logvar = self.encode(
            x.view(-1, 3 * self._image_w * self._image_h))
        z = self.reparameterize(mean, logvar)
        return self.decode(z), mean, logvar, z


def get_data_loaders(path,
                     resize=(200, 200),
                     batch_size=16,
                     loader_processes=4,
                     shuffle=True,
                     small=False):
    ds = datasets.ImageFolder(
        path,
        transforms.Compose([
            transforms.Resize(resize),
            transforms.ToTensor(),
        ]))

    ds_size = len(ds)
    if small:
        ds_size = int(ds_size * 0.1)

    # TODO, we should probably split in a way that is sensitive
    # to time series. We don't want to put data into validation that
    # is extremely close data in train (due to being very close in time).
    indices = list(range(ds_size))
    np.random.shuffle(indices)

    split = int(np.floor(0.7 * ds_size))

    train_indices, test_indices = indices[:split], indices[split:]

    # Creating PT data samplers and loaders.
    train_sampler = SubsetRandomSampler(train_indices)
    test_ds = torch.utils.data.Subset(ds, test_indices)

    train_loader = torch.utils.data.DataLoader(ds,
                                               batch_size=batch_size,
                                               sampler=train_sampler,
                                               pin_memory=True,
                                               num_workers=loader_processes)
    test_loader = torch.utils.data.DataLoader(test_ds,
                                              batch_size=batch_size,
                                              pin_memory=True,
                                              num_workers=loader_processes)

    return train_loader, test_loader


# NOTE: Took this from:
# https://github.com/pytorch/examples/blob/master/vae/main.py
# Some modifications:
# 1. Clamped the inputs to binary_cross_entropy in attempt to avoid a bug.
def loss_function(recon_x, x, mean, logvar, kld_weight):
    BCE = F.binary_cross_entropy(torch.clamp(recon_x, 0.000001, 0.999999),
                                 torch.clamp(x.view(-1, 3 * 200 * 200),
                                             0.000001, 0.999999),
                                 reduction='sum')

    # see Appendix B from VAE paper:
    # Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
    # https://arxiv.org/abs/1312.6114
    # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    KLD = -0.5 * torch.sum(1 + logvar - mean.pow(2) - logvar.exp())

    return BCE, KLD, BCE + (KLD * kld_weight)


def train(epoch, device, model, optimizer, train_loader, kld_weight):
    model.train()

    total_loss = 0
    total_bce_loss = 0
    total_kld_loss = 0
    for i, (data, _) in tqdm(enumerate(train_loader)):
        optimizer.zero_grad()
        X = data.to(device)

        y, mean, logvar, z = model(X)

        bce_loss, kld_loss, loss = loss_function(y, X, mean, logvar,
                                                 kld_weight)
        loss.backward()
        optimizer.step()
        bce_loss = bce_loss.detach().cpu().numpy()
        kld_loss = kld_loss.detach().cpu().numpy()
        loss = loss.detach().cpu().numpy()

        total_loss += loss
        total_bce_loss += bce_loss
        total_kld_loss += kld_loss

        if (i / len(train_loader)) > 0.2:
            # leave early due to weird problems. shuffling helps, save more.
            break

    print(
        json.dumps({
            'epoch': epoch,
            'metric': 'train_loss',
            'value': total_loss
        }))
    print(
        json.dumps({
            'epoch': epoch,
            'metric': 'train_kld_loss',
            'value': total_kld_loss
        }))
    print(
        json.dumps({
            'epoch': epoch,
            'metric': 'train_bce_loss',
            'value': total_bce_loss
        }))

    return total_loss / len(train_loader)


def test(epoch, device, model, test_loader, kld_weight):
    model.eval()

    total_loss = 0
    total_bce_loss = 0
    total_kld_loss = 0
    with torch.no_grad():
        for i, (data, _) in tqdm(enumerate(test_loader)):
            X = data.to(device)
            y, mean, logvar, z = model(X)
            bce_loss, kld_loss, loss = loss_function(y, X, mean, logvar,
                                                     kld_weight)

            bce_loss = bce_loss.detach().cpu().numpy()
            kld_loss = kld_loss.detach().cpu().numpy()
            loss = loss.detach().cpu().numpy()

            total_loss += loss
            total_bce_loss += bce_loss
            total_kld_loss += kld_loss

            if i == 0:
                # For the same first batch of validation, emit the images.
                unstacked_y = y.view(X.shape).cpu()
                stacked_image = torch.cat((X.cpu(), unstacked_y), dim=0)
                torchvision.utils.save_image(
                    torchvision.utils.make_grid(stacked_image),
                    f'validate{epoch:04}.png')

            if (i / len(test_loader)) > 0.2:
                break
    print(
        json.dumps({
            'epoch': epoch,
            'metric': 'test_loss',
            'value': total_loss
        }))
    print(
        json.dumps({
            'epoch': epoch,
            'metric': 'test_kld_loss',
            'value': total_kld_loss
        }))
    print(
        json.dumps({
            'epoch': epoch,
            'metric': 'test_bce_loss',
            'value': total_bce_loss
        }))

    return total_loss / len(test_loader)


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


import json


def main():
    parser = argparse.ArgumentParser(description='Train a VAE of sheep')
    parser.add_argument('data')
    parser.add_argument('--small', type=bool, default=False)
    parser.add_argument('--model_path', type=str, default='model')
    parser.add_argument('--kld_weight', type=float, default=5.0)
    parser.add_argument('--latent-size', type=int, default=10)
    parser.add_argument('--batch-size', type=int, default=128)
    parser.add_argument('--learning-rate', type=float, default=1e-3)
    parser.add_argument('--loader-processes', type=int, default=4)
    parser.add_argument('--epochs', type=int, default=5)

    args = parser.parse_args()

    # write args to disk so they can be used for hyperparameter search.
    with open('args.json', 'w') as f:
        json.dump(vars(args), f)

    model_path = args.model_path
    data_path = args.data



    device = torch.device("cuda")

    model = VAE(latent_size=10).to(device)

    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)

    epoch = 0

    train_data_loader, test_data_loader = get_data_loaders(
        data_path,
        small=args.small,
        batch_size=args.batch_size,
        loader_processes=args.loader_processes)

    epochs = []
    metrics = {'train': [], 'test': []}

    while epoch < args.epochs:
        print('epoch', epoch)

        epoch += 1
        epochs.append(epoch)
        metrics['train'].append(
            train(epoch, device, model, optimizer, train_data_loader,
                  args.kld_weight))

        save_model(model_path, model, optimizer, epoch)

        metrics['test'].append(
            test(epoch, device, model, test_data_loader, args.kld_weight))


if __name__ == '__main__':
    main()
