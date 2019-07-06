from itertools import islice

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

from models import VAE

import visdom


def get_data_loaders(path,
                     resize=(200, 200),
                     batch_size=16,
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

    train_indices, test_indices = indices[split:], indices[:split]

    # Creating PT data samplers and loaders.
    train_sampler = SubsetRandomSampler(train_indices)
    test_ds = torch.utils.data.Subset(ds, test_indices)

    train_loader = torch.utils.data.DataLoader(ds,
                                               batch_size=batch_size,
                                               sampler=train_sampler,
                                               pin_memory=True)
    test_loader = torch.utils.data.DataLoader(test_ds,
                                              batch_size=batch_size,
                                              pin_memory=True)

    return train_loader, test_loader


# NOTE: Took this from:
# https://github.com/pytorch/examples/blob/master/vae/main.py
# Some modifications:
# 1. Clamped the inputs to binary_cross_entropy in attempt to avoid a bug.
def loss_function(recon_x, x, mean, logvar):
    BCE = F.binary_cross_entropy(torch.clamp(recon_x, 0.000001, 0.999999),
                                 torch.clamp(x.view(-1, 3 * 200 * 200),
                                             0.000001, 0.999999),
                                 reduction='sum')

    KLD_weight = 5.0
    # see Appendix B from VAE paper:
    # Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
    # https://arxiv.org/abs/1312.6114
    # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    KLD = -0.5 * torch.sum(1 + logvar - mean.pow(2) - logvar.exp())

    return BCE + (KLD * KLD_weight)


def train(epoch, device, model, optimizer, train_loader, vis):
    model.train()

    total_loss = 0
    zs = []
    for i, (data, _) in tqdm(enumerate(train_loader)):
        optimizer.zero_grad()
        X = data.to(device)

        y, mean, logvar, z = model(X)

        zs.extend(z.detach().cpu().numpy())
        loss = loss_function(y, X, mean, logvar)
        total_loss += loss.detach().cpu().numpy()
        loss.backward()
        optimizer.step()

        if (i / len(train_loader)) > 0.2:
            # leave early due to weird problems. shuffling helps, save more.
            break

    print('embedding')
    print(np.array(zs[:200]).shape)
    tsne_embedding(vis, np.array(zs), 'train embedding')
    print('done embedding')
    print('total_loss', total_loss)

    return total_loss / len(train_loader)


def tsne_embedding(vis, zs, name):
    pass
    #import numpy as np
    #from sklearn.manifold import TSNE
    #X_embedded = TSNE(n_components=2).fit_transform(zs)
    #vis.scatter(X_embedded, name=name)


def test(epoch, device, model, test_loader, vis):
    model.eval()

    total_loss = 0
    zs = []
    with torch.no_grad():
        for i, (data, _) in tqdm(enumerate(test_loader)):
            X = data.to(device)
            y, mean, logvar, z = model(X)
            zs.extend(z.cpu().numpy())
            total_loss += loss_function(y, X, mean, logvar).cpu().numpy()

            if i == 0:
                # For the same first batch of validation, emit the images.
                unstacked_y = y.view(X.shape).cpu()
                stacked_image = torch.cat((X.cpu(), unstacked_y), dim=0)
                torchvision.utils.save_image(
                    torchvision.utils.make_grid(stacked_image),
                    f'validate{epoch:04}.png')
                vis.images(stacked_image, opts=dict(caption=f'epoch {epoch}'))

            if (i / len(test_loader)) > 0.2:
                break

    print('embedding')
    print(np.array(zs[:200]).shape)
    tsne_embedding(vis, np.array(zs), 'test embedding')
    print('done embedding')
    print('total_loss', total_loss)
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


def main():
    parser = argparse.ArgumentParser(description='Train a VAE of sheep')
    parser.add_argument('model_path')
    parser.add_argument('data_path')
    parser.add_argument('--small', type=bool, default=False)

    args = parser.parse_args()

    device = torch.device("cuda")

    model = VAE(latent_size=4).to(device)

    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    epoch = load_state(args.model_path, model, optimizer)

    train_data_loader, test_data_loader = get_data_loaders(args.data_path,
                                                           small=args.small,
                                                           batch_size=128)

    vis = visdom.Visdom()

    epochs = []
    metrics = {'train': [], 'test': []}

    while epoch < 100:
        print('epoch', epoch)

        epoch += 1
        epochs.append(epoch)
        metrics['train'].append(
            train(epoch, device, model, optimizer, train_data_loader, vis))

        save_model(args.model_path, model, optimizer, epoch)

        metrics['test'].append(
            test(epoch, device, model, test_data_loader, vis))

        Y = np.array([metrics['train'], metrics['test']]).T
        vis.line(X=epochs,
                 Y=Y,
                 opts=dict(title=f'epoch {epoch}', legend=['Train', 'Test']))


if __name__ == '__main__':
    main()
