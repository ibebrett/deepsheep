import torch

from torch import nn


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


class View(nn.Module):
    def __init__(self, shape):
        super(View, self).__init__()
        self.shape = shape

    def forward(self, x):
        return x.view(*self.shape)


class ConvVAE(nn.Module):
    '''Simple VAE network, based on the example VAE found in 
    the pytorch examples.
    '''

    def __init__(self, image_w=200, image_h=200, latent_size=30):
        super(ConvVAE, self).__init__()

        self._image_w = image_w
        self._image_h = image_h
        self._latent_size = latent_size

        # Encoder layer into latent space.
        self._encoder = nn.Sequential(nn.Conv2d(3, 1, 20, stride=20),
                                      nn.Sigmoid(), View([-1, 100]))

        # Latent space to prob space
        self._mean = nn.Sequential(nn.Linear(100, self._latent_size),
                                   nn.Sigmoid())

        self._std = nn.Sequential(nn.Linear(100, self._latent_size),
                                  nn.Sigmoid())

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
        mean, logvar = self.encode(x)
        z = self.reparameterize(mean, logvar)
        return self.decode(z), mean, logvar, z


MODELS = {'simple': VAE, 'conv': ConvVAE}
