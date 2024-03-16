import torch
import torch.nn as nn
import torch.nn.functional as F


class Encoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, latent_dim):
        """
        input_dim: int, dimension of input data
        hidden_dim: int, dimension of hidden layer
        latent_dim: int, dimension of latent space

        Linear layers are used instead of convolutional layers for simplicity.
        """
        super(Encoder, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc_mean = nn.Linear(hidden_dim, latent_dim)
        self.fc_log_var = nn.Linear(hidden_dim, latent_dim)

    def forward(self, x):
        hidden = F.relu(self.fc1(x))
        mean = self.fc_mean(hidden)
        log_var = self.fc_log_var(hidden)
        return mean, log_var


class Decoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, latent_dim):
        """
        input_dim: int, dimension of input data
        hidden_dim: int, dimension of hidden layer
        latent_dim: int, dimension of latent space
        """
        super(Decoder, self).__init__()
        self.fc1 = nn.Linear(latent_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, input_dim)

    def forward(self, x):
        hidden = F.relu(self.fc1(x))
        reconstruction = torch.sigmoid(
            self.fc2(hidden)
        )  # use sigmoid activation function to make sure the output is between 0 and 1
        return reconstruction


class VAE(nn.Module):
    def __init__(self, input_dim, hidden_dim, latent_dim):
        """
        input_dim: int, dimension of input data
        hidden_dim: int, dimension of hidden layer
        latent_dim: int, dimension of latent space
        """
        super(VAE, self).__init__()
        self.encoder = Encoder(
            input_dim, hidden_dim, latent_dim
        )  # Encoder is used to encode input data into latent space
        self.decoder = Decoder(
            input_dim, hidden_dim, latent_dim
        )  # Decoder is used to decode latent space into input data

    def reparameterize(self, mean, log_var):
        """
        mean: tensor, mean of latent space
        log_var: tensor, log variance of latent space
        """
        std = torch.exp(
            0.5 * log_var
        )  # standard deviation, 0.5 is used to convert log variance to variance
        eps = torch.randn_like(
            std
        )  # random noise, same shape as std because we want to sample from normal distribution with mean 0 and variance 1
        return (
            mean + eps * std
        )  # reparameterization trick, sample from normal distribution with mean and variance of latent space to get latent space

    def forward(self, x):
        mean, log_var = self.encoder(x)
        z = self.reparameterize(mean, log_var)
        reconstruction = self.decoder(z)
        return reconstruction, mean, log_var
