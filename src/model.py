import torch
from torch import nn


class Encoder(nn.Module):
    def __init__(self, input_size: int, hidden_dim: int):
        super(Encoder, self).__init__()

        self.hidden_layer = nn.Sequential(
            nn.Linear(input_size, hidden_dim),
            nn.ELU()
        )
        self.mean = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh()
        )
        self.var = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh()
        )

    def forward(self, inputs):
        latent = self.hidden_layer(inputs)
        mean = self.mean(latent)
        log_var = self.var(latent)
        return mean, log_var


class Sampler(nn.Module):
    def __init__(self):
        super(Sampler, self).__init__()

    def forward(self, mean, log_var):
        eps = torch.randn_like(log_var)
        if self.training:
            return mean + torch.exp(0.5 * log_var) * eps
        return mean


class Decoder(nn.Module):
    def __init__(self, input_size: int, hidden_dim: int):
        super(Decoder, self).__init__()

        self.output_layer = nn.Sequential(
            nn.Linear(hidden_dim, input_size),
            nn.Softmax(dim=1)
        )

    def forward(self, inputs):
        return self.output_layer(inputs)


class MVAE(nn.Module):
    def __init__(self, input_size: int, hidden_dim: int):
        super(MVAE, self).__init__()
        self.input_size = input_size
        self.hidden_dim = hidden_dim

        self.encoder = Encoder(input_size=input_size, hidden_dim=hidden_dim)
        self.sampler = Sampler()
        self.decoder = Decoder(input_size=input_size, hidden_dim=hidden_dim)

    def forward(self, inputs):
        mean, log_var = self.encoder(inputs)
        if self.training:
            z = self.sampler(mean, log_var)
            outputs = self.decoder(z)
            return outputs, mean, log_var
        else:
            outputs = self.decoder(mean)
            return outputs
