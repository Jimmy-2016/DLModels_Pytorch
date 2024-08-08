import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt


# Define the affine coupling layer
class AffineCouplingLayer(nn.Module):
    def __init__(self, in_features):
        super(AffineCouplingLayer, self).__init__()
        self.in_features = in_features
        self.split_size = in_features // 2

        # Define the neural network for the coupling layer
        self.nn = nn.Sequential(
            nn.Linear(self.split_size, 128),
            nn.ReLU(),
            nn.Linear(128, self.split_size * 2)  # Output both scaling and shifting parameters
        )

    def forward(self, x):
        x1, x2 = x[:, :self.split_size], x[:, self.split_size:]
        params = self.nn(x1)
        s, t = params[:, :self.split_size], params[:, self.split_size:]
        s = torch.tanh(s)  # Ensure the scaling is bounded
        x2 = x2 * torch.exp(s) + t
        return torch.cat([x1, x2], dim=1), s

    def inverse(self, y):
        y1, y2 = y[:, :self.split_size], y[:, self.split_size:]
        params = self.nn(y1)
        s, t = params[:, :self.split_size], params[:, self.split_size:]
        s = torch.tanh(s)
        y2 = (y2 - t) * torch.exp(-s)
        return torch.cat([y1, y2], dim=1), s


# Define the normalizing flow model
class NormalizingFlow(nn.Module):
    def __init__(self, in_features, num_layers):
        super(NormalizingFlow, self).__init__()
        self.layers = nn.ModuleList([AffineCouplingLayer(in_features) for _ in range(num_layers)])

    def forward(self, x):
        log_det_jacobian = 0
        for layer in self.layers:
            x, s = layer(x)
            log_det_jacobian += s.sum(dim=1)
        return x, log_det_jacobian

    def inverse(self, y):
        for layer in reversed(self.layers):
            y, s = layer.inverse(y)
        return y


# Generate the target distribution (circle)
def target_distribution(num_samples):
    theta = np.linspace(0, 2 * np.pi, num_samples)
    r = 1  # Radius of the circle
    x = r * np.cos(theta)
    y = r * np.sin(theta)
    return np.vstack([x, y]).T

