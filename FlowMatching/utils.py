import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from IPython.display import HTML, display
from matplotlib.animation import FuncAnimation
from torch import nn
from tqdm import tqdm
from zuko.utils import odeint

def create_dataset(size: int = 100_000):
    complex_points = torch.polar(torch.tensor(1.0), torch.rand(size) * 2 * torch.pi)
    X = torch.stack((complex_points.real, complex_points.imag)).T
    upper = complex_points.imag > 0
    left = complex_points.real < 0
    X[upper, 1] = 0.5
    X[upper & left, 0] = -0.5
    X[upper & ~left, 0] = 0.5
    noise = torch.zeros_like(X)
    noise[upper] = torch.randn_like(noise[upper]) * 0.10
    noise[~upper] = torch.randn_like(noise[~upper]) * 0.05
    X += noise
    X -= X.mean(axis=0)
    X /= X.std(axis=0)
    return X + noise


def plot_dataset(X, bins, ax=None, **kwargs):
    if ax is None:
        ax = plt.gca()
    ax.hist2d(*X.T, bins=bins)
    ax.set_xlabel('feature 0')
    ax.set_ylabel('feature 1')
    ax.set(**kwargs)


class FlowModel(nn.Module):

    def forward(self, X, time):
        raise NotImplementedError()


class ExampleFlow(FlowModel):  # a random model

    def forward(self, X, time):
        result = torch.zeros_like(X)
        result[:, 0] = -X[:, 0] * time
        result[:, 1] = X[:, 1] * (1 - time)
        return result


@torch.no_grad()
def plot_flow_at_time(flow_model, time, ax=None, **kwargs):
    if ax is None:
        ax = plt.gca()
    points = torch.linspace(-1, 1, 10)
    flow_input = torch.cartesian_prod(points, points)
    flow_output = flow_model(flow_input, time=torch.full(flow_input.shape[:1], time))
    ax.quiver(
        torch.stack(torch.chunk(flow_input[:, 0], len(points))).numpy(),
        torch.stack(torch.chunk(flow_input[:, 1], len(points))).numpy(),
        torch.stack(torch.chunk(flow_output[:, 0], len(points))).numpy(),
        torch.stack(torch.chunk(flow_output[:, 1], len(points))).numpy(),
        scale=len(points),
    )
    ax.set_xlabel('feature 0')
    ax.set_ylabel('feature 1')
    ax.set(**kwargs)


def animate_flow(flow_model, frames: int = 20):
    def plot_frame(time):
        plt.cla()
        plot_flow_at_time(flow_model, time=time, title=f'flow at time={time:.2f}')

    fig = plt.figure(figsize=(8, 8))
    FuncAnimation(fig, plot_frame, frames=np.linspace(0, 1, frames))
    plt.show()
    plt.close()


@torch.no_grad()
def run_flow(flow_model, x_0, t_0, t_1, device='cpu'):
    def f(t: float, x):
        return flow_model(x, time=torch.full(x.shape[:1], t, device=device))

    return odeint(f, x_0, t_0, t_1, phi=flow_model.parameters())


def animate_flow_run(flow_model, X, frames=20, device='cpu'):
    bins = [
        np.linspace(X[:, 0].min().cpu(), X[:, 0].max().cpu(), 128),
        np.linspace(X[:, 1].min().cpu(), X[:, 1].max().cpu(), 128),
    ]

    def plot_frame(time):
        plt.cla()
        plot_dataset(run_flow(flow_model, X, 0, time, device=device).cpu(), bins=bins,
                     title=f'distribution at time {time:.2f}')

    fig = plt.figure(figsize=(8, 8))
    FuncAnimation(fig, plot_frame, frames=np.linspace(0, 1, frames))
    plt.show()
    plt.close()




class TimeEmbedding(nn.Module):

    def __init__(self, dim: int):
        super().__init__()
        assert dim % 2 == 0
        self.dim = dim
        self.register_buffer('freqs', torch.arange(1, dim // 2 + 1) * torch.pi)

    def forward(self, t):
        emb = self.freqs * t[..., None]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb



class FlowNeuralNetwork(FlowModel):

    def __init__(self, n_features, time_embedding_size=8, n_blocks=2):
        super().__init__()
        self.time_embedding = TimeEmbedding(time_embedding_size)
        hidden_size = n_features + time_embedding_size
        blocks = []
        for _ in range(n_blocks):
            blocks.append(nn.Sequential(
                nn.BatchNorm1d(hidden_size),
                nn.Linear(hidden_size, hidden_size),
                nn.ReLU(),
            ))
        self.blocks = nn.ModuleList(blocks)
        self.final = nn.Linear(hidden_size, n_features)


    def forward(self, X, time):
        X = torch.cat([X, self.time_embedding(time)], axis=1)
        for block in self.blocks:
            X = X + block(X)
        X = self.final(X)
        return X


def conditional_flow_matching_loss(flow_model, x):
    sigma_min = 1e-4
    t = torch.rand(x.shape[0], device=x.device)
    noise = torch.randn_like(x)

    x_t = (1 - (1 - sigma_min) * t[:, None]) * noise + t[:, None] * x
    optimal_flow = x - (1 - sigma_min) * noise
    predicted_flow = flow_model(x_t, time=t)

    return (predicted_flow - optimal_flow).square().mean()