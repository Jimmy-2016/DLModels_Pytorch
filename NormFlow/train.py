
from utils import *


## Params
d, k = 2, 1
n_samples = 1024
epochs = 1000
batch_size = 128
##
model = R_NVP(d, k, hidden=100)
optim = torch.optim.Adam(model.parameters(), lr=0.1)
scheduler = torch.optim.lr_scheduler.ExponentialLR(optim, 0.999)
# x_data = torch.randn(n_samples, 2)
# target_data = torch.tensor(target_distribution(n_samples), dtype=torch.float32)

losses = []
for ep in range(epochs):
    # get batch
    X, _ = datasets.make_moons(n_samples=batch_size, noise=.05)
    X = torch.from_numpy(StandardScaler().fit_transform(X)).float()

    optim.zero_grad()
    z, log_pz, log_jacob = model(X)
    loss = (-log_pz - log_jacob).mean()
    losses.append(loss.item())

    loss.backward()
    optim.step()
    scheduler.step()

    if ep % 100 == 0:
        print(f'Epoch {ep}, Loss: {loss.item()}')

view(model, losses)
