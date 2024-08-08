import matplotlib.pyplot as plt
import torch.optim

from utils import *


dataset = create_dataset()
noise = torch.randn_like(dataset)

fig, axs = plt.subplots(1, 2, figsize=(12, 6))
plot_dataset(noise, bins=64, ax=axs[0], title='noise')
plot_dataset(dataset, bins=64, ax=axs[1], title='dataset')


animate_flow(ExampleFlow())
#
# animate_flow_run(ExampleFlow(), noise)

model = FlowNeuralNetwork(n_features=2, n_blocks=4)
# animate_flow(model)
# animate_flow_run(model, noise)
# animate_flow_run(model, noise)


## Train the model

model.train()
optimizer = torch.optim.Adam(model.parameters(), lr=0.1)

for epoch in tqdm(range(1000)):
    model.zero_grad()
    loss = conditional_flow_matching_loss(model, create_dataset(1000))
    loss.backward()
    optimizer.step()

model = model.eval()
animate_flow_run(model, noise)


# plt.show()