import matplotlib.pyplot as plt

from utils import *


dataset = create_dataset()
noise = torch.randn_like(dataset)

fig, axs = plt.subplots(1, 2, figsize=(12, 6))
plot_dataset(noise, bins=64, ax=axs[0], title='noise')
plot_dataset(dataset, bins=64, ax=axs[1], title='dataset')


animate_flow(ExampleFlow())
#
# animate_flow_run(ExampleFlow(), noise)

model = FlowNeuralNetwork(n_features=2)
animate_flow(model)
animate_flow_run(model, noise)
animate_flow_run(model, noise)


# plt.show()