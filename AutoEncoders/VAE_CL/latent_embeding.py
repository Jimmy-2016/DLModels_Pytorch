

from vae_model import *
import torch.utils.data
import matplotlib.pyplot as plt
import torch.optim as optim
import numpy as np
import torchvision
import umap


batch_size_test = 20000

test_loader = torch.utils.data.DataLoader(
    torchvision.datasets.MNIST('./data/', train=False, download=True,
                               transform=torchvision.transforms.Compose([
                                   torchvision.transforms.ToTensor(),
                                   torchvision.transforms.Normalize((0.1307,), (0.3081,))
                               ])


    ),
    batch_size=batch_size_test, shuffle=True
)


torch.manual_seed(1)
np.random.seed(1)

Conditional = True
Contrast = True

if Contrast == True:
    Conditional = False

if Conditional:
    PATH = './saved_model/con_model.pth'
elif Contrast:
    PATH = './saved_model/Contrast_model.pth'
else:
    PATH = './saved_model/nocon_model.pth'


model = myVAE(layers=[784, 100, 50, 10], conditional=Conditional)

model.load_state_dict(torch.load(PATH))


exmaple = enumerate(test_loader)
batch_index, (data, target) = next(exmaple)

input = data.view(data.shape[0], -1)
# latenlayer = model.encoder(input)
out, mu, sigma, encoder_out, z = model(input, target)



# plot_exmaples(data, model(data.view(data.shape[0], -1)))

print('############## End #################')


##

umap_hparams = {'n_neighbors': 5,
                'min_dist': 0.1,
                'n_components': 2,
                'metric': 'euclidean'}


fig, ax = plt.subplots(constrained_layout=False)
ax.set(xticks=[], yticks=[])

umap_embedding = umap.UMAP(n_neighbors=umap_hparams['n_neighbors'], min_dist=umap_hparams['min_dist'],
                           n_components=umap_hparams['n_components'],
                           metric=umap_hparams['metric']).fit_transform(mu.detach().numpy())
scatter = ax.scatter(x=umap_embedding[:, 0], y=umap_embedding[:, 1], s=20, c=target, cmap='tab10')

cbar = plt.colorbar(scatter, boundaries=np.arange(11)-0.5)
cbar.set_ticks(np.arange(10))
cbar.set_ticklabels(np.arange(10))

plt.title('UMAP Dimensionality reduction', fontsize=25, fontweight='bold')


plt.show()
