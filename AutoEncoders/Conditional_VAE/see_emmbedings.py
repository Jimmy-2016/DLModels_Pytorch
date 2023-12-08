

from model import *
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


PATH = './saved_model/model_contrast.pth'
Conditional = 0

z_dim = 4
model = VAE(CNNLayerEncoder=[10, 16],
                CNNLayerDecoder=[16, 10, 1],
                  z_dim=4,
                  stride=2,
                  filter_size=3,
                  pool=2,
                  paddign=2,
            num_targetN=8,
            conditional=Conditional)
model.load_state_dict(torch.load(PATH))


exmaple = enumerate(test_loader)
batch_index, (data, target) = next(exmaple)

if Conditional:
    tmpmat = target.unsqueeze(0) * torch.ones(data.shape[-2], data.shape[-1]).unsqueeze(2)
    tmpmat = torch.permute(tmpmat, (2, 0, 1)).unsqueeze(1)
    input = torch.concat((data, tmpmat), dim=1)
    re_const, mu, sigma = model(input, target)

else:
    input = data
    # input = tmpdata
    # input = torch.concat((tmpdata, tmptar * torch.ones_like(tmpdata)), dim=1)
    re_const, mu, sigma = model(input)


# tmpmat = target.unsqueeze(0)*torch.ones(data.shape[-2], data.shape[-1]).unsqueeze(2)
# tmpmat = torch.permute(tmpmat, (2, 0, 1)).unsqueeze(1)
# input = torch.concat((data, tmpmat), dim=1)
#
# latenlayer = model.encoder_unfaltten(input)[0]
# out, mu, sigma = model(input, target)



# plot_exmaples(data, model(data.view(data.shape[0], -1)))

print('############## End #################')


##

umap_hparams = {'n_neighbors': 5,
                'min_dist': 0.1,
                'n_components': 2,
                'metric': 'euclidean'}


fig, ax = plt.subplots(constrained_layout=False)
# ax.set(xticks=[], yticks=[])
ax.set_xlim([-10, 20])
ax.set_ylim([-10, 20])

umap_embedding = umap.UMAP(n_neighbors=umap_hparams['n_neighbors'], min_dist=umap_hparams['min_dist'],
                           n_components=umap_hparams['n_components'],
                           metric=umap_hparams['metric']).fit_transform(mu.detach().numpy())
scatter = ax.scatter(x=umap_embedding[:, 0], y=umap_embedding[:, 1], s=20, c=target, cmap='tab10')

cbar = plt.colorbar(scatter, boundaries=np.arange(11)-0.5)
cbar.set_ticks(np.arange(10))
cbar.set_ticklabels(np.arange(10))

plt.title('UMAP Dimensionality reduction', fontsize=25, fontweight='bold')


plt.show()
