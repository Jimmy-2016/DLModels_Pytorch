import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from model import *
import torch.utils.data
import matplotlib.pyplot as plt
import torch.optim as optim
import numpy as np
from torch.distributions.multivariate_normal import MultivariateNormal

# torch.manual_seed(1)
# np.random.seed(1)


PATH = './saved_model/model1.pth'


z_dim = 5
model =  VAE(CNNLayerEncoder=[8, 16],
                CNNLayerDecoder=[16, 8, 1],
                  z_dim=5,
                  stride=2,
                  filter_size=3,
                  pool=2,
                  paddign=2,
            num_targetN=4,
            conditional=1)
model.load_state_dict(torch.load(PATH))

num_example = 10
# sigma = torch.randn((num_example, z_dim))
# mu = torch.randn((num_example, z_dim))
# epsilon = torch.randn_like(sigma)
# z = mu + sigma * epsilon
noise = torch.randn((num_example, z_dim))
z = noise
# model.embedding(0*torch.ones((num_example, 4).to(torch.int64)))
# z = torch.cat((z, 0*torch.ones((num_example, 10))), dim=-1)
z = torch.cat((z, model.embedding(5*torch.ones((num_example, 4)).to(torch.int64))[:, :, 0]), dim=-1)

predict = model.decoder(model.fc(z))


for i in range(num_example):
    fig = plt.figure()
    plt.imshow(np.squeeze(predict[i, :, :].detach()))


plt.colorbar()
plt.show()
