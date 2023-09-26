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

torch.manual_seed(1)
np.random.seed(1)


PATH = './saved_model/model1.pth'

z_dim = 4
model = VAE(CNNLayerEncoder=[10, 16],
                CNNLayerDecoder=[16, 10, 1],
                  z_dim=4,
                  stride=2,
                  filter_size=3,
                  pool=2,
                  paddign=2,
            num_targetN=8,
            conditional=1)
model.load_state_dict(torch.load(PATH))

num_example = 6
noise = torch.randn((num_example, z_dim))
condition_num = 1
label = condition_num * torch.ones(num_example).to(torch.int64)
z = noise
z = torch.cat((z, model.embedding(label)), dim=-1)
predict = model.decoder(model.fc(z))

for i in range(num_example):
    fig = plt.figure()
    plt.imshow(np.squeeze(predict[i, :, :].detach()))



##

plt.show()
