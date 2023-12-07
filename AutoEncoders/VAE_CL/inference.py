import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from vae_model import *
import torch.utils.data
import matplotlib.pyplot as plt
import torch.optim as optim
import numpy as np
from torch.distributions.multivariate_normal import MultivariateNormal

torch.manual_seed(1)
np.random.seed(1)


Conditional = True

if Conditional:
    PATH = './saved_model/con_model.pth'
else:
    PATH = './saved_model/nocon_model.pth'

model = myVAE(layers=[784, 100, 50, 10], conditional=Conditional)

model.load_state_dict(torch.load(PATH))

num_example = 6
noise = torch.randn((num_example, 10))
condition_num = 7
label = condition_num * torch.ones(num_example).to(torch.int64)
z = noise
z = torch.cat((z, model.embeding(label)), dim=-1)
predict = model.decoder(model.fc_decoder(z))

for i in range(num_example):
    fig = plt.figure()
    plt.imshow(np.squeeze(predict[i, :].view(28, 28).detach()))



##

plt.show()
