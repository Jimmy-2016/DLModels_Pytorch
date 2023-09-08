import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from model import *
import torch.utils.data
import matplotlib.pyplot as plt
import torch.optim as optim

## Params
n_epochs = 2
batch_size_train = 128
batch_size_test = 6
log_interval = 1
Conditional = 1
lr = 0.001
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
noise_dim = 200

print(device)



def disp_example(loader, indx):
    exmaple = enumerate(loader)
    batch_index, (data, target) = next(exmaple)

    print(data.shape)

    plt.imshow(data[indx, :, :, :].squeeze())

    print(target.shape)
    print(target[indx])


def plot_exmaples(exmaple, output):
    fig = plt.figure()

    for i in range(6):
      plt.subplot(2, 3, i+1)
      plt.tight_layout()
      plt.imshow(exmaple[i][0], cmap='gray', interpolation='none')
      plt.title("Prediction: {}".format(
        output.data.max(1, keepdim=True)[1][i].item()))
      plt.xticks([])
      plt.yticks([])

## Dataset
train_loader = torch.utils.data.DataLoader(
    torchvision.datasets.MNIST('./data/', train=True, download=True,
                               transform=torchvision.transforms.Compose([
                                   torchvision.transforms.ToTensor(),
                                   torchvision.transforms.Normalize((0.1307,), (0.3081,))
                               ])


    ),
    batch_size=batch_size_train, shuffle=True
)

test_loader = torch.utils.data.DataLoader(
    torchvision.datasets.MNIST('./data/', train=False, download=True,
                               transform=torchvision.transforms.Compose([
                                   torchvision.transforms.ToTensor(),
                                   torchvision.transforms.Normalize((0.1307,), (0.3081,))
                               ])


    ),
    batch_size=batch_size_test, shuffle=True
)


# disp_example(train_loader, 5)
disc_model = Dicriminator(discLayer=[1, 8, 16, 32],
                  stride=2,
                  filter_size=3,
                  pool=2,
                  paddign=2,
                  classnum=0,
                  conditional=0
    )
gen_model = Generator(Genlayer=[noise_dim, 16, 8, 1],
              z_dim=noise_dim,
              stride=2,
              filter_size=5,
              pool=0,
              paddign=0,
              classnum=0,
              conditional=0)

initialize_weights(gen_model)
initialize_weights(disc_model)




criteria = nn.BCELoss()
opt_disc = torch.optim.AdamW(disc_model.parameters(), lr=0.02)
opt_gen = torch.optim.AdamW(gen_model.parameters(), lr=0.001)

train_loss_gen = []
train_loss_disc = []

test_losses = []
for i in range(n_epochs):
   correct = 0

   for _, (tmpdata, tmptar) in enumerate(train_loader):
       real = F.sigmoid(tmpdata)
       noise = torch.rand((batch_size_train, noise_dim, 1, 1))
       fake = gen_model(noise)[:, :, :28, :28]

       ### Train Discriminator: max log(D(x)) + log(1 - D(G(z)))
       disc_real = disc_model(real).reshape(-1)
       disc_real_loss = criteria(disc_real, torch.ones_like(disc_real))
       disc_fake = disc_model(fake.detach()).reshape(-1)
       disc_fake_loss = criteria(disc_fake, torch.zeros_like(disc_fake))
       disc_loss = disc_fake_loss + disc_real_loss

       opt_disc.zero_grad()
       disc_loss.backward()
       opt_disc.step()

       ### Train Generator: min log(1 - D(G(z))) <-> max log(D(G(z))
       output = disc_model(fake).reshape(-1)
       gen_loss = criteria(output, torch.ones_like(output))

       opt_gen.zero_grad()
       gen_loss.backward()
       opt_gen.step()



       # correct += (predict.argmax(axis=1) == tmptar.argmax(axis=1)).sum()

   train_loss_disc.append(disc_loss.item())
   train_loss_gen.append(gen_loss.item())


   if i % log_interval == 0:
       print('Train Epoch: {} \tDiscLoss: {:.6f}'.format(
           i,  disc_loss.item()))
       print('Train Epoch: {} \tGenLoss: {:.6f}'.format(
           i, gen_loss.item()))

torch.save(gen_model.state_dict(), './saved_model/Genmodel.pth')
torch.save(gen_model.state_dict(), './saved_model/Genoptimizer.pth')

torch.save(disc_model.state_dict(), './saved_model/Dismodel.pth')
torch.save(disc_model.state_dict(), './saved_model/Discoptimizer.pth')


##
exmaple = enumerate(test_loader)
batch_index, (data, target) = next(exmaple)
# if Conditional:
#     tmpmat = target.unsqueeze(0) * torch.ones(data.shape[-2], data.shape[-1]).unsqueeze(2)
#     tmpmat = torch.permute(tmpmat, (2, 0, 1)).unsqueeze(1)
#     input = torch.concat((data, tmpmat), dim=1)
# else:
#     input = data
z = torch.randn((20, noise_dim, 1, 1))
predict = gen_model(z)

for i in range(6):
    fig = plt.figure()
    plt.imshow(np.squeeze(predict[i, :, :].detach()))

# for i in range(6):
#     fig = plt.figure()
#     plt.imshow(np.squeeze(data[i, :, :].detach()))

# plot_exmaples(data, model(data.view(data.shape[0], -1)))

print('############## End #################')







plt.show()
