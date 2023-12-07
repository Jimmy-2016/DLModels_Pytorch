
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from vae_model import *
import torch.utils.data
import matplotlib.pyplot as plt
import torch.optim as optim

## Params
n_epochs = 10
batch_size_train = 128
batch_size_test = 10
log_interval = 2

lr = 0.001
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
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


def loss_fn(recon_x, x, mu, logvar):
    # BCE = F.binary_cross_entropy(recon_x, x, size_average=False)
    BCE = F.mse_loss(recon_x, x, size_average=False)

    # see Appendix B from VAE paper:
    # Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
    # KLD = 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    # logvar = torch.log((Sigma + 1e-8)**2)
    KLD = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())

    return BCE + KLD, BCE, KLD


# disp_example(train_loader, 5)

model = myVAE(layers=[784, 50, 40, 10]).to(device)
modellayers = list(model.children())[0]
model
criteria = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=lr)
train_loss = []
test_losses = []
for i in range(n_epochs):
   correct = 0

   for _, (tmpdata, tmptar) in enumerate(train_loader):

       # if Conditional:
       # tmpmat = tmptar.unsqueeze(0)*torch.ones(tmpdata.shape[-2], tmpdata.shape[-1]).unsqueeze(2)
       # tmpmat = torch.permute(tmpmat, (2, 0, 1)).unsqueeze(1)
       # input = torch.concat((tmpdata, tmpmat), dim=1)



       # re_const, mu, sigma = model(input, tmptar)
       input = tmpdata.view(tmpdata.shape[0], -1)
       re_const, mu, sigma, encoder_out, z = model(input)

       activations = []
       # layeractive = input
       # for li in modellayers:
       #     input = li(input)
       #     activations.append(input)


       optimizer.zero_grad()
       tmptar = F.one_hot(tmptar)
       loss, bce, kl = loss_fn(re_const, input, mu, sigma)
       loss.backward()
       optimizer.step()
       # correct += (predict.argmax(axis=1) == tmptar.argmax(axis=1)).sum()

   train_loss.append(loss.item())

   if i % log_interval == 0:
       print('Train Epoch: {} \tLoss: {:.6f}'.format(
           i,  loss.item()))

torch.save(model.state_dict(), './saved_model/model1.pth')
torch.save(optimizer.state_dict(), './saved_model/optimizer1.pth')


##
exmaple = enumerate(test_loader)
batch_index, (data, target) = next(exmaple)
# tmpmat = target.unsqueeze(0)*torch.ones(data.shape[-2], data.shape[-1]).unsqueeze(2)
# tmpmat = torch.permute(tmpmat, (2, 0, 1)).unsqueeze(1)
# input = torch.concat((data, tmpmat), dim=1)

input = data.squeeze(1).view(data.shape[0], -1)

predict = model(input)[0]

for i in range(6):
    fig = plt.figure()
    plt.imshow(predict[i, :].detach().view(28, 28))

for i in range(6):
    fig = plt.figure()
    plt.imshow(np.squeeze(data[i, :, :].detach()))


print('############## End #################')







plt.show()
