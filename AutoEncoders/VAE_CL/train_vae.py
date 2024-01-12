
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from vae_model import *
import torch.utils.data
import matplotlib.pyplot as plt
import torch.optim as optim

## Params
torch.manual_seed(1)
np.random.seed(1)


n_epochs = 10
batch_size_train = 64
batch_size_test = 10
log_interval = 2

lr = 0.001
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

Conditional = True
Contrast = False


# if Contrast == True:
#     Conditional = False


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


def loss_fn(recon_x, x, mu, sigma):
    # BCE = F.binary_cross_entropy(recon_x, x, size_average=False)
    BCE = F.mse_loss(recon_x, x, size_average=False)

    # see Appendix B from VAE paper:

    KLD = 0.5 * torch.sum(mu.pow(2) + sigma.pow(2) - torch.log(sigma.pow(2)) - 1).clamp(1000)


    return BCE + KLD, BCE, KLD


# disp_example(train_loader, 5)

model = myVAE(layers=[784, 100, 50, 10], conditional=Conditional).to(device)
modellayers = list(model.children())[0]
model
criteria = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=lr)
train_loss = []
test_losses = []
for i in range(n_epochs):
   correct = 0

   for _, (tmpdata, tmptar) in enumerate(train_loader):
       input = tmpdata.view(tmpdata.shape[0], -1)
       re_const, mu, sigma, encoder_out, z = model(input, tmptar)

       if Contrast:
           targetdist = torch.zeros((len(tmptar), len(tmptar)))
           for tri in range(10):
               indx = torch.where(tmptar == tri)[0]
               targetdist[np.ix_(indx, indx)] = 1

           targetdist = targetdist.view(-1)

       activations = []

       optimizer.zero_grad()
       # tmptar = F.one_hot(tmptar)
       loss, bce, kl = loss_fn(re_const, input, mu, torch.log(torch.pow(sigma + 1e-10, 2)))

       contrast_loss = 0
       if Contrast:
           dist = torch.cdist(mu, mu).view(-1)

           contrast_loss = (1 - targetdist) * torch.pow(dist, 2) \
                  + (targetdist) * torch.pow(torch.clamp(1 - dist, min=0.0), 2)
           contrast_loss = torch.mean(contrast_loss)

           # contrastive_loss(dist, targetdist, margin=2)
           loss += contrast_loss

       loss.backward()
       optimizer.step()

   train_loss.append(loss.item())

   if i % log_interval == 0:
       print('Train Epoch: {} \tLoss: {:.6f}'.format(
           i,  loss.item()))

if Conditional:
    torch.save(model.state_dict(), './saved_model/con_model1.pth')
elif Contrast:
    torch.save(model.state_dict(), './saved_model/Contrast_model1.pth')
else:
    torch.save(model.state_dict(), './saved_model/nocon_model1.pth')

if Contrast and Conditional:
    torch.save(model.state_dict(), './saved_model/concontrast_model1.pth')

torch.save(optimizer.state_dict(), './saved_model/optimizer.pth')


##
exmaple = enumerate(test_loader)
batch_index, (data, target) = next(exmaple)

input = data.squeeze(1).view(data.shape[0], -1)

predict = model(input, target)[0]

for i in range(6):
    fig = plt.figure()
    plt.imshow(predict[i, :].detach().view(28, 28))

for i in range(6):
    fig = plt.figure()
    plt.imshow(np.squeeze(data[i, :, :].detach()))


print('############## End #################')







plt.show()
