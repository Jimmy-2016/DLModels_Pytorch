
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from model import *
import torch.utils.data
import matplotlib.pyplot as plt
import torch.optim as optim

## Params
n_epochs = 5
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


# disp_example(train_loader, 5)

model = MLP(num_in=784, num_hid=[50, 30, 20], num_out=10).to(device)
model
criteria = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=lr)
train_loss = []
test_losses = []
for i in range(n_epochs):
   correct = 0

   for _, (tmpdata, tmptar) in enumerate(train_loader):
       predict = model(tmpdata.view(tmpdata.shape[0], -1))

       optimizer.zero_grad()
       tmptar = F.one_hot(tmptar)
       loss = criteria(predict, tmptar.float())
       loss.backward()
       optimizer.step()
       correct += (predict.argmax(axis=1) == tmptar.argmax(axis=1)).sum()

   train_loss.append(loss.item())

   if i % log_interval == 0:
       print('Train Epoch: {} \tLoss: {:.6f}'.format(
           i,  loss.item()))
       print(100. * correct / len(train_loader.dataset))

torch.save(model.state_dict(), './saved_model/model.pth')
torch.save(optimizer.state_dict(), './saved_model/optimizer.pth')


##
exmaple = enumerate(test_loader)
batch_index, (data, target) = next(exmaple)
predict = model(data.view(data.shape[0], -1)).argmax(1)

for i in range(6):
    fig = plt.figure()
    plt.imshow(data[i, :].reshape(28, 28))
    plt.title(str(predict[i].item()))

# plot_exmaples(data, model(data.view(data.shape[0], -1)))

print('############## End #################')







plt.show()
