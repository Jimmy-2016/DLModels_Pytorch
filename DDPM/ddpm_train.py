
from utils import *

no_train = False
fashion = False
batch_size = 128
n_epochs = 20
lr = 0.001
store_path = "ddpm_fashion.pt" if fashion else "ddpm_mnist.pt"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(device)
##

train_loader = torch.utils.data.DataLoader(
    torchvision.datasets.MNIST('./data/', train=True, download=True,
                               transform=torchvision.transforms.Compose([
                                   torchvision.transforms.ToTensor(),
                                   Lambda(lambda x: (x - 0.5) * 2),
                                   # torchvision.transforms.Normalize((0.1307,), (0.3081,))
                               ])


    ),
    batch_size=batch_size, shuffle=True
)

exmaple = enumerate(train_loader)
batch_index, (img, target) = next(exmaple)

show_images(img)
