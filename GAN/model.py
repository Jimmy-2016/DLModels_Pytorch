
import torch
import torch.nn as nn
from torchsummary import summary


class CNNBlock(nn.Module):
    def __init__(self, num_in, num_filters, filter_size, stride, paddingsize, pool):
        super(CNNBlock, self).__init__()

        self.nonlinear = nn.ReLU()
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.BachNorm = nn.BatchNorm2d(num_filters)
        self.cnn = nn.Conv2d(in_channels=num_in,
                             out_channels=num_filters,
                         kernel_size=filter_size,
                         stride=stride,
                         padding=paddingsize)

    def forward(self, x):
        x = self.cnn(x)
        x = self.BachNorm(x)
        x = self.nonlinear(x)

        return self.maxpool(x)


class DCNNBlock(nn.Module):
    def __init__(self, num_in, num_filters, filter_size, stride, paddingsize):
        super(DCNNBlock, self).__init__()

        self.num_in = num_in
        self.num_filter = num_filters
        self.filter_size = filter_size
        self.stride = stride
        self.paddingsize = paddingsize
        self.nonlinear = nn.ReLU()
        self.BachNorm = nn.BatchNorm2d(self.num_filter)
        self.cnn = nn.ConvTranspose2d(in_channels=num_in,
                             out_channels=self.num_filter,
                             kernel_size=self.filter_size,
                             stride=self.stride,
                             padding=self.paddingsize)

    def forward(self, x):
        x = self.cnn(x)
        x = self.BachNorm(x)
        x = self.nonlinear(x)
        return x



class Dicriminator(nn.Module):
    def __init__(self, discLayer, filter_size, stride, paddign, pool, classnum, conditional=1):
        super(Dicriminator, self).__init__()

        self.disclayer = discLayer
        self.filter_size = filter_size
        self.stride = stride
        self.padding = paddign
        self.pool = nn.MaxPool2d(kernel_size=pool, stride=2)
        self.calssnum = classnum
        self.cnn = self._makelayer(self.disclayer)
        self.sigmoid = nn.Sigmoid()

    def _makelayer(self, layers):
        Layer = []
        for i in range(len(layers)-1):
            Layer.append(CNNBlock(num_in=layers[i],
                                  num_filters=layers[i+1],
                                  filter_size=self.filter_size,
                                  stride=self.stride,
                                  paddingsize=self.padding,
                                  pool=self.pool
                                  ))
        return nn.Sequential(*Layer)

    def forward(self, x):
        return self.sigmoid(self.cnn(x))


class Generator(nn.Module):
    def __init__(self, Genlayer, z_dim, filter_size, stride, paddign, pool, classnum, conditional=1):
        super(Generator, self).__init__()

        self.genlayer = Genlayer
        self.zdim = z_dim
        self.filter_size = filter_size
        self.stride = stride
        self.padding = paddign
        self.pool = nn.MaxPool2d(kernel_size=pool, stride=2)
        self.calssnum = classnum
        self.cnn = self._makelayer(self.genlayer)
        self.sigmoid = nn.Sigmoid()

    def _makelayer(self, layers):
        Layer = []
        for i in range(len(layers) - 1):
            Layer.append(DCNNBlock(num_in=layers[i],
                                  num_filters=layers[i + 1],
                                  filter_size=self.filter_size,
                                  stride=self.stride,
                                  paddingsize=self.padding,
                                  ))
        return nn.Sequential(*Layer)

    def forward(self, x):
        return self.sigmoid(self.cnn(x))

def initialize_weights(model):
    # Initializes weights according to the DCGAN paper
    for m in model.modules():
        if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d, nn.BatchNorm2d)):
            nn.init.normal_(m.weight.data, 0.0, 0.02)



if __name__ == "__main__":
    input_img = torch.rand(200, 1, 28, 28)
    noise_dim = 30
    disc_model = Dicriminator(discLayer=[1, 8, 16, 32],
                  stride=2,
                  filter_size=3,
                  pool=2,
                  paddign=2,
                  classnum=0,
                  conditional=0
    )
    print('disc shape = ' + str(disc_model(input_img).shape))
    z = torch.randn((20, noise_dim, 1, 1))
    gen_model = Generator(Genlayer=[noise_dim, 16, 8, 1],
                  z_dim=noise_dim,
                  stride=2,
                  filter_size=5,
                  pool=0,
                  paddign=0,
                  classnum=0,
                  conditional=0)

    gen_img = gen_model(z)
    print(gen_img.shape)

    summary(gen_model, (noise_dim, 1, 1), batch_size=-1)










