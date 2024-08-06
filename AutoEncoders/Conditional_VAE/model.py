
import torch
import torch.nn as nn
from torchsummary import summary
import torch.nn.functional as F


class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.shape[0], -1)
class UnFlatten(nn.Module):
    def __init__(self, nhid, emsize):
        super(UnFlatten, self).__init__()
        self.nhid = nhid
        self.emsize = emsize
    def forward(self, x):
        return x.view(x.shape[0], self.emsize[0], self.emsize[1], self.emsize[2])


class CNNBlock(nn.Module):
    def __init__(self, num_in, num_filters, filter_size, stride, paddingsize, pool):
        super(CNNBlock, self).__init__()

        self.num_in = num_in
        self.num_filter = num_filters
        self.filter_size = filter_size
        self.stride = stride
        self.pool = nn.MaxPool2d(kernel_size=pool, stride=2)
        self.paddingsize = paddingsize
        self.nonlinear = nn.ReLU()
        self.BachNorm = nn.BatchNorm2d(self.num_filter)
        self.cnn = nn.Conv2d(in_channels=num_in,
                             out_channels=self.num_filter,
                             kernel_size=self.filter_size,
                             stride=self.stride,
                             padding=self.paddingsize)

    def forward(self, x):
        x = self.cnn(x)
        x = self.BachNorm(x)
        x = self.nonlinear(x)
        x = self.pool(x)
        return x

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


class VAE(nn.Module):
    def __init__(self, CNNLayerEncoder, CNNLayerDecoder, z_dim, filter_size, stride, paddign, pool, num_targetN, conditional=1):
        super(VAE, self).__init__()

        self.CNNlayer_encoder = CNNLayerEncoder
        self.CNNlayer_decoder = CNNLayerDecoder
        self.stride = stride
        self.pool = pool
        self.padding = paddign
        self.filter_size = filter_size
        self.encoder_faltten = self._makeLayer_encoder(1, conditional=conditional)
        self.encoder_unfaltten = self._makeLayer_encoder(0, conditional=conditional)
        self.conditional = conditional
        self.num_targetN = num_targetN
        if self.conditional:
            data = torch.rand((60, 2, 28, 28))
        else:
            data = torch.rand((60, 1, 28, 28))
        self.nhid = self.encoder_faltten(data).shape[-1]
        self.out_shape = self.encoder_unfaltten(data).shape
        self.emsize = self.out_shape[1:]
        self.z_dim = z_dim
        self.embedding = nn.Embedding(10, num_targetN)
        self.decoder = self._makeLayer_decoder()
        self.fc_mean = nn.Linear(self.nhid, z_dim)
        self.fc_logvar = nn.Linear(self.nhid, z_dim)
        if self.conditional:
            self.fc = nn.Linear(z_dim+self.num_targetN, self.nhid)
        else:
            self.fc = nn.Linear(z_dim, self.nhid)


    def _makeLayer_encoder(self, flatten, conditional):
        layers = []
        if conditional:
            layers.append(CNNBlock(2,
                                   self.CNNlayer_encoder[0],
                                   filter_size=self.filter_size,
                                   stride=self.stride,
                                   paddingsize=self.padding, pool=self.pool)
                          )
        else:
            layers.append(CNNBlock(1,
                                   self.CNNlayer_encoder[0],
                                   filter_size=self.filter_size,
                                   stride=self.stride,
                                   paddingsize=self.padding, pool=self.pool)
                          )
        for i in range(len(self.CNNlayer_encoder) - 1):
            layers.append(CNNBlock(self.CNNlayer_encoder[i],
                                   self.CNNlayer_encoder[i + 1],
                                   filter_size=self.filter_size,
                                   stride=self.stride,
                                   paddingsize=self.padding,
                                   pool=self.pool)
                          )
        if flatten:
            layers.append(Flatten())

        return nn.Sequential(*layers)

    def _makeLayer_decoder(self):
        layers = []
        layers.append(UnFlatten(nhid=self.nhid, emsize=self.emsize))
        layers.append(DCNNBlock(self.out_shape[1],
                               self.CNNlayer_decoder[0],
                               filter_size=6,
                               stride=1,
                               paddingsize=1)
                      )
        for i in range(len(self.CNNlayer_decoder) - 1):
            layers.append(DCNNBlock(self.CNNlayer_decoder[i],
                                   self.CNNlayer_decoder[i + 1],
                                   filter_size=6,
                                   stride=2,
                                   paddingsize=1,
                                    )
                          )
            layers.append(nn.Tanh())
        return nn.Sequential(*layers)



    def forward(self, input, label=None):

        x = input
        x = self.encoder_faltten(x)
        mu, logvar = self.fc_mean(x), self.fc_logvar(x)
        sigma = torch.exp(0.5 * logvar)
        epsilon = torch.randn_like(logvar)
        z = mu + sigma * epsilon
        if self.conditional:
            label = self.embedding(label)
            z = torch.cat((z, label), dim=-1)
        z = self.fc(z)
        return self.decoder(z), mu, logvar


if __name__ == '__main__':
    input = torch.rand((20, 1, 28, 28))
    Layers = [8, 16]
    model = VAE(CNNLayerEncoder=Layers,
                CNNLayerDecoder=[16, 8, 1],
                  z_dim=5,
                  stride=2,
                  filter_size=3,
                  pool=2,
                  paddign=2,
                num_targetN=5,
                conditional=0)

    print(model(input, 1))
    print(model)
    summary(model, (2, 28, 28), batch_size=-1)
    print(model(input, 1)[0].shape)



