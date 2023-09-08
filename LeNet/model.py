
import torch
import torch.nn as nn
from torchsummary import summary


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


class LeNet(nn.Module):
    def __init__(self, CNNLayer, DenseLayer, filter_size, stride, paddign, pool):
        super(LeNet, self).__init__()

        self.CNNlayer = CNNLayer
        self.DensLayer = DenseLayer
        self.stride = stride
        self.pool = pool
        self.padding = paddign
        self.filter_size = filter_size
        self.cnn = self._makeLayer_cnn()

        data = torch.rand((60, 1, 28, 28))
        batch_size = data.shape[0]
        self.out_shape = self.cnn(data).view(batch_size, -1).shape[-1]

        self.linear = self._make_layer_dens()



    def _makeLayer_cnn(self):
        layers = []
        layers.append(CNNBlock(1,
                               self.CNNlayer[0],
                               filter_size=self.filter_size,
                               stride=self.stride,
                               paddingsize=self.padding, pool=self.pool)
                      )
        for i in range(len(self.CNNlayer)-1):
            layers.append(CNNBlock(self.CNNlayer[i],
                                   self.CNNlayer[i+1],
                                   filter_size=self.filter_size,
                                   stride=self.stride,
                                   paddingsize=self.padding,
                                   pool=self.pool)
                          )

        return nn.Sequential(*layers)

    def _make_layer_dens(self):
        layers = []
        layers.append(nn.Sequential(
            nn.Linear(self.out_shape, self.DensLayer[0]),
            nn.BatchNorm1d(self.DensLayer[0]),
            nn.ReLU()
        ))
        for i in range(len(self.DensLayer)-1):
            layers.append(nn.Sequential(
                nn.Linear(self.DensLayer[i], self.DensLayer[i+1]),
                nn.BatchNorm1d(self.DensLayer[i+1]),
                nn.ReLU()
            ))
        layers.append(nn.Sequential(
            nn.Linear(self.DensLayer[-1], 10),
            nn.Sigmoid()
        ))
        return nn.Sequential(*layers)


    def forward(self, x):
        x = self.cnn(x).view(x.shape[0], -1)
        x = self.linear(x)

        return x


if __name__ == "__main__":
    input = torch.rand((60, 1, 28, 28))
    model = LeNet(CNNLayer=[8, 16, 32],
                  DenseLayer=[80, 20],
                  stride=2,
                  filter_size=3,
                  pool=2,
                  paddign=2)

    print(model(input))
    print(model)
    summary(model, (1, 28, 28), batch_size=-1)


