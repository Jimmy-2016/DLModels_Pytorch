import numpy as np
import torch
import torch.nn as nn
from torchsummary import summary

# this is added to the github

class block(nn.Module):
    def __init__(self, num_in, num_out, ifend):
        super().__init__()
        self.ifend = ifend

        self.fc = nn.Sequential(
            nn.Linear(num_in, num_out),
            nn.BatchNorm1d(num_out),
            nn.ReLU()
        )
        self.fc_end = nn.Sequential(
            nn.Linear(num_in, num_out),
            nn.BatchNorm1d(num_out),
            nn.Sigmoid()
        )

    def forward(self, x):
        if self.ifend:
            return self.fc_end(x)
        else:
            return self.fc(x)



class MLP(nn.Module):
    def __init__(self, num_in, num_out, num_hid):
        super(MLP, self).__init__()

        self.num_in = num_in
        self.num_out = num_out
        self.num_hid = num_hid
        self.fc_input = block(self.num_in, self.num_hid[0], 0)
        self.fc_hid = self._make_layer()
        self.fc_out = block(self.num_hid[-1], self.num_out, 1)
        self.nonlinear = nn.ReLU()

    def _make_layer(self):
        layers = []
        for i in range(len(self.num_hid)-1):
            layers.append(block(self.num_hid[i], self.num_hid[i+1], 0))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.fc_input(x)
        x = self.fc_hid(x)
        x = self.fc_out(x)
        return x


if __name__ == "__main__":
    model = MLP(num_in=784, num_hid=[50, 40, 20], num_out=10)
    print(model(torch.rand((10, 784))))
    print(model)
    summary(model, (0, 784), batch_size=-1)



