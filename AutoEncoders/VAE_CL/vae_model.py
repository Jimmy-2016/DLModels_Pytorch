
import numpy as np
import torch
import torch.nn as nn
from torchsummary import summary




class fc_layer(nn.Module):
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





class myVAE(nn.Module):
    def __init__(self, layers):
        super(myVAE, self).__init__()

        self.layers = layers[:-1]
        self.latent = layers[-1]
        self.reverse_layers = layers[::-1][1:]
        self.encoder = self._makelayers(encoder=True)
        self.decoder = self._makelayers(encoder=False)
        self.fc_mean = fc_layer(self.layers[-1], self.latent, 0)
        self.fc_std = fc_layer(self.layers[-1], self.latent, 0)
        self.fc_decoder = fc_layer(self.latent, self.reverse_layers[0], 0)




    def _makelayers(self, encoder= True):
        layers = []
        if encoder:
            for li in range(len(self.layers)-1):  # Encoder
                layers.append(fc_layer(self.layers[li], self.layers[li+1], 0))
        else:
            for li in range(len(self.reverse_layers)-1):  # Decoder
                layers.append(fc_layer(self.reverse_layers[li], self.reverse_layers[li+1], 0))

        # layers.append(fc_layer(self.reverse_layers[-2], self.reverse_layers[-1], 0))

        return nn.Sequential(*layers)

    def forward(self, x):

        encoder_out = self.encoder(x)
        mu = self.fc_mean(encoder_out)
        sigma = self.fc_std(encoder_out)
        noise = torch.rand_like(sigma)
        z = mu + sigma * noise
        y = self.fc_decoder(z)
        return self.decoder(y), mu, sigma, encoder_out, z


        # return self.fc(x)



if __name__ == "__main__":
    layers = [784, 50, 40, 10]
    model = myVAE(layers=layers)
    print(model(torch.rand((10, 784))))
    print(model)
    summary(model, (0, 784), batch_size=-1)








