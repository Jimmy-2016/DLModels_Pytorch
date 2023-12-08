
import numpy as np
import torch
import torch.nn as nn
from torchsummary import summary




class fc_layer(nn.Module):
    def __init__(self, num_in, num_out):
        super().__init__()

        self.fc = nn.Sequential(
            nn.Linear(num_in, num_out),
            # nn.BatchNorm1d(num_out),
            nn.ReLU()
        )

    def forward(self, x):

        return self.fc(x)





class myVAE(nn.Module):
    def __init__(self, layers, conditional=True):
        super(myVAE, self).__init__()
        self.embdedim = 5
        self.layers = layers[:-1]
        self.latent = layers[-1]
        self.reverse_layers = layers[::-1][1:]
        self.encoder = self._makelayers(encoder=True, conditonal=conditional)
        self.decoder = self._makelayers(encoder=False)
        self.fc_mean = fc_layer(self.layers[-1], self.latent)
        self.fc_sigma = fc_layer(self.layers[-1], self.latent)
        if conditional:
            self.fc_decoder = fc_layer(self.latent + self.embdedim, self.reverse_layers[0])
        else:
            self.fc_decoder = fc_layer(self.latent, self.reverse_layers[0])

        self.condition = conditional
        self.embeding = nn.Embedding(10, self.embdedim, max_norm=True)


    def _makelayers(self, encoder= True, conditonal=False):
        layers = []
        if encoder:
            if conditonal == True:
                layers.append(fc_layer(self.layers[0] + self.embdedim, self.layers[1]))
            else:
                layers.append(fc_layer(self.layers[0], self.layers[1]))
        else:
            if conditonal == True:
                layers.append(fc_layer(self.reverse_layers[0] + self.embdedim, self.reverse_layers[1]))
            else:
                layers.append(fc_layer(self.reverse_layers[0], self.reverse_layers[1]))


        if encoder:
            for li in range(1, len(self.layers)-1):  # Encoder
                layers.append(fc_layer(self.layers[li], self.layers[li+1]))
        else:
            for li in range(1, len(self.reverse_layers)-2):  # Decoder
                layers.append(fc_layer(self.reverse_layers[li], self.reverse_layers[li+1]))

            layers.append(nn.Sequential(nn.Linear(self.reverse_layers[-2], self.reverse_layers[-1]),
                                        nn.LeakyReLU()))

        return nn.Sequential(*layers)

    def forward(self, x, targets):
        if self.condition:
            x = torch.cat((x, self.embeding(targets)), dim=1)

        encoder_out = self.encoder(x)
        mu = self.fc_mean(encoder_out)
        sigma = self.fc_sigma(encoder_out)
        noise = torch.rand_like(sigma)
        z = mu + sigma * noise
        if self.condition:
            labels = self.embeding(targets)
            z = torch.cat((z, labels), dim=1)

        y = self.fc_decoder(z)
        return self.decoder(y), mu, sigma, encoder_out, z


        # return self.fc(x)


# class contrastive_loss(nn.Module):
#     def __init__(self, margin=2):
#         super(contrastive_loss, self).__init__()
#         self.margin = margin
#
#     def forward(self, dist, label):
#         # dist = torch.nn.functional.pairwise_distance(x1, x2)(x1, x2)
#
#
#         return loss
#
# # def criterion(x1, x2, label, margin: float = 2.0):





if __name__ == "__main__":
    layers = [784, 50, 40, 10]
    model = myVAE(layers=layers)
    print(model(torch.rand((10, 784)), torch.randint(0, 9, size=(10,))))
    print(model)
    # summary(model, (0, 784), batch_size=-1)








