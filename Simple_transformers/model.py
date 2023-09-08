import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from torchsummary import summary


class MyTransformer_TS(nn.Module):
    def __init__(self, n_input, n_output, n_head, num_layers):
        super(MyTransformer_TS, self).__init__()

        self.embedding = nn.Linear(n_input, n_input)
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=n_input, nhead=n_head)
        self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=num_layers)
        self.fc = nn.Linear(n_input, n_output)

    def forward(self, x):
        x = self.embedding(x)
        x = self.transformer_encoder(x)
        x = self.fc(x)
        return x



def patch_attention(m):
    forward_orig = m.forward

    def wrap(*args, **kwargs):
        kwargs['need_weights'] = True
        kwargs['average_attn_weights'] = False

        return forward_orig(*args, **kwargs)

    m.forward = wrap



class SaveOutput:
    def __init__(self):
        self.outputs = []

    def __call__(self, module, module_in, module_out):
        self.outputs.append(module_out[1])

    def clear(self):
        self.outputs = []


def create_ds(siglen, noise_level, freqs):
    num_samples = len(freqs)
    input = np.zeros([num_samples, siglen])
    output = np.zeros_like(input)
    for i in range(num_samples):
        noise = np.random.normal(0, noise_level, siglen+1)
        tmpsig = np.sin(freqs[i] * np.arange(siglen + 1)) + noise
        input[i, :] = tmpsig[:-1]
        output[i, :] = tmpsig[1:]

    return input.astype(float), output.astype(float)


def plot_random_samples(numplots, input, output):

    plt.figure()
    plotindx = np.random.permutation(len(input))
    for i in range(numplots):
        plt.subplot(5, 2, i + 1)
        plt.plot(input[plotindx[i], :])
        plt.plot(output[plotindx[i], :])

    plt.show()





if __name__ == "__main__":
    model = MyTransformer_TS(n_input=300, n_output=1, n_head=3, num_layers=2)
    input = torch.rand((10, 300))
    print(model(input))
    print(model)
    print(model(input).shape)
    summary(model, (10, 300), batch_size=10)






