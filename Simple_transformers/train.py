import matplotlib.pyplot as plt
import torch

from model import *

torch.random.manual_seed(0)



freq = np.random.uniform(0.1, .5, 20)
noise = 0.1
sig_len = 300

input, output = create_ds(sig_len, noise, freq)
# plot_random_samples(10, input, output)

