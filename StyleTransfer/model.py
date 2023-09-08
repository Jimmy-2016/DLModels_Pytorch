
import torch
import torch.nn as nn
import torchvision.models as models

class VGG(nn.Module):
    def __init__(self):
        super(VGG, self).__init__()

        self.chosen_feature = ["19", "28"]
        self.model = models.vgg19(pretrained=True).features[:29]

    def forward(self, x):
        features = []
        for layernum, layer in enumerate(self.model):
            x = layer(x)
            if str(layernum) in self.chosen_feature:
                features.append(x)

        return features

