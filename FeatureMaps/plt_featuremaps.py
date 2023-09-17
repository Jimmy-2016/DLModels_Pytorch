
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from torchvision import models, transforms, utils
from PIL import Image

transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(0, 1)
]

)

img = Image.open(str('./1.jpg'))
plt.imshow(img)

## load the model

model = models.resnet18(pretrained=True)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)
print(model)
print(device)

## get the ccn layers by chidlren method, Always use children method

model_weights =[]
conv_layers = []

model_children = list(model.children())
counter = 0
for i in range(len(model_children)):
    if type(model_children[i]) == nn.Conv2d:
        counter += 1
        model_weights.append(model_children[i].weight)
        conv_layers.append(model_children[i])
    elif type(model_children[i]) == nn.Sequential:
        for j in range(len(model_children[i])):
            for child in model_children[i][j].children():
                if type(child) == nn.Conv2d:
                    counter += 1
                    model_weights.append(child.weight)
                    conv_layers.append(child)
print(f"Total convolution layers: {counter}")
print("conv_layers")

## Get the cnn layer by module method

# model_weights_m =[]
# conv_layers_m = []
# model_modules = list(model.modules())
#
#
# for module in model.modules():
#     if isinstance(module, nn.Conv2d):
#         model_weights_m.append(module.weight)
#         conv_layers_m.append(module)


##

image = transforms(img)
image = image[1:4, :]

print(f"Image shape before: {image.shape}")
image = image.unsqueeze(0)
print(f"Image shape after: {image.shape}")
image = image.to(device)

##
outputs = []
names = []
for layer in conv_layers:
    image = layer(image)
    outputs.append(image)
    names.append(str(layer))
print(len(outputs))
for feature_map in outputs:
    print(feature_map.shape)


##
processed = []
for feature_map in outputs:
    feature_map = feature_map.squeeze(0)
    gray_scale = torch.sum(feature_map,0)
    gray_scale = gray_scale / feature_map.shape[0]
    processed.append(gray_scale.data.cpu().numpy())
for fm in processed:
    print(fm.shape)

##
fig = plt.figure()
for i in range(len(processed)):
    a = fig.add_subplot(5, 4, i+1)
    imgplot = plt.imshow(processed[i])
    a.axis("off")
    a.set_title('Layer ' + str(i))
plt.savefig(str('feature_maps.jpg'), bbox_inches='tight')


plt.show()