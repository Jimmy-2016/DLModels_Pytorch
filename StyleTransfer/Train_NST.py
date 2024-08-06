
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transform
from PIL import Image
from model import *
import matplotlib.pyplot as plt




def load_image(image_name):
    image = Image.open(image_name)
    return loader(image).to(device).unsqueeze(0)

device = torch.device('cude' if torch.cuda.is_available() else 'cpu')
print(device)
imsize = 64


loader = transform.Compose(
    [
        transform.Resize((imsize, imsize)),
        transform.ToTensor(),
        # transform.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),

    ]
)

original_img = load_image("org.png")
style_img = load_image("style.jpeg")

plt.figure()
plt.imshow(original_img.detach().squeeze(0)[0, :])

plt.figure()
plt.imshow(style_img.detach().squeeze(0)[0, :])


gen_image = original_img.clone().requires_grad_(True)
model = VGG().to(device).eval()  # to prevent from training

# Hyperparameters
total_steps = 1000
learning_rate = 0.001
alpha = 1
beta = 1
optimizer = torch.optim.Adam([gen_image], lr=learning_rate)

for steps in range(total_steps):
    gen_features = model(gen_image)
    org_feature = model(original_img)
    style_feature = model(style_img)
    org_loss = 0
    style_loss = 0
    for gen_feat, org_feat, st_feat in zip(
        gen_features, org_feature, style_feature
    ):
        org_loss += torch.mean(gen_feat - org_feat)**2
        batch, channel, w, h = gen_feat.shape
        G = gen_feat.view(channel, w * h).mm(gen_feat.view(channel, w * h).t())
        A = st_feat.view(channel, w * h).mm(st_feat.view(channel, w *h).t())
        style_loss += torch.mean(A - G)**2
    total_loss = alpha * org_loss + beta * style_loss
    optimizer.zero_grad()
    total_loss.backward()
    optimizer.step()

    if steps % 10 == 0:
        print(total_loss)



plt.figure()
plt.imshow(gen_image.detach().squeeze(0)[0, :])



plt.show()














