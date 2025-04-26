# coding: utf-8

import torch
import torchvision
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF

import matplotlib.pyplot as plt

from torch.utils.data import Dataset
from torch.utils.data import DataLoader

import os
import numpy as np
import random

from transformers.utils import logging
logging.set_verbosity_error()

from transformers import pipeline
from PIL import Image

# transform
transform = transforms.Compose([
    transforms.ToTensor()
])

# dataset
trainset = torchvision.datasets.FashionMNIST(
    root="./data",
    train=True,
    download=True,
    transform=transform
)

plt.imshow(trainset.data[0], cmap="gray")
plt.show()

plt.hist(trainset.data[0].reshape(-1))
plt.show()

class MyDataset(Dataset):

    def __init__(self, dataset, threshold=50):
        self.dataset = dataset
        self.threshold = threshold

    def __getitem__(self, index):
        image, label = self.dataset[index]
        mask = (image > self.threshold/255.0).float()

        return image, mask

    def __len__(self):
        return len(self.dataset)

mydataset = MyDataset(trainset)

image, mask = mydataset[0]

plt.subplot(1,2,1)
plt.imshow(image.squeeze(), cmap="gray")
plt.subplot(1,2,2)
plt.imshow(mask.squeeze(), cmap="gray")
plt.show()

# dataset & dataloader
trainloader = DataLoader(mydataset, batch_size=32, shuffle=True)

# network
import torch.nn as nn

class MyNet(nn.Module):

    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, padding=1)
        self.conv2 = nn.Conv2d(32, 32, 3, padding=1)
        self.conv3 = nn.Conv2d(32, 1, 1)

    def forward(self, x):
        x = self.conv1(x)
        x = torch.relu(x)
        x = self.conv2(x)
        x = torch.relu(x)
        x = self.conv3(x)
        x = torch.sigmoid(x)
        return x

net = MyNet()

# loss
loss_fn = nn.BCELoss()

# optimizer
optimizer = torch.optim.Adam(net.parameters(), lr=0.001)

# training
for epoch in range(5):

    running_loss = 0.0

    for images, masks in trainloader:

        optimizer.zero_grad()

        outputs = net(images)
        loss = loss_fn(outputs, masks)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    print(f"epoch{epoch+1}, loss: {running_loss/len(trainloader)}")

print("Finished Training")

# inference
image, mask = mydataset[0]

output = net(image.unsqueeze(0))

plt.subplot(1,3,1)
plt.imshow(image.squeeze(), cmap="gray")
plt.subplot(1,3,2)
plt.imshow(mask.squeeze(), cmap="gray")
plt.subplot(1,3,3)
plt.imshow(output.detach().squeeze(), cmap="gray")
plt.show()

# SAM Mask generation
sam_pipe = pipeline("mask-generation", "./models/Zigeng/SlimSAM-uniform-77")
raw_image = Image.open('meta_llamas.jpg')
raw_image.resize((720, 375))
output = sam_pipe(raw_image, points_per_batch=32)
from helper import show_pipe_masks_on_image
show_pipe_masks_on_image(raw_image, output)

# Faster Inference
from transformers import SamModel, SamProcessor
model = SamModel.from_pretrained("./models/Zigeng/SlimSAM-uniform-77")
processor = SamProcessor.from_pretrained("./models/Zigeng/SlimSAM-uniform-77")
raw_image.resize((720, 375))
input_points = [[[1600, 700]]]
inputs = processor(raw_image, input_points=input_points, return_tensors="pt")

with torch.no_grad():
    outputs = model(**inputs)

predicted_masks = processor.image_processor.post_process_masks(
    outputs.pred_masks,
    inputs["original_sizes"],
    inputs["reshaped_input_sizes"]
)

predicted_mask = predicted_masks[0]
from helper import show_mask_on_image
for i in range(3):
    show_mask_on_image(raw_image, predicted_mask[:, i])

# Depth Estimation with DPT
depth_estimator = pipeline(task="depth-estimation", model="./models/Intel/dpt-hybrid-midas")
raw_image = Image.open('gradio_tamagochi_vienna.png')
raw_image.resize((806, 621))
output = depth_estimator(raw_image)

prediction = torch.nn.functional.interpolate(
    output["predicted_depth"].unsqueeze(1),
    size=raw_image.size[::-1],
    mode="bicubic",
    align_corners=False,
)

import numpy as np
output = prediction.squeeze().numpy()
formatted = (output * 255 / np.max(output)).astype("uint8")
depth = Image.fromarray(formatted)
depth

# Gradio demo
import gradio as gr

def launch(input_image):
    out = depth_estimator(input_image)
    prediction = torch.nn.functional.interpolate(
        out["predicted_depth"].unsqueeze(1),
        size=input_image.size[::-1],
        mode="bicubic",
        align_corners=False,
    )
    output = prediction.squeeze().numpy()
    formatted = (output * 255 / np.max(output)).astype("uint8")
    depth = Image.fromarray(formatted)
    return depth

iface = gr.Interface(launch, inputs=gr.Image(type='pil'), outputs=gr.Image(type='pil'))
iface.launch(share=True, server_port=int(os.environ['PORT1']))
iface.close()
