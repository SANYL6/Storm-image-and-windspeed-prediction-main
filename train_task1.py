import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.autograd as autograd
from torchvision import transforms
from torch.autograd import Variable
from torchvision.utils import save_image
import torchvision.transforms as transforms
from torchvision.transforms import ToTensor
from torch.utils.data import TensorDataset, DataLoader
from torch.utils.data import DataLoader, BatchSampler, Dataset
from skimage.metrics import structural_similarity as ssim
# from pytorch_ssim._init_ import ssim, ms_ssim, SSIM, MS_SSIM
from storm_tasks.task1.datasets import StormDataset
from storm_tasks.task1.network import Seq2SeqAutoencoder
dtype = torch.cuda.FloatTensor
from storm_tasks.task1.train_process import train,evaluate
import os
from PIL import Image, UnidentifiedImageError
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import json
import random

# prepare cpu
device = 'cpu'
if torch.cuda.device_count() > 0 and torch.cuda.is_available():
    print("Cuda installed! Running on GPU!")
    device = 'cuda'
else:
    print("No GPU available!")

# load datasets
dataset_root = r"C:\Users\ThinkStation\Desktop\Tropical-Storm-Image-and-Wind-Speed-Prediction-Tool-main\Tropical-Storm-Image-and-Wind-Speed-Prediction-Tool-main\Tropical-Storm-Image-and-Wind-Speed-Prediction-Tool-main\storm"
dataset = StormDataset(root=dataset_root, transform=True, sequence_length=6, stride=1)

# display datasets
num_items_to_check = 3
for i in range(num_items_to_check):
    images, target_image = dataset[i]
    print(f"Sample {i}:")
    print(f"- Sequence Length: {len(images)}")
    print(f"- Shape of Each Image in Sequence: {images[0].shape}")
    print(f"- Shape of Target Image: {target_image.shape}")

    # Optionally display the images
    fig, axs = plt.subplots(1, len(images) + 1, figsize=(15, 5))
    for j, img in enumerate(images):
        axs[j].imshow(img.squeeze(), cmap='gray')
        axs[j].set_title(f"Image {j}")
        axs[j].axis('off')

    axs[-1].imshow(target_image.squeeze(), cmap='gray')
    axs[-1].set_title("Target Image")
    axs[-1].axis('off')

    plt.show()

# Model initialization
input_channels = 1
hidden_channels = 128
output_channels = 1
kernel_size = 1
sequence_length = 5

model = Seq2SeqAutoencoder(input_channels, hidden_channels, output_channels, kernel_size)

# Determine split sizes
total_size = len(dataset)
train_size = int(0.7 * total_size)
val_size = int(0.15 * total_size)
test_size = total_size - (train_size + val_size)

# Split the dataset
train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(
    dataset, [train_size, val_size, test_size])

# Define batch size
batch_size = 5

# Create data loaders
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

model.to(device)

# Optimizer
optimizer = optim.Adam(model.parameters(), lr=0.001)

# training process
num_epochs = 30

for epoch in range(num_epochs):
    train_loss = train(model, train_loader, optimizer, device)
    val_loss = evaluate(model, val_loader, device)
    print(f"Epoch {epoch+1}, Train Loss: {train_loss}, Validation loss: {val_loss}")

torch.save(model.state_dict(), 'result/seq2seq_model_30.pth')

