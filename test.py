from storm_tasks.task1.datasets import StormDataset
from storm_tasks.task1.model import Seq2SeqAutoencoder
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torchvision.transforms import ToTensor
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
from storm_tasks.task1.model import Seq2SeqAutoencoder
dtype = torch.cuda.FloatTensor
from storm_tasks.task1.train_process import train,evaluate
import os
from PIL import Image, UnidentifiedImageError
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import json
import random
# Model initialization
input_channels = 1
hidden_channels = 128
output_channels = 1
kernel_size = 1
sequence_length = 5

# prepare cpu
device = 'cpu'
if torch.cuda.device_count() > 0 and torch.cuda.is_available():
    print("Cuda installed! Running on GPU!")
    device = 'cuda'
else:
    print("No GPU available!")

model = Seq2SeqAutoencoder(input_channels, hidden_channels, output_channels, kernel_size)
model.load_state_dict(torch.load(r'C:\Users\ThinkStation\Desktop\Tropical-Storm-Image-and-Wind-Speed-Prediction-Tool-main\Tropical-Storm-Image-and-Wind-Speed-Prediction-Tool-main\Tropical-Storm-Image-and-Wind-Speed-Prediction-Tool-main\task1_model——30.pth'))

def get_last_tensors():
  image_directory = r"C:\Users\ThinkStation\Desktop\Tropical-Storm-Image-and-Wind-Speed-Prediction-Tool-main\Tropical-Storm-Image-and-Wind-Speed-Prediction-Tool-main\Tropical-Storm-Image-and-Wind-Speed-Prediction-Tool-main\storm\pjj"

  # Get all .jpg files in the directory
  image_files = [os.path.join(image_directory, f) for f in os.listdir(image_directory) if f.endswith('.jpg')]

  image_files.sort()

  # Get the last four images
  last_four_images = image_files[-4:]

  print("Last four images:", last_four_images)

  images = []
  for img_path in last_four_images:
    img = Image.open(img_path).convert('L')
    transform = transforms.Compose([
        transforms.Resize((336, 336)),
        transforms.ToTensor()
        ])
    img = transform(img)
    images.append(img)

  last_tensors = torch.stack(images).unsqueeze(0)
  last_three = torch.stack(images[-3:]).unsqueeze(0)
  last_two = torch.stack(images[-2:]).unsqueeze(0)

  return last_tensors, last_three, last_two

last_tensors, last_three, last_two  = get_last_tensors()

def make_prediction(model, device, last_tensors):
  model.eval()
  model.to(device)

  with torch.no_grad():
      last_tensors = last_tensors.to(device)
      prediction = model(last_tensors)

  return prediction

prediction_1 = make_prediction(model, device, last_tensors)

last_three = last_three.to(device)
# Concatenate along the second dimension (dimension index 1)
last_tensors_2 = torch.cat((last_three, prediction_1), dim=1)

prediction_2 = make_prediction(model, device, last_tensors_2)

last_two = last_two.to(device)

# Concatenate along the second dimension (dimension index 1)
last_tensors_3 = torch.cat((last_two, prediction_1, prediction_2), dim=1)

prediction_3 = make_prediction(model, device, last_tensors_3)


def save_tensor(tensor, i):
  tensor = tensor.squeeze()
  tensor = (tensor - tensor.min()) / (tensor.max() - tensor.min())

  # Ensure tensor is of type float32 for saving
  tensor = tensor.type(torch.FloatTensor)

  # Save the tensor as an image
  save_image(tensor, f'test_predict_{i}.jpg')

  # Plotting the image
  plt.imshow(tensor.cpu().detach().numpy(), cmap='gray')
  plt.axis('off')
  plt.show()

save_tensor(prediction_1, 1)
save_tensor(prediction_2, 2)
save_tensor(prediction_3, 3)

