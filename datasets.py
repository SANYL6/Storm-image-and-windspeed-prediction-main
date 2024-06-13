import torch
from torch.utils.data import Dataset
from torchvision.transforms import ToTensor
import os
from PIL import Image, UnidentifiedImageError
import torchvision.transforms as transforms
import numpy as np
import random
import json
from sklearn.model_selection import train_test_split
from operator import itemgetter
import math


def set_seed(seed):
    """
    Use this to set ALL the random seeds to a fixed value and take out any
    randomness from cuda kernels
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.enabled = True
    return True


class StormDataset(Dataset):
    """
    Custom dataset for handling sequences of storm images for prediction.

    Args:
        root (str): Root directory containing storm image data.
        transform (bool, optional): Apply image transformations. Default is True.
        sequence_length (int, optional): Length of the image sequences. Default is 5.
        folder (str, optional): Subfolder containing storm data. Default is None.
    """

    def __init__(self, root, transform=True, sequence_length=5, stride=1, folder=None):
        self.transform = transform
        self.root = root
        self.sequence_length = sequence_length
        self.folder = folder
        self.stride = stride
        self.data = self._get_all_storm_sequences()

    def _get_all_storm_sequences(self):
        all_sequences = []
        if self.folder:
        # just consider one specfic folder
            storm_path = os.path.join(self.root, self.folder)
            storm_sequences = self._get_storm_sequences(storm_path)
            for seq in storm_sequences:
                all_sequences.append(seq)
        else:
        # go through all folders
            for storm_folder in os.listdir(self.root):
                storm_path = os.path.join(self.root, storm_folder)
                if os.path.isdir(storm_path):
                    storm_sequences = self._get_storm_sequences(storm_path)
                    for seq in storm_sequences:
                        all_sequences.append(seq)
        return all_sequences


    def _get_storm_sequences(self, storm_path):
        '''split storms into sequence'''
        paths = self._get_image_paths(storm_path)
        sequences = []
        for i in range(0, len(paths) - self.sequence_length, self.stride):
            sequence = paths[i:i + self.sequence_length]
            sequences.append(sequence)
        return sequences

    def _get_image_paths(self, storm_path, exts=(".jpg")):
        data = []
        for root, dirs, files in os.walk(storm_path):
            for file in files:
                if file.endswith(exts):
                    img_path = os.path.join(root, file)
                    num_path = img_path.rstrip('.jpg')
                    try:
                        features_path = num_path + "_features.json"
                        if os.path.exists(features_path):
                            with open(features_path, 'r') as f:
                                features = json.load(f)
                                time_feature = features.get('relative_time')
                                if time_feature is not None:
                                    data.append((img_path, int(time_feature)))
                    except UnidentifiedImageError:
                        print('image error')
                        pass

        data.sort(key=lambda x: x[1])
        return [item[0] for item in data]

    def __getitem__(self, idx):
        sequence_paths = self.data[idx]
        images = []
        for img_path in sequence_paths[:-1]:
            img = Image.open(img_path).convert('L')
            if self.transform:
                transform = transforms.Compose([
                    transforms.Resize((336, 336)),
                    transforms.ToTensor()
                ])
                img = transform(img)
            images.append(img)

        target_img_path = sequence_paths[-1]
        target_img = Image.open(target_img_path).convert('L')
        if self.transform:
            target_img = transform(target_img)

        return torch.stack(images), target_img

    def __len__(self):
        return len(self.data)

    def __str__(self):
        class_string = self.__class__.__name__
        class_string += f"\n\tlen : {self.__len__()}"
        for key, value in self.__dict__.items():
            class_string += f"\n\t{key} : {value}"
        return class_string