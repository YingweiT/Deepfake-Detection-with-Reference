import os
import json
from pathlib import Path
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from torchvision import transforms


def load_images(folder, target_size=(256, 256), max_samples=1000, grey=True):
    images = []
    expect_shape = target_size if grey else (target_size[0], target_size[1], 3)
    for filename in tqdm(os.listdir(folder)[:max_samples]):
        if filename.lower().endswith((".png", ".jpg", ".jpeg")):
            try:
                img = Image.open(os.path.join(folder, filename))
                if grey:
                    img = img.convert("L")
                    img_resized = np.array(img.resize(target_size)) / 255.0
                else:
                    img = img.convert("RGB")
                    img_resized = np.array(img.resize(target_size)) / 255.0
                assert img_resized.shape == expect_shape
                images.append(img_resized)
            except Exception as e:
                print(f"Skipped {filename}: {str(e)}")
    return np.array(images)


def load_images_nature(folder, size=256, max_samples=1000, grey=True):
    images = []
    transform = transforms.Compose([transforms.Resize(size), transforms.CenterCrop(size)])
    expect_shape = (size, size) if grey else (size, size, 3)
    for filename in tqdm(os.listdir(folder)[:max_samples]):
        if filename.lower().endswith((".png", ".jpg", ".jpeg")):
            try:
                img = Image.open(os.path.join(folder, filename))
                img = transform(img)
                if grey:
                    img = img.convert("L")
                    img_resized = np.array(img) / 255.0
                else:
                    img = img.convert("RGB")
                    img_resized = np.array(img) / 255.0
                assert img_resized.shape == expect_shape
                images.append(img_resized)
            except Exception as e:
                print(f"Skipped {filename}: {str(e)}")
    return np.array(images)
