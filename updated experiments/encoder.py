import os
import random
import shutil
from pathlib import Path
import json
import math

from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, TensorDataset
from PIL import Image
from torchvision import transforms

from kymatio.torch import Scattering2D
import pywt
from collections import defaultdict
import clip
from transformers import AutoImageProcessor, AutoModel, AutoProcessor, AutoModelForImageClassification

from scipy.special import softmax

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


class ClipDataset(Dataset):
    def __init__(self, image_paths, transform):
        self.image_paths = image_paths
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        path = self.image_paths[idx]
        try:
            image = Image.open(path).convert("RGB")
            image = self.transform(image)
            return image, str(path)
        except:
            print("Failure open image.")
            return None


class Dinov2Dataset(Dataset):
    def __init__(self, image_paths):
        self.image_paths = image_paths

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        path = self.image_paths[idx]
        try:
            image = Image.open(path).convert("RGB")
            return image, str(path)
        except Exception as e:
            print(f"Failure open image because of {e}")
            return None


class WSTDataset(Dataset):
    def __init__(self, image_paths, transform):
        self.image_paths = image_paths
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        path = self.image_paths[idx]
        try:
            image = Image.open(path).convert("L")
            image = self.transform(image)
            return image, str(path)
        except:
            print("Failure open image.")
            return None


def collate_fn(batch):
    batch = [item for item in batch if item is not None]
    if not batch:
        return None, None
    images, paths = zip(*batch)
    return torch.stack(images), paths


def dinov2_collate_fn(batch):
    batch = [item for item in batch if item is not None]
    if not batch:
        return None, None
    images, paths = zip(*batch)
    return list(images), paths


def resize_short_side(img, target_short_side):
    w, h = img.size
    if w < h:
        new_w = target_short_side
        new_h = int(h * (target_short_side / w))
    else:
        new_h = target_short_side
        new_w = int(w * (target_short_side / h))
    return img.resize((new_w, new_h), Image.BILINEAR)


wst_shape = (256, 256)
wst_preprocess = transforms.Compose(
    [
        transforms.Lambda(lambda img: resize_short_side(img, min(wst_shape))),
        transforms.CenterCrop(wst_shape),
        transforms.ToTensor(),
    ]
)


class Encoder:
    def __init__(self, model_name="dinov2", patchify=False, cls_token=False, J=None, device=DEVICE):
        self.model_name = model_name
        self.patchify = patchify
        self.cls_token = cls_token
        self.device = device
        if self.model_name == "clip":
            self.model, self.preprocess = clip.load("ViT-B/32", device=device)
        elif self.model_name == "dinov2":
            self.preprocess = AutoImageProcessor.from_pretrained("facebook/dinov2-with-registers-base")
            self.model = AutoModel.from_pretrained("facebook/dinov2-with-registers-base").to(device)
        elif self.model_name == "wst":
            if not J:
                print("Error, not parameter J for wst.")
            else:
                self.J = J
                self.preprocess = wst_preprocess
                self.model = Scattering2D(J=J, shape=wst_shape).to(device)

    def __call__(self, input_dir, batch_size=81):
        if self.model_name == "clip":
            input_dir = Path(input_dir)
            image_paths = list(input_dir.glob("*"))
            dataset = ClipDataset(image_paths, transform=self.preprocess)
            dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)

            if self.patchify:
                assert self.cls_token == False
                self.model.float()
                visual = self.model.visual
                patch_tokens_all = []
                for images, paths in tqdm(dataloader, desc="Clip Patches Encoding"):
                    with torch.no_grad():
                        x = images.to(self.device)
                        x = visual.conv1(x)  # shape = [*, width, grid, grid]
                        x = x.reshape(x.shape[0], x.shape[1], -1)  # shape = [*, width, grid ** 2]
                        x = x.permute(0, 2, 1)  # shape = [*, grid ** 2, width]
                        x = torch.cat(
                            [
                                visual.class_embedding.to(x.dtype)
                                + torch.zeros(x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device),
                                x,
                            ],
                            dim=1,
                        )  # shape = [*, grid ** 2 + 1, width]
                        x = x + visual.positional_embedding.to(x.dtype)
                        x = visual.ln_pre(x)
                        x = x.permute(1, 0, 2)  # NLD -> LND
                        x = visual.transformer(x)
                        x = x.permute(1, 0, 2)
                        patch_tokens = visual.ln_post(x[:, 1:, :])
                        patch_tokens_all.append(patch_tokens.cpu())
                patches = torch.cat(patch_tokens_all, dim=0)
                print(patches.shape)
                return patches

            else:
                assert self.patchify == False and self.cls_token == False
                self.model.eval()
                all_features = []
                with torch.no_grad():
                    for images, paths in tqdm(dataloader, desc="Clip Direct Encoding"):
                        images = images.to(self.device)
                        feats = self.model.encode_image(images).cpu()
                        all_features.append(feats)
                all_features = torch.cat(all_features)
                print(all_features.shape)
                return all_features

        elif self.model_name == "dinov2":
            input_dir = Path(input_dir)
            image_paths = list(input_dir.glob("*"))
            dataset = Dinov2Dataset(image_paths)
            dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, collate_fn=dinov2_collate_fn)

            if self.patchify:
                assert self.cls_token == False
                self.model.eval()
                patch_tokens_all = []
                for images, paths in tqdm(dataloader, desc="Dinov2 Patches Encoding"):
                    # processor expects a list of PIL images
                    inputs = self.preprocess(images=images, return_tensors="pt").to(self.device)
                    with torch.no_grad():
                        outputs = self.model(**inputs)
                        patch_tokens = outputs.last_hidden_state[:, 5:, :]  # remove CLS and registers
                        patch_tokens_all.append(patch_tokens.detach().cpu())

                patches = torch.cat(patch_tokens_all, dim=0)
                print(patches.shape)
                return patches

            elif self.cls_token:
                assert self.patchify == False
                self.model.eval()
                cls_token_all = []
                for images, paths in tqdm(dataloader, desc="Dinov2 Cls Encoding"):
                    inputs = self.preprocess(images=images, return_tensors="pt").to(self.device)
                    with torch.no_grad():
                        outputs = self.model(**inputs)
                        cls_token = outputs.last_hidden_state[:, 0, :]  # Only CLS
                        cls_token_all.append(cls_token.detach().cpu())
                cls_tokens = torch.cat(cls_token_all, dim=0)
                print(cls_tokens.shape)
                return cls_tokens

            else:
                assert self.patchify == False and self.cls_token == False
                self.model.eval()
                all_features = []
                with torch.no_grad():
                    for images, paths in tqdm(dataloader, desc="Dinov2 Direct Encoding"):
                        inputs = self.preprocess(images=images, return_tensors="pt").to(self.device)
                        outputs = self.model(**inputs)
                        feats = outputs.last_hidden_state.mean(dim=1)
                        all_features.append(feats.detach().cpu())
                all_features = torch.cat(all_features)
                print(all_features.shape)
                return all_features

        elif self.model_name == "wst":
            assert self.cls_token == False and self.patchify == False
            input_dir = Path(input_dir)
            image_paths = list(input_dir.glob("*"))
            dataset = WSTDataset(image_paths, transform=self.preprocess)
            dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
            wst_results = []
            with torch.no_grad():
                for batch in tqdm(dataloader, desc=f"WST-{self.J} Direct Encoding"):
                    batch_images, batch_paths = batch
                    batch_images = batch_images.to(self.device)  # shape: [batch_size, 1, H, W]
                    coeffs = self.model(batch_images)  # 对整个批次进行散射变换，形状: [batch_size, C, C', H', W']
                    coeffs = coeffs.squeeze(1)
                    wst_results.append(coeffs.cpu())
            wst_tensor = torch.cat(wst_results, dim=0)
            print(wst_tensor.shape)
            return wst_tensor

            # if self.J == 2:
            #     ## output (B,81,64,64)
            #     target_channel = 64
            #     target_size = 32
            # elif self.J ==3:
            #     ## output (B,217, 32, 32)
            #     target_channel = 128
            #     target_size = 16
