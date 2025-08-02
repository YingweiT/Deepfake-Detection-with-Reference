import os
import json
import pandas as pd
import math
from pathlib import Path
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

from utils import prepare_for_clip_batch, prepare_for_dinov2_batch

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def clip_encoder(input_tensor, model, batch_size=64, device=DEVICE):
    model.eval()
    dataset = TensorDataset(input_tensor)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    all_features = []
    for batch in tqdm(dataloader, desc="WSTClip Encoding"):
        tensors = batch[0].to(device)  # (batch_size, 3, 256, 256)

        # Resize + Normalize
        tensors = prepare_for_clip_batch(tensors)

        with torch.no_grad():
            feats = model.encode_image(tensors)

        all_features.append(feats.cpu())
    all_features = torch.cat(all_features, dim=0)
    return all_features


def dinov2_encoder(input_tensor, model, mode="cls", batch_size=64, device=DEVICE):
    model.eval()
    dataset = TensorDataset(input_tensor)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    if mode == "cls":
        cls_tokens = []
        for batch in tqdm(dataloader, desc="WSTDinov2 Cls Encoding"):
            tensors = batch[0].to(device)
            tensors = prepare_for_dinov2_batch(tensors)
            with torch.no_grad():
                outputs = model(tensors)
                cls_token = outputs.last_hidden_state[:, 0, :]  # Only CLS
                cls_tokens.append(cls_token.detach().cpu())
        cls_tokens = torch.cat(cls_tokens, dim=0)
        return cls_tokens
    elif mode == "mean":
        features_all = []
        for batch in tqdm(dataloader, desc="WSTDinov2 Mean Encoding"):
            tensors = batch[0].to(device)
            tensors = prepare_for_dinov2_batch(tensors)
            with torch.no_grad():
                outputs = model(tensors)
                feat = outputs.last_hidden_state.mean(dim=1)  # Only CLS
                features_all.append(feat.detach().cpu())
        features = torch.cat(features_all, dim=0)
        return features
    elif mode == "patch":
        patch_tokens_all = []
        for batch in tqdm(dataloader, desc="WSTDinov2 Patches Encoding"):
            tensors = batch[0].to(device)
            tensors = prepare_for_dinov2_batch(tensors)
            with torch.no_grad():
                outputs = model(tensors)
                patch_tokens = outputs.last_hidden_state[:, 5:, :]  # remove CLS
                patch_tokens_all.append(patch_tokens.detach().cpu())
        patches = torch.cat(patch_tokens_all, dim=0)
        return patches
