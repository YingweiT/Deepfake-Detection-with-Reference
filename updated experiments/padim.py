import os
import json
from pathlib import Path
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, TensorDataset
from PIL import Image
from torchvision import transforms
from rpo import GaussianProjector, SparseProjector

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def gt_compute(embeddings, eps=1e-3):
    mean = embeddings.mean(dim=0, keepdim=True)
    X = embeddings - mean
    cov = X.T @ X / (embeddings.size(0) - 1)
    cov += eps * torch.eye(cov.size(0), device=embeddings.device)
    return mean, cov


def mahalanobis_distance(x, mean, cov):
    x = x.to(torch.float32).view(-1)
    mean = mean.to(torch.float32).view(-1)
    delta = x - mean
    cov = cov.to(torch.float32)

    try:
        sol = torch.linalg.solve(cov, delta.unsqueeze(1))  # [D, 1]
        dist_squared = delta @ sol.squeeze()
        if dist_squared < 0:
            print("Warning: distance squared < 0", dist_squared.item())
            dist_squared = torch.clamp(dist_squared, min=0.0)
        dist = torch.sqrt(dist_squared)
        return dist
    except RuntimeError as e:
        print("Runtime error in Mahalanobis:", e)
        return torch.tensor(float("nan"), device=x.device)


def mahalanobis_detector(gt_tensor, real_tensor, fake_tensor, device=DEVICE):
    gt_tensor, real_tensor, fake_tensor = gt_tensor.to(device), real_tensor.to(device), fake_tensor.to(device)
    gt_mean, gt_cov = gt_compute(gt_tensor)

    test_tensor = torch.cat([real_tensor, fake_tensor], dim=0)
    test_scores = []
    for sample in tqdm(test_tensor, desc="Mahalanobis scoring:"):
        distance = mahalanobis_distance(sample, gt_mean, gt_cov)
        test_scores.append(distance.cpu())
    scores = np.array(test_scores)
    return scores


def padim_detector(gt_patches, real_patches, fake_patches, target_dim=100, device=DEVICE):
    gt_patches, real_patches, fake_patches = gt_patches.to(device), real_patches.to(device), fake_patches.to(device)
    projector = SparseProjector(input_dim=gt_tensor.shape[2], target_dim=target_dim, device=device, seed=2025)
    test_patches = torch.cat([real_patches, fake_patches], dim=0)
    gt_patches = projector.project(gt_patches)
    test_patches = projector.project(test_patches)

    labels = np.concatenate((np.zeros(real_patches.shape[0]), np.ones(fake_patches.shape[0])))
    gt_coeffs = []
    test_scores = []

    for i in range(gt_patches.shape[1]):
        gt_tensor = gt_patches[:, i, :]
        gt_mean, gt_cov = gt_compute(gt_tensor)
        gt_mean, gt_cov = gt_mean.to(device), gt_cov.to(device)
        gt_coeffs.append((gt_mean, gt_cov))

    for sample in tqdm(test_patches, desc="PaDiM Scoring"):
        score_per_patch = []
        for i in range(test_patches.shape[1]):
            distance = mahalanobis_distance(sample[i], gt_coeffs[i][0], gt_coeffs[i][1])
            score_per_patch.append(distance.cpu())
        test_scores.append(max(score_per_patch))
    scores = np.array(test_scores)

    return scores
