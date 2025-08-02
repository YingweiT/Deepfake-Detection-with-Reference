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
from sklearn.neighbors import NearestNeighbors

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def greedy_coreset_selection(X, l, device=DEVICE):
    X = X.to(device)  # X: [N, D]
    N, D = X.shape
    L = int(N * l)

    selected_idx = []
    remaining_mask = torch.ones(N, dtype=torch.bool, device=device)
    dtype = X.dtype
    if dtype == torch.float16:
        mask_val = -1e4
    else:
        mask_val = -1e9

    idx = torch.randint(0, N, (1,)).item()
    selected_idx.append(idx)
    remaining_mask[idx] = False

    min_dists = torch.cdist(X, X[idx].unsqueeze(0)).squeeze(1)  # [N]

    for _ in range(L):
        min_dists[~remaining_mask] = mask_val
        next_idx = torch.argmax(min_dists).item()
        selected_idx.append(next_idx)
        remaining_mask[next_idx] = False

        dist_to_new_point = torch.cdist(X, X[next_idx].unsqueeze(0)).squeeze(1)
        min_dists = torch.minimum(min_dists, dist_to_new_point)

    return X[selected_idx].cpu(), selected_idx


def batch_anomaly_scores(test_patches_batch, memory_bank, b=10, device=DEVICE):
    """
    test_patches_batch: [B, N_test, D]
    memory_bank: [N_mem, D]
    return:
        scores: [B], anomaly scores for every batch
    """
    B, N_test, D = test_patches_batch.shape
    N_mem = memory_bank.shape[0]

    test_patches_batch = test_patches_batch.to(device)
    memory_bank = memory_bank.to(device)

    # Step 1: Compute distances from every patch to memory bank [B, N_test, N_mem]
    dists = torch.cdist(test_patches_batch, memory_bank.unsqueeze(0).expand(B, -1, -1))  # broadcast memory_bank

    # Step 2: Search distance and index of nearest neighbour for every patch [B, N_test]
    min_dists, nn_indices = torch.min(dists, dim=2)  # minimum distance and index

    # Step 3: Search index of patch in every batch which has maximum in minimum distances [B]
    max_min_dists, max_idx = torch.max(min_dists, dim=1)  # max(min(distances)) for every batch

    # Step 4: Take corresponding patch and the corresponding NN point in memory bank [B, D]
    batch_idx = torch.arange(B, device=device)
    m_test_star = test_patches_batch[batch_idx, max_idx]  # [B, D]
    m_star = memory_bank[nn_indices[batch_idx, max_idx]]  # [B, D]

    # Step 5: Compute b-NN points of m_star [B, b, D]
    # For every batch element, compute distance between m_star and memory bank
    m_star_expand = m_star.unsqueeze(1)  # [B,1,D]
    dists_b = torch.cdist(m_star_expand, memory_bank.unsqueeze(0).expand(B, -1, -1)).squeeze(1)  # [B, N_mem]
    _, neighbors_b_idx = torch.topk(dists_b, b, largest=False)  # [B, b]
    neighbors_b = memory_bank[neighbors_b_idx]  # [B, b, D]

    # Step 6: Compute weights of distance
    dist_to_neighbors = torch.norm(neighbors_b - m_test_star.unsqueeze(1), dim=2)  # [B, b]
    d_star = torch.norm(m_test_star - m_star, dim=1)  # [B]

    # Step 7: Compute softmax weights
    scores_cat = torch.cat([d_star.unsqueeze(1), dist_to_neighbors], dim=1)  # [B, b+1]
    probs = F.softmax(scores_cat, dim=1)  # [B, b+1]
    weight = 1 - probs[:, 0]  # [B]

    # Step 8: Final weighted scores
    final_scores = weight * max_min_dists  # [B]

    return final_scores


def anomaly_scores(test_patches, memory_bank, batch_size=64, b=10, device=DEVICE):
    """
    test_patches: [N_total, N_patches, D]
    memory_bank: [N_mem, D]
    return:
    final_scores: [N_total]
    """
    N_total = test_patches.shape[0]
    memory_bank = memory_bank.to(device)

    final_scores_list = []

    with torch.no_grad():
        for start_idx in tqdm(range(0, N_total, batch_size), desc="PatchCore Scoring"):
            end_idx = min(start_idx + batch_size, N_total)
            batch = test_patches[start_idx:end_idx].to(device)  # [batch_size, N_patches, D]
            scores_batch = batch_anomaly_scores(batch, memory_bank, b=b, device=device)  # [batch_size]
            final_scores_list.append(scores_batch)

    final_scores = torch.cat(final_scores_list, dim=0)
    return final_scores.cpu().numpy()


def coreset_detector(memory_bank, real_tensor, fake_tensor, b=10):
    test_samples = torch.cat([real_tensor, fake_tensor], dim=0)
    L = len(test_samples)
    test_samples = test_samples.cpu()
    nbrs = NearestNeighbors(n_neighbors=1, algorithm="auto").fit(memory_bank)
    distances, indices = nbrs.kneighbors(test_samples)
    distances = distances.squeeze()
    indices = indices.squeeze()

    m_stars = memory_bank[indices.tolist()]
    final_scores = []
    for i in tqdm(range(L), desc="Coreset Scoring"):
        m_star = m_stars[i]
        m_test_star = test_samples[i]

        nbrs_b = NearestNeighbors(n_neighbors=b, algorithm="auto").fit(memory_bank)
        _, neighbors_b_idx = nbrs_b.kneighbors(m_star.reshape(1, -1))
        neighbors_b = memory_bank[neighbors_b_idx[0]]  # shape: (b, D)

        dist_to_neighbors = np.linalg.norm(m_test_star - neighbors_b, axis=1)
        d_star = np.linalg.norm(m_test_star - m_star)
        exps = np.exp(dist_to_neighbors - np.max(dist_to_neighbors))  # subtract max
        weight = 1 - (np.exp(d_star - np.max(dist_to_neighbors)) / np.sum(exps))

        final_scores.append(weight * distances[i])
    return final_scores


def patchcore_detector(memory_bank, real_patches, fake_patches, b=10, batch_size=64, device=DEVICE):
    test_patches = torch.cat([real_patches, fake_patches], dim=0)
    scores = anomaly_scores(test_patches, memory_bank, batch_size=batch_size, b=b, device=device)
    return scores
