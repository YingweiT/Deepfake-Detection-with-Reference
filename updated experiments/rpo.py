import os
import json
import math
import pandas as pd
from pathlib import Path
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt

from PIL import Image
from torchvision import transforms

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from sklearn.random_projection import SparseRandomProjection, GaussianRandomProjection
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import roc_auc_score, precision_recall_curve, auc, roc_curve
from scipy.special import softmax


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


class SparseProjector:
    def __init__(self, input_dim, target_dim, s=None, device=DEVICE, seed=2025):
        """
        input_dim: original dimension D
        target_dim: target dimension d
        s: parameter of sparsity, by default sqrt(D)
        device: 'cuda' or 'cpu'
        """
        self.input_dim = input_dim
        self.target_dim = target_dim
        self.device = device
        self.s = s or int(math.sqrt(input_dim))
        self.seed = seed

        if seed is not None:
            torch.manual_seed(seed)

        self.projection_matrix = self._generate_sparse_projection_matrix()

    def _generate_sparse_projection_matrix(self):
        D, d, s = self.input_dim, self.target_dim, self.s
        prob_nonzero = 1.0 / s

        # initialize as 0
        R = torch.zeros(D, d, device=self.device)

        # generate mask：1/(2s) prob to be +1，1/(2s) to be -1, the rest is 0
        rand_vals = torch.rand(D, d, device=self.device)

        # generate +1 / -1 's positions
        pos_mask = rand_vals < (1 / (2 * s))
        neg_mask = (rand_vals >= (1 / (2 * s))) & (rand_vals < (1 / s))

        # fill blanks
        R[pos_mask] = math.sqrt(s)
        R[neg_mask] = -math.sqrt(s)

        return R  # shape: [D, d]

    def project(self, X):
        """
        X: Tensor of shape [N, D], must be on same device
        Return: [N, d]
        """
        if X.device != self.device:
            X = X.to(self.device)
        return X @ self.projection_matrix


class GaussianProjector:
    def __init__(self, input_dim, target_dim, device=DEVICE, seed=2025):
        """
        input_dim: original dimension D
        target_dim: target dimension d
        device: 'cuda' or 'cpu'
        seed: seed of randomness
        """
        self.input_dim = input_dim
        self.target_dim = target_dim
        self.device = device
        self.seed = seed

        if seed is not None:
            torch.manual_seed(seed)

        self.projection_matrix = self._generate_gaussian_projection_matrix()

    def _generate_gaussian_projection_matrix(self):
        D, d = self.input_dim, self.target_dim
        # Normal distribution generate D×d matrix，usually / sqrt(d) to contral std（following conditions of Johnson–Lindenstrauss）
        R = torch.randn(D, d, device=self.device) / math.sqrt(d)
        return R  # shape: [D, d]

    def project(self, X):
        """
        X: Tensor of shape [N, D], must be on same device
        Return: [N, d]
        """
        if X.device != self.device:
            X = X.to(self.device)
        return X @ self.projection_matrix


def mad(x, dim=-1, keepdim=False):
    """
    Calculate median absolute deviation (MAD)
    MAD = median(|x - median(x)|)
    """
    med = x.median(dim=dim, keepdim=True)[0]
    mad = (x - med).abs().median(dim=dim, keepdim=keepdim)[0]

    mad = torch.clamp(mad, min=1e-6)
    return mad


class GaussianRPO:
    def __init__(self, D, dtype, M=1000, device="cuda", seed=None):
        """
        D: Data Dimension
        M: Number of projectors
        """
        self.D = D
        self.M = M
        self.device = device
        if seed is not None:
            torch.manual_seed(seed)

        # Gaussian random projection M x D
        U = torch.randn(M, D, device=device, dtype=dtype)
        U = U / U.norm(dim=1, keepdim=True)  # Normalization to sphere of norm 1
        self.U = U  # projector matrix

    def fit(self, X):
        """
        Input X: shape (B, D)
        Calculation of all MED 和 MAD to cache
        """
        X = X.to(self.device)
        # M x B，projection
        proj = torch.matmul(self.U, X.T)  # shape (M, B)
        self.med = proj.median(dim=1)[0]  # shape (M,)
        self.mad = mad(proj, dim=1)  # shape (M,)

    def score(self, x):
        """
        Calculation of RPO score for input batch
        x: (N, D)
        return shape (N,)
        """
        x = x.to(self.device)
        # (M, N) = (M, D) @ (D, N)
        proj_x = torch.matmul(self.U, x.T)  # shape (M, N)

        # broadcast : (M, 1)
        med = self.med.unsqueeze(1)
        mad = self.mad.unsqueeze(1)

        # computation
        score = torch.abs(proj_x - med) / mad  # shape (M, N)

        # choose the max score as final score for each sample
        rpo_score, _ = score.max(dim=0)  # shape (N,)

        return rpo_score


class SparseRPO:
    def __init__(self, D, dtype, M=1000, s=None, device="cuda", seed=None):
        """
        D: original dimension
        M: number of projectors
        s: parameter of scale of sparsity, by default sqrt(D)
        """
        self.D = D
        self.M = M
        self.device = device
        self.s = s or int(math.sqrt(D))
        if seed is not None:
            torch.manual_seed(seed)

        self.U = self._generate_sparse_projection_matrix(dtype)

    def _generate_sparse_projection_matrix(self, dtype):
        D, M, s = self.D, self.M, self.s
        prob_nonzero = 1.0 / s

        # initialize all zeros
        U = torch.zeros(M, D, device=self.device, dtype=dtype)

        # generate uniform random matrix
        rand_vals = torch.rand(M, D, device=self.device)

        pos_mask = rand_vals < (1 / (2 * s))
        neg_mask = (rand_vals >= (1 / (2 * s))) & (rand_vals < (1 / s))

        val = math.sqrt(s)
        U[pos_mask] = val
        U[neg_mask] = -val

        # Note: following the implementation of sklearn, there is no normalization

        return U

    def fit(self, X):
        """
        calculate MED and MAD for every projection direction
        X: (B, D)
        """
        X = X.to(self.device)
        proj = torch.matmul(self.U, X.T)  # (M, B)
        self.med = proj.median(dim=1)[0]  # (M,)
        self.mad = mad(proj, dim=1)  # (M,)

    def score(self, x):
        """
        RPO score calculation
        x: (N, D)
        """
        x = x.to(self.device)
        proj_x = torch.matmul(self.U, x.T)  # (M, N)
        med = self.med.unsqueeze(1)  # (M, 1)
        mad = self.mad.unsqueeze(1)  # (M, 1)

        score = torch.abs(proj_x - med) / mad  # (M, N)
        rpo_score, _ = score.max(dim=0)  # (N,)

        return rpo_score
