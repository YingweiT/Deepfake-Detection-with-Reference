import math
import torch
import torch.nn.functional as F
from einops import rearrange


def wst_m(x, projector, flatten=True):
    B, C, H, W = x.shape
    x = x.permute(0, 2, 3, 1)
    x_flat = x.reshape(B * H * W, C)
    x_proj = projector.project(x_flat)
    x_proj = x_proj.reshape(B, H, W, -1)
    x_proj = x_proj.permute(0, 3, 1, 2)
    if flatten:
        x_final = x_proj.flatten(start_dim=1)
    else:
        x_final = x_proj
    return x_final


def compute_fixed_pca(tensor, n_components=48):
    # Input: wst_tensor shape: (B, C=81, H, W)
    B, C, H, W = tensor.shape
    X = tensor.permute(0, 2, 3, 1).reshape(-1, C)  # shape: (B*H*W, C)

    # Centralization
    X_mean = X.mean(dim=0, keepdim=True)
    X_centered = X - X_mean

    # SVD (or use torch.pca_lowrank)
    U, S, Vh = torch.linalg.svd(X_centered, full_matrices=False)
    Vh_reduced = Vh[:n_components]  # shape: (D, C)

    return Vh_reduced, X_mean


def apply_fixed_pca(tensor, Vh_reduced, X_mean):
    # wst_tensor: (B, C=81, H, W)
    B, C, H, W = tensor.shape
    X = tensor.permute(0, 2, 3, 1).reshape(-1, C)  # (B*H*W, C)
    X_centered = X - X_mean[None, :]  # (B*H*W, C)
    X_reduced = X_centered @ Vh_reduced.T  # (B*H*W, D)
    X_reduced_reshaped = X_reduced.view(B, H, W, -1).permute(0, 3, 1, 2)  # (B, D, H, W)

    return X_reduced_reshaped


def normalize_image(tensor):

    B, C, H, W = tensor.shape
    tensor_flat = tensor.view(B, -1)  # (B, C*H*W)
    min_val = tensor_flat.min(dim=1, keepdim=True)[0]
    max_val = tensor_flat.max(dim=1, keepdim=True)[0]
    normalized = (tensor_flat - min_val) / (max_val - min_val + 1e-8)
    return normalized.view(B, C, H, W)


def normalize_channel(tensor):

    B, C, H, W = tensor.shape
    tensor_flat = tensor.view(B, C, -1)  # (B, C, H*W)
    min_val = tensor_flat.min(dim=2, keepdim=True)[0]
    max_val = tensor_flat.max(dim=2, keepdim=True)[0]
    normalized = (tensor_flat - min_val) / (max_val - min_val + 1e-8)
    return normalized.view(B, C, H, W)


def reshape_to_3(x_reduced):
    # x_reduced: (B, D=48, 64, 64)
    B, D, H, W = x_reduced.shape
    if D == 48:
        x_split = x_reduced.view(B, 3, 16, H, W)  # (B, 3, 16, 64, 64)
        x_reshaped = rearrange(x_split, "B c (h w) H W -> B c (h H) (w W)", h=4, w=4)
    elif D == 192:
        x_split = x_reduced.view(B, 3, 64, H, W)  # (B, 3, 64, 32, 32)
        x_reshaped = rearrange(x_split, "B c (h w) H W -> B c (h H) (w W)", h=8, w=8)
    return x_reshaped  # (B, 3, 256, 256)


def prepare_for_clip_batch(wst_tensor: torch.Tensor) -> torch.Tensor:

    # Step 1: Resize to (B, 3, 224, 224)
    resized = F.interpolate(wst_tensor, size=224, mode="bicubic", align_corners=False)

    # Step 2: Normalize (Use broadcast to realize batch normalize)
    mean = torch.tensor([0.48145466, 0.4578275, 0.40821073], device=wst_tensor.device).view(1, 3, 1, 1)
    std = torch.tensor([0.26862954, 0.26130258, 0.27577711], device=wst_tensor.device).view(1, 3, 1, 1)

    normalized = (resized - mean) / std  # automatically broadcast to batch
    return normalized


def prepare_for_dinov2_batch(wst_tensor: torch.Tensor) -> torch.Tensor:

    wst_tensor = wst_tensor * 0.00392156862745098
    x = wst_tensor

    crop_size = 224
    _, _, H, W = x.shape
    top = (H - crop_size) // 2
    left = (W - crop_size) // 2
    x = x[:, :, top : top + crop_size, left : left + crop_size]  # (B, 3, 224, 224)

    # Step 3: Normalize using ImageNet mean/std
    mean = torch.tensor([0.485, 0.456, 0.406], device=x.device).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], device=x.device).view(1, 3, 1, 1)
    x = (x - mean) / std

    return x
