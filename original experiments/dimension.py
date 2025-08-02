import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn import manifold
from sklearn.metrics.pairwise import rbf_kernel
from skimage import io, color, transform
from scipy.fftpack import fft2, fftshift, ifft2, ifftshift
from scipy.stats import permutation_test
from scipy.spatial.distance import pdist, squareform
import umap
import os
import json
from pathlib import Path
from tqdm import tqdm
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from torchvision import transforms
import preprocess
from sklearn.manifold import TSNE


def plot_pca_components(pca_result, labels, explained_variance, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    plt.figure(figsize=(12, 5))

    plt.subplot(121)
    plt.scatter(pca_result[labels == 1, 0], pca_result[labels == 1, 1], alpha=0.5, label="Real")
    plt.scatter(pca_result[labels == 0, 0], pca_result[labels == 0, 1], alpha=0.5, label="AI")
    plt.title(f"Top 2 PCs (Total Var: {explained_variance[:2].sum():.1%})")
    plt.xlabel(f"PC1 ({explained_variance[0]:.1%})")
    plt.ylabel(f"PC2 ({explained_variance[1]:.1%})")

    plt.subplot(122)
    plt.scatter(pca_result[labels == 1, -2], pca_result[labels == 1, -1], alpha=0.5, label="Real")
    plt.scatter(pca_result[labels == 0, -2], pca_result[labels == 0, -1], alpha=0.5, label="AI")
    plt.title(f"Bottom 2 PCs (Total Var: {explained_variance[-2:].sum():.1%})")
    plt.xlabel(f"PC{len(explained_variance)-1} ({explained_variance[-2]:.1%})")
    plt.ylabel(f"PC{len(explained_variance)} ({explained_variance[-1]:.1%})")

    plt.legend()
    plt.tight_layout()
    plt.savefig(path)
    plt.close()
    # plt.show()


def pca_transform(
    input_path_ai, input_path_nature, output_path, n_components=64, size=256, max_samples=1000, grey=True
):
    ai_imgs = preprocess.load_images(input_path_ai, (size, size), max_samples, grey)
    # ai_imgs = ai_imgs.reshape(len(ai_imgs), -1)
    nature_imgs = preprocess.load_images_nature(input_path_nature, size, max_samples, grey)
    # nature_imgs = nature_imgs.reshape(len(nature_imgs), -1)
    print(ai_imgs.shape, nature_imgs.shape)
    all_imgs = np.concatenate([nature_imgs, ai_imgs])
    all_imgs = all_imgs.reshape(len(all_imgs), -1)  # (n_samples, H*W(*3))
    labels = np.array([1] * len(nature_imgs) + [0] * len(ai_imgs))
    pca = PCA(n_components=n_components)
    pca_result = pca.fit_transform(all_imgs)
    plot_pca_components(pca_result, labels, pca.explained_variance_ratio_, output_path)
    return pca, pca_result


def tSNE_transform(input_1, input_2, output_path, name, labels=None, n_components=128, random_state=2025):
    Path(output_path).mkdir(parents=True, exist_ok=True)
    if isinstance(input_1, torch.Tensor):
        input_1 = input_1.cpu().numpy()
    if isinstance(input_2, torch.Tensor):
        input_2 = input_2.cpu().numpy()

    all_input = np.concatenate([input_1, input_2], axis=0)
    # labels = np.array([1]*len(input_2) + [0]*len(input_1))
    tsne = TSNE(n_components=n_components, perplexity=30, random_state=random_state)
    tsne_results = tsne.fit_transform(all_input)
    ai_tsne = tsne_results[: len(input_1)]
    real_tsne = tsne_results[len(input_1) :]

    if n_components == 2:

        plt.figure(figsize=(15, 6))

        plt.subplot(1, 2, 1)
        plt.scatter(ai_tsne[:, 0], ai_tsne[:, 1], c="red", label="AI Images", s=5, alpha=0.6)
        plt.scatter(real_tsne[:, 0], real_tsne[:, 1], c="blue", label="Real Images", s=5, alpha=0.6)
        plt.title("tSNE Visualization")
        plt.legend()
        plt.tight_layout()
        plt.savefig(f"{output_path}/{name}_tsne_result.png")
        plt.close()

    else:
        ai_tsne_tensor = torch.from_numpy(ai_tsne).float()
        real_tsne_tensor = torch.from_numpy(real_tsne).float()

        torch.save(
            {
                "ai_tsne": ai_tsne_tensor,
                "real_tsne": real_tsne_tensor,
            },
            f"{output_path}/{name}.pt",
        )

        print(f"Results saved to {output_path}/")
        return {
            "ai_tsne": ai_tsne_tensor,
            "real_tsne": real_tsne_tensor,
        }


def UMAP_transform(input_1, input_2, output_path, name, labels=None, n_components=128, random_state=2025):
    Path(output_path).mkdir(parents=True, exist_ok=True)
    if isinstance(input_1, torch.Tensor):
        input_1 = input_1.cpu().numpy()
    if isinstance(input_2, torch.Tensor):
        input_2 = input_2.cpu().numpy()

    all_input = np.concatenate([input_1, input_2], axis=0)
    # all_input = all_input.reshape(len(all_input), -1)
    # labels = np.array([1]*len(input_2) + [0]*len(input_1))
    umap_reducer = umap.UMAP(n_components=n_components, random_state=random_state)
    umap_results = umap_reducer.fit_transform(all_input)
    ai_umap = umap_results[: len(input_1)]
    real_umap = umap_results[len(input_1) :]

    if n_components == 2:

        plt.figure(figsize=(15, 6))

        plt.subplot(1, 2, 1)
        plt.scatter(ai_umap[:, 0], ai_umap[:, 1], c="red", label="AI Images", s=5, alpha=0.6)
        plt.scatter(real_umap[:, 0], real_umap[:, 1], c="blue", label="Real Images", s=5, alpha=0.6)
        plt.title("UMAP Visualization")
        plt.legend()
        plt.tight_layout()
        plt.savefig(f"{output_path}/{name}_umap_result.png")
        plt.close()

    else:
        ai_umap_tensor = torch.from_numpy(ai_umap).float()
        real_umap_tensor = torch.from_numpy(real_umap).float()

        torch.save(
            {
                "ai_umap": ai_umap_tensor,
                "real_umap": real_umap_tensor,
            },
            f"{output_path}/{name}.pt",
        )

        print(f"Results saved to {output_path}/")
        return {
            "ai_umap": ai_umap_tensor,
            "real_umap": real_umap_tensor,
        }


if __name__ == "__main__":
    with open("classes.json", "r", encoding="utf-8") as f:
        data = json.load(f)
    classes_idx = data["1k_idx"]
    classes_names = data["21k_idx"]
    for i in range(10):
        cls = classes_idx[i]
        # bgan = preprocess.load_images(f"../Data/GenImage/{cls}/bgan")
        # midj = preprocess.load_images(f"../Data/GenImage/{cls}/midj")
        # sd_15 = preprocess.load_images(f"../Data/GenImage/{cls}/sd_15")
        # nature = preprocess.load_images_nature(f"../Data/GenImage/{cls}/nature")
        # UMAP_transform(bgan, nature, "../Data/UMAP/original", name=cls + "_bgan", n_components=2)
        # UMAP_transform(midj, nature, "../Data/UMAP/original", name=cls + "_midj", n_components=2)
        # UMAP_transform(sd_15, nature, "../Data/UMAP/original", name=cls + "_sd_15", n_components=2)
        # bgan_dino = torch.load(f"../Data/Features/dinov2/{cls}/bgan.pt", weights_only=True)["features"]
        # midj_dino = torch.load(f"../Data/Features/dinov2/{cls}/midj.pt", weights_only=True)["features"]
        # sd_15_dino = torch.load(f"../Data/Features/dinov2/{cls}/sd_15.pt", weights_only=True)["features"]
        # nature_dino = torch.load(f"../Data/Features/dinov2/{cls}/nature.pt", weights_only=True)["features"]
        bgan_clip = torch.load(f"../Data/Features/clip/{cls}/bgan.pt", weights_only=True)["features"]
        midj_clip = torch.load(f"../Data/Features/clip/{cls}/midj.pt", weights_only=True)["features"]
        sd_15_clip = torch.load(f"../Data/Features/clip/{cls}/sd_15.pt", weights_only=True)["features"]
        nature_clip = torch.load(f"../Data/Features/clip/{cls}/nature.pt", weights_only=True)["features"]
        tSNE_transform(bgan_clip, nature_clip, "../Data/tSNE/clip/bgan", name=cls, n_components=2)
        tSNE_transform(midj_clip, nature_clip, "../Data/tSNE/clip/midj", name=cls, n_components=2)
        tSNE_transform(sd_15_clip, nature_clip, "../Data/tSNE/clip/sd_15", name=cls, n_components=2)
        UMAP_transform(bgan_clip, nature_clip, "../Data/UMAP/clip/bgan", name=cls, n_components=2)
        UMAP_transform(midj_clip, nature_clip, "../Data/UMAP/clip/midj", name=cls, n_components=2)
        UMAP_transform(sd_15_clip, nature_clip, "../Data/UMAP/clip/sd_15", name=cls, n_components=2)
