import numpy as np
import torch
import os
import json
from pathlib import Path
from tqdm import tqdm
from PIL import Image
from torchvision import transforms
import pandas as pd
import matplotlib.pyplot as plt

import torch.nn.functional as F
from sklearn.random_projection import SparseRandomProjection
from sklearn.neighbors import NearestNeighbors
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import roc_auc_score, precision_recall_curve, auc, roc_curve

device = "cuda" if torch.cuda.is_available() else "cpu"
SRC_PATH = "../Data/GenImage/"
generator_names = ["adm", "bgan", "glide", "midj", "sd_14", "sd_15", "vqdm", "wukong"]
with open("classes.json", "r", encoding="utf-8") as f:
    data = json.load(f)
classes_idx = data["1k_idx"]
classes_names = data["21k_idx"]


def greedy_coreset_selection(X, l):
    N, D = X.shape
    L = int(N * l)
    selected = []
    remaining = list(range(N))

    idx = np.random.choice(remaining)
    selected.append(idx)
    remaining.remove(idx)

    for _ in tqdm(range(L - 1)):
        dists = []
        for i in remaining:
            dist = np.min(np.linalg.norm(X[i] - X[selected], axis=1))
            dists.append(dist)
        new_idx = remaining[np.argmax(dists)]
        selected.append(new_idx)
        remaining.remove(new_idx)

    return X[selected], selected


def core_detector(test_samples, memory_bank, b=10):
    L = len(test_samples)
    test_samples = test_samples.cpu()
    nbrs = NearestNeighbors(n_neighbors=1, algorithm="auto").fit(memory_bank)
    distances, indices = nbrs.kneighbors(test_samples)
    distances = distances.squeeze()
    indices = indices.squeeze()

    m_stars = memory_bank[indices.tolist()]
    final_scores = []
    for i in range(L):
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


def detector(test_vectors, memory_bank, labels, save_path, b=10):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    test_scores = []
    scores = core_detector(test_vectors, memory_bank, b)
    test_scores += [-score for score in scores]
    test_scores = np.array(test_scores)

    fpr, tpr, thresholds = roc_curve(labels, scores)
    distances = np.sqrt((1 - tpr) ** 2 + fpr**2)
    best_threshold = thresholds[np.argmin(distances)]
    print("Best threshold(ROC):", best_threshold)
    roc_auc = roc_auc_score(labels, scores)
    # print("AUROC:", roc_auc)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    ax1.plot(fpr, tpr, color="darkorange", lw=2, label=f"ROC curve (area = {roc_auc:.2f})")
    ax1.plot([0, 1], [0, 1], color="navy", lw=2, linestyle="--")
    ax1.set_xlim([0.0, 1.0])
    ax1.set_ylim([0.0, 1.05])
    ax1.set_xlabel("False Positive Rate (FPR)")
    ax1.set_ylabel("True Positive Rate (TPR)")
    ax1.set_title("Receiver Operating Characteristic (ROC) Curve")
    ax1.legend(loc="lower right")

    precision, recall, thresholds = precision_recall_curve(labels, scores)
    pr_auc = auc(recall, precision)
    # print("AUPRC:", pr_auc)

    ax2.plot(recall, precision, color="blue", lw=2, label=f"PR curve (area = {pr_auc:.2f})")
    ax2.set_xlim([0.0, 1.0])
    ax2.set_ylim([0.0, 1.05])
    ax2.set_xlabel("Recall")
    ax2.set_ylabel("Precision")
    ax2.set_title("Precision-Recall Curve")
    ax2.legend(loc="best")

    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    return roc_auc, pr_auc


if __name__ == "__main__":
    for embedder in ["clip", "dinov2", "wst"]:
        auroc = []
        auprc = []
        generator = []
        for cls in classes_idx:
            bgan = torch.load(f"../Data/Features/{embedder}/{cls}/bgan.pt", weights_only=True)["features"]
            midj = torch.load(f"../Data/Features/{embedder}/{cls}/midj.pt", weights_only=True)["features"]
            sd_15 = torch.load(f"../Data/Features/{embedder}/{cls}/sd_15.pt", weights_only=True)["features"]
            nature = torch.load(f"../Data/Features/{embedder}/{cls}/nature.pt", weights_only=True)["features"]
            nature_2 = torch.load(f"../Data/Features/{embedder}/{cls}/nature_2.pt", weights_only=True)["features"]
            bgan_m = torch.cat([bgan, nature_2], dim=0).to(device)
            midj_m = torch.cat([midj, nature_2], dim=0).to(device)
            sd_15_m = torch.cat([sd_15, nature_2], dim=0).to(device)
            labels = np.concatenate((np.zeros(bgan.shape[0]), np.ones(nature_2.shape[0])))
            memory_bank, m_idx = greedy_coreset_selection(nature, l=0.5)

            r1, p1 = detector(
                bgan_m,
                memory_bank,
                labels,
                f"../Data/Core_results/{embedder}/{cls}/bgan.png",
            )
            generator.append("bgan")
            auroc.append(r1)
            auprc.append(p1)
            r2, p2 = detector(
                midj_m,
                memory_bank,
                labels,
                f"../Data/Core_results/{embedder}/{cls}/midj.png",
            )
            generator.append("midj")
            auroc.append(r2)
            auprc.append(p2)
            r3, p3 = detector(
                sd_15_m,
                memory_bank,
                labels,
                f"../Data/Core_results/{embedder}/{cls}/sd_15.png",
            )
            generator.append("sd_15")
            auroc.append(r3)
            auprc.append(p3)
        data = {
            "CLASS": [x for x in classes_idx for _ in range(3)],
            "GENERATOR": generator,
            "AUROC": auroc,
            "AUPRC": auprc,
        }
        df = pd.DataFrame(data)
        df.to_csv(embedder + "_core_result.csv", index=False)
        print(embedder + f" auroc: {np.mean(auroc)}, auprc : {np.mean(auprc)}")
