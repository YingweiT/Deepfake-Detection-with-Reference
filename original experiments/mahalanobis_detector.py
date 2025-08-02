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
from sklearn.metrics import roc_auc_score, precision_recall_curve, auc, roc_curve
import pandas as pd

device = "cuda" if torch.cuda.is_available() else "cpu"
SRC_PATH = "../Data/GenImage/"
generator_names = ["adm", "bgan", "glide", "midj", "sd_14", "sd_15", "vqdm", "wukong"]
with open("classes.json", "r", encoding="utf-8") as f:
    data = json.load(f)
classes_idx = data["1k_idx"]
classes_names = data["21k_idx"]


def gt_compute(gt_path, eps=1e-3):
    embeddings = torch.load(gt_path, weights_only=True)["features"].to(device)
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


# def combine_embeddings(path1, path2):
#     embeddings1 = torch.load(path1, weights_only=True)["features"]
#     embeddings2 = torch.load(path2, weights_only=True)["features"]


def detector(test_images, gt_path, labels, save_path):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    gt_mean, gt_cov = gt_compute(gt_path)
    gt_mean, gt_cov = gt_mean.to(device), gt_cov.to(device)
    # test_results = []
    test_scores = []
    for sample in test_images:
        distance = mahalanobis_distance(sample, gt_mean, gt_cov)
        test_scores.append(-distance.cpu())
    #     if distance >= threshold:
    #         test_results.append(0)
    #     else:
    #         test_results.append(1)
    # results = np.array(test_results)
    scores = np.array(test_scores)
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
            nature_2 = torch.load(f"../Data/Features/{embedder}/{cls}/nature_2.pt", weights_only=True)["features"]
            bgan_m = torch.cat([bgan, nature_2], dim=0).to(device)
            midj_m = torch.cat([midj, nature_2], dim=0).to(device)
            sd_15_m = torch.cat([sd_15, nature_2], dim=0).to(device)
            labels = np.concatenate((np.zeros(bgan.shape[0]), np.ones(nature_2.shape[0])))
            r1, p1 = detector(
                bgan_m,
                f"../Data/Features/{embedder}/{cls}/nature.pt",
                labels,
                f"../Data/Results/{embedder}/{cls}/bgan.png",
            )
            generator.append("bgan")
            auroc.append(r1)
            auprc.append(p1)
            r2, p2 = detector(
                midj_m,
                f"../Data/Features/{embedder}/{cls}/nature.pt",
                labels,
                f"../Data/Results/{embedder}/{cls}/midj.png",
            )
            generator.append("midj")
            auroc.append(r2)
            auprc.append(p2)
            r3, p3 = detector(
                sd_15_m,
                f"../Data/Features/{embedder}/{cls}/nature.pt",
                labels,
                f"../Data/Results/{embedder}/{cls}/sd_15.png",
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
        df.to_csv(embedder + "_padim_result.csv", index=False)
        print(embedder + f" auroc: {np.mean(auroc)}, auprc : {np.mean(auprc)}")
