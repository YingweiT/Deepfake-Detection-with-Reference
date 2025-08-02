import torch
import numpy as np
from sklearn.covariance import LedoitWolf
from tqdm import tqdm
from scipy.spatial.distance import mahalanobis
import pandas as pd
import ot
import json
from pytorch_fid import fid_score
import torch_fidelity
from scipy import linalg
from functools import partial

device = "cuda" if torch.cuda.is_available() else "cpu"
SRC_PATH = "../Data/GenImage/"
generator_names = ["adm", "bgan", "glide", "midj", "sd_14", "sd_15", "vqdm", "wukong"]
with open("classes.json", "r", encoding="utf-8") as f:
    data = json.load(f)
classes_idx = data["1k_idx"]
classes_names = data["21k_idx"]


def precalcul(pooled, X_real):
    X_real = X_real.float()
    # auto-bandwidth parameters(on gpu)
    dist_sq = torch.cdist(X_real, X_real).square()
    gamma_rbf = 1.0 / (2 * torch.median(dist_sq[dist_sq > 0]))  # 避免除零
    sigma_lap = torch.median(torch.cdist(X_real, X_real, p=1))

    return gamma_rbf, sigma_lap


def permutation_test_gpu(X, Y, fn, n_perm=1000):
    observed = fn(X, Y)
    pooled = torch.cat([X, Y])
    null_dist = torch.zeros(n_perm, device=X.device)

    for i in range(n_perm):
        idx = torch.randperm(len(pooled), device=X.device)
        X_perm = pooled[idx[: len(X)]]
        Y_perm = pooled[idx[len(X) :]]
        null_dist[i] = fn(X_perm, Y_perm)

    p_value = (torch.sum(null_dist >= observed) + 1) / (n_perm + 1)
    return observed.item(), p_value.item()


def rbf_kernel(X, Y, gamma=None):
    if gamma is None:
        dist = torch.cdist(X, X).square()
        gamma = 1.0 / (2 * torch.median(dist[dist > 0]))
    K_XY = torch.exp(-gamma * torch.cdist(X.float(), Y.float()).square())
    return K_XY


def laplacian_kernel(X, Y, sigma=None):
    if sigma is None:
        sigma = torch.median(torch.cdist(X.float(), X.float(), p=1))  # L1距离中位数
    K_XY = torch.exp(-torch.cdist(X.float(), Y.float(), p=1) / sigma)
    return K_XY


# def compute_mmd(K_XX, K_YY, K_XY):
#     """general MMD² computation"""
#     return K_XX.mean() + K_YY.mean() - 2 * K_XY.mean()


def compute_mmd(K_XX, K_YY, K_XY):
    """without bias MMD²"""
    n = K_XX.shape[0]
    m = K_YY.shape[0]
    K_XX_sum = (K_XX.sum() - K_XX.diag().sum()) / (n * (n - 1))
    K_YY_sum = (K_YY.sum() - K_YY.diag().sum()) / (m * (m - 1))
    K_XY_mean = K_XY.mean()
    return K_XX_sum + K_YY_sum - 2 * K_XY_mean


def mmd_with_kernel(X, Y, kernel_fn, **kernel_kwargs):
    """choose kernel MMD²"""
    K_XX = kernel_fn(X, X, **kernel_kwargs)
    K_YY = kernel_fn(Y, Y, **kernel_kwargs)
    K_XY = kernel_fn(X, Y, **kernel_kwargs)
    return compute_mmd(K_XX, K_YY, K_XY)


def w2_distance_ot(X: torch.Tensor, Y: torch.Tensor):
    """optimal transport without parameterization for Wasserstein-2"""
    X = X.detach().cpu().numpy()
    Y = Y.detach().cpu().numpy()
    n = X.shape[0]
    m = Y.shape[0]

    a = np.ones(n) / n
    b = np.ones(m) / m

    M = ot.dist(X, Y, metric="euclidean") ** 2

    w2_squared = ot.emd2(a, b, M)
    return np.sqrt(w2_squared)


def evaluate_mmd(X, Y, kernel_fn, class_name=None, model_name=None, n_perm=500, **kernel_kwargs):
    X, Y = X.to(device), Y.to(device)
    mmd_obs = mmd_with_kernel(X, Y, kernel_fn, **kernel_kwargs)

    pooled = torch.cat([X, Y])
    null_dist = torch.zeros(n_perm, device="cuda")
    for i in range(n_perm):
        idx = torch.randperm(len(pooled), device="cuda")
        X_perm = pooled[idx[: len(X)]]
        Y_perm = pooled[idx[len(X) :]]
        null_dist[i] = mmd_with_kernel(X_perm, Y_perm, kernel_fn, **kernel_kwargs)

    p_value = (torch.sum(null_dist >= mmd_obs) + 1) / (n_perm + 1)

    if "gamma" in kernel_kwargs:
        effect_size = mmd_obs * kernel_kwargs["gamma"]
        param = kernel_kwargs["gamma"].item()
    elif "sigma" in kernel_kwargs:
        effect_size = mmd_obs / (kernel_kwargs["sigma"] * kernel_kwargs["sigma"])
        param = kernel_kwargs["sigma"].item()

    return {
        "class": class_name,
        "model": model_name,
        "metric": kernel_fn.__name__,
        "distance": mmd_obs.item(),
        "p_value": p_value.item(),
        "effect_size": effect_size.item(),
        "params": param,
    }


def compute_fid(
    real_images_folder,
    generated_images_folder,
    class_name,
    model_name,
    batch_size=50,
    device=torch.device("cuda"),
    dims=2048,
    num_workers=0,
):
    # inception_model = torchvision.models.inception_v3(pretrained=True)
    fid_value = fid_score.calculate_fid_given_paths(
        [real_images_folder, generated_images_folder], batch_size, device, dims, num_workers
    )

    return {"class": class_name, "model": model_name, "metric": "fid", "distance": fid_value}


def calculate_frechet_distance(real_embeddings, ai_embeddings, eps=1e-6):
    # mu1, sigma1, mu2, sigma2,

    mu1 = torch.mean(real_embeddings, dim=0)
    sigma1 = torch.cov(real_embeddings.T)
    mu2 = torch.mean(ai_embeddings, dim=0)
    sigma2 = torch.cov(ai_embeddings.T)

    mu1 = mu1.cpu().numpy()
    mu2 = mu2.cpu().numpy()

    sigma1 = sigma1.cpu().numpy()
    sigma2 = sigma2.cpu().numpy()

    assert mu1.shape == mu2.shape, "mean dim not corresponding"
    assert sigma1.shape == sigma2.shape, "covariance dim not corresponding"

    diff = mu1 - mu2

    covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)
    if not np.isfinite(covmean).all():
        msg = ("fid calculation produces singular product; " "adding %s to diagonal of cov estimates") % eps
        print(msg)
        offset = np.eye(sigma1.shape[0]) * eps
        covmean = linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))

    if np.iscomplexobj(covmean):
        covmean = covmean.real
    tr_covmean = np.trace(covmean)

    result = diff.dot(diff) + np.trace(sigma1) + np.trace(sigma2) - 2 * tr_covmean

    return torch.tensor(result, device=real_embeddings.device)


def process(nature, ai, class_name, model_name):
    results = []

    nature = nature.to(device)
    ai = ai.to(device)

    pooled = torch.cat([nature, ai], dim=0)
    gamma_rbf, sigma_lap = precalcul(pooled, nature)
    kernel_configs = [
        {"kernel_fn": rbf_kernel, "gamma": gamma_rbf},
        {"kernel_fn": laplacian_kernel, "sigma": sigma_lap},
    ]
    for config in kernel_configs:
        result = evaluate_mmd(nature, ai, class_name=class_name, model_name=model_name, **config)
        results.append(result)

    w2d, p_value = permutation_test_gpu(nature, ai, w2_distance_ot, 500)

    results.append(
        {
            "class": class_name,
            "model": model_name,
            "metric": "w2_ot",
            "distance": w2d,
            "p_value": p_value,
            "effect_size": np.nan,
            "params": np.nan,
        }
    )

    fd, p_value = permutation_test_gpu(nature, ai, calculate_frechet_distance, 500)

    results.append(
        {
            "class": class_name,
            "model": model_name,
            "metric": "fd",
            "distance": fd,
            "p_value": p_value,
            "effect_size": np.nan,
            "params": np.nan,
        }
    )

    return results


if __name__ == "__main__":
    results_dino = []
    results_clip = []
    results_wst = []
    # results_fid = []
    for i in range(10):
        cls = classes_idx[i]

        # results_fid.append(compute_fid(SRC_PATH + cls + "/nature", SRC_PATH + "nature/" + cls, cls, "baseline"))
        # results_fid.append(compute_fid(SRC_PATH + cls + "/nature", SRC_PATH + cls + "/bgan", cls, "bgan"))
        # results_fid.append(compute_fid(SRC_PATH + cls + "/nature", SRC_PATH + cls + "/midj", cls, "midj"))
        # results_fid.append(compute_fid(SRC_PATH + cls + "/nature", SRC_PATH + cls + "/sd_15", cls, "sd_15"))

        bgan_dino = torch.load(f"../Data/Features/dinov2/{cls}/bgan.pt", weights_only=True)["features"]
        midj_dino = torch.load(f"../Data/Features/dinov2/{cls}/midj.pt", weights_only=True)["features"]
        sd_15_dino = torch.load(f"../Data/Features/dinov2/{cls}/sd_15.pt", weights_only=True)["features"]
        nature_dino = torch.load(f"../Data/Features/dinov2/{cls}/nature.pt", weights_only=True)["features"]
        nature_2_dino = torch.load(f"../Data/Features/dinov2/{cls}/nature_2.pt", weights_only=True)["features"]

        bgan_clip = torch.load(f"../Data/Features/clip/{cls}/bgan.pt", weights_only=True)["features"]
        midj_clip = torch.load(f"../Data/Features/clip/{cls}/midj.pt", weights_only=True)["features"]
        sd_15_clip = torch.load(f"../Data/Features/clip/{cls}/sd_15.pt", weights_only=True)["features"]
        nature_clip = torch.load(f"../Data/Features/clip/{cls}/nature.pt", weights_only=True)["features"]
        nature_2_clip = torch.load(f"../Data/Features/clip/{cls}/nature_2.pt", weights_only=True)["features"]

        bgan_wst = torch.load(f"../Data/Features/wst/{cls}/bgan.pt", weights_only=True)["features"]
        midj_wst = torch.load(f"../Data/Features/wst/{cls}/midj.pt", weights_only=True)["features"]
        sd_15_wst = torch.load(f"../Data/Features/wst/{cls}/sd_15.pt", weights_only=True)["features"]
        nature_wst = torch.load(f"../Data/Features/wst/{cls}/nature.pt", weights_only=True)["features"]
        nature_2_wst = torch.load(f"../Data/Features/wst/{cls}/nature_2.pt", weights_only=True)["features"]

        results_dino += process(
            nature_dino,
            nature_2_dino,
            class_name=cls,
            model_name="baseline",
        )
        results_dino += process(
            nature_dino,
            bgan_dino,
            class_name=cls,
            model_name="bgan",
        )
        results_dino += process(
            nature_dino,
            midj_dino,
            class_name=cls,
            model_name="midj",
        )
        results_dino += process(
            nature_dino,
            sd_15_dino,
            class_name=cls,
            model_name="sd_15",
        )

        results_clip += process(
            nature_clip,
            nature_2_clip,
            class_name=cls,
            model_name="baseline",
        )
        results_clip += process(
            nature_clip,
            bgan_clip,
            class_name=cls,
            model_name="bgan",
        )
        results_clip += process(
            nature_clip,
            midj_clip,
            class_name=cls,
            model_name="midj",
        )
        results_clip += process(
            nature_clip,
            sd_15_clip,
            class_name=cls,
            model_name="sd_15",
        )

        results_wst += process(
            nature_wst,
            nature_2_wst,
            class_name=cls,
            model_name="baseline",
        )
        results_wst += process(
            nature_wst,
            bgan_wst,
            class_name=cls,
            model_name="bgan",
        )
        results_wst += process(
            nature_wst,
            midj_wst,
            class_name=cls,
            model_name="midj",
        )
        results_wst += process(
            nature_wst,
            sd_15_wst,
            class_name=cls,
            model_name="sd_15",
        )

    df_dino = pd.DataFrame(results_dino)
    df_dino.to_csv("results_dino.csv", index=False)
    df_clip = pd.DataFrame(results_clip)
    df_clip.to_csv("results_clip.csv", index=False)
    # df_fid = pd.DataFrame(results_fid)
    # df_fid.to_csv("results_fid.csv", index=False)
    df_wst = pd.DataFrame(results_wst)
    df_wst.to_csv("results_wst.csv", index=False)
