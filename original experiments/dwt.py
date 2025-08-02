import os
import json
from pathlib import Path
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt

import pywt
from PIL import Image
from torchvision import transforms
from kymatio.torch import Scattering2D
from collections import defaultdict


def resize_and_center_crop(image, target_size=(256, 256)):
    target_w, target_h = target_size
    w, h = image.size

    scale = max(target_w / w, target_h / h)
    new_w, new_h = int(w * scale), int(h * scale)
    image = image.resize((new_w, new_h), resample=Image.BILINEAR)

    left = (new_w - target_w) // 2
    top = (new_h - target_h) // 2
    right = left + target_w
    bottom = top + target_h
    image = image.crop((left, top, right, bottom))

    return image


def load_image(image_path):
    image = Image.open(image_path).convert("L")
    image = resize_and_center_crop(image)
    return np.array(image)


def load_images_from_folder(folder):
    images = []
    for filename in os.listdir(folder):
        if filename.lower().endswith((".png", ".jpg", ".jpeg")):
            img_path = os.path.join(folder, filename)
            img = Image.open(img_path).convert("L")
            img = resize_and_center_crop(img)
            images.append(np.array(img))
    return images


def extract_wavelet_stats(image_array, wavelet="sym4", level=3):
    coeffs = pywt.wavedec2(image_array, wavelet=wavelet, level=level)
    stats = {}

    for i, c in enumerate(coeffs[1:], start=1):
        cH, cV, cD = c
        for name, comp in zip(["cH", "cV", "cD"], [cH, cV, cD]):
            stats[f"{name}_L{i}"] = {"mean": np.mean(comp), "std": np.std(comp), "energy": np.sum(comp**2)}

    return stats


def compute_folder_wavelet_stats(image_list):
    all_stats = []
    for img in image_list:
        stats = extract_wavelet_stats(img)
        all_stats.append(stats)

    return all_stats


def aggregate_stats(stats_list):
    aggregate = defaultdict(list)

    for stats in stats_list:
        for key, sub in stats.items():
            for subkey, val in sub.items():
                aggregate[f"{key}_{subkey}"].append(val)

    return {k: np.mean(v) for k, v in aggregate.items()}


def compute_stats_for_folders(real_folder, ai_folders, wavelet="sym4", level=3):
    real_images = load_images_from_folder(real_folder)
    ai_images_list = [load_images_from_folder(folder) for folder in ai_folders]

    real_stats = aggregate_stats([extract_wavelet_stats(img) for img in real_images])

    ai_stats_list = []
    for ai_images in ai_images_list:
        stats = aggregate_stats([extract_wavelet_stats(img) for img in ai_images])
        ai_stats_list.append(stats)

    return real_stats, ai_stats_list


def plot_multi_folder_comparison(real_stats, ai_stats_list, ai_folder_names, path, level=3):

    os.makedirs(os.path.dirname(path), exist_ok=True)
    coeff_types = ["cH", "cV", "cD"]
    metrics = ["energy", "mean", "std"]
    n_ai_folders = len(ai_stats_list)

    # 3 columns × n rows
    fig, axes = plt.subplots(n_ai_folders, 3, figsize=(15, 5 * n_ai_folders))
    if n_ai_folders == 1:
        axes = axes.reshape(1, -1)

    for row in range(n_ai_folders):
        ai_stats = ai_stats_list[row]
        folder_name = ai_folder_names[row]

        for col, metric in enumerate(metrics):
            ax = axes[row, col]

            real_values = []
            ai_values = []
            labels = []

            for coeff in coeff_types:
                for lvl in range(1, level + 1):
                    key = f"{coeff}_L{lvl}"
                    real_values.append(real_stats[f"{key}_{metric}"])
                    ai_values.append(ai_stats[f"{key}_{metric}"])
                    labels.append(key)

            x = np.arange(len(labels))
            width = 0.35
            ax.bar(x - width / 2, real_values, width, label="Real", color="blue")
            ax.bar(x + width / 2, ai_values, width, label=f"AI: {folder_name}", color="orange")

            ax.set_xticks(x)
            ax.set_xticklabels(labels, rotation=45, ha="right")
            ax.set_title(f"{metric.upper()} Comparison ({folder_name})")
            ax.legend()

    plt.tight_layout()
    plt.savefig(path)
    plt.close()


SRC_PATH = "../Data/GenImage/"

if __name__ == "__main__":
    with open("classes.json", "r", encoding="utf-8") as f:
        data = json.load(f)
    classes_idx = data["1k_idx"]
    classes_names = data["21k_idx"]

    for i in range(10):
        cls = classes_idx[i]
        real_folder = SRC_PATH + cls + "/nature"
        ai_folders = [SRC_PATH + cls + "/bgan", SRC_PATH + cls + "/midj", SRC_PATH + cls + "/sd_15"]
        ai_folder_names = ["bgan", "midj", "sd_15"]

        real_stats, ai_stats_list = compute_stats_for_folders(real_folder, ai_folders)

        plot_multi_folder_comparison(real_stats, ai_stats_list, ai_folder_names, f"../Data/DWT/{cls}.png")
