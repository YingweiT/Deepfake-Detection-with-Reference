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


from kymatio.torch import Scattering2D
import pywt
from collections import defaultdict

device = "cuda" if torch.cuda.is_available() else "cpu"
SRC_PATH = "../Data/GenImage/"
generator_names = ["adm", "bgan", "glide", "midj", "sd_14", "sd_15", "vqdm", "wukong"]
with open("classes.json", "r", encoding="utf-8") as f:
    data = json.load(f)
classes_idx = data["1k_idx"]
classes_names = data["21k_idx"]


def resize_short_side(img, target_short_side):
    w, h = img.size
    if w < h:
        new_w = target_short_side
        new_h = int(h * (target_short_side / w))
    else:
        new_h = target_short_side
        new_w = int(w * (target_short_side / h))
    return img.resize((new_w, new_h), Image.BILINEAR)


def load_and_preprocess_images(
    folder_path,
    target_size=(256, 256),
    grayscale=False,
):
    if grayscale:
        channel_mode = "L"  # 灰度
        num_channels = 1
    else:
        channel_mode = "RGB"
        num_channels = 3

    transform = transforms.Compose(
        [
            transforms.Lambda(lambda img: resize_short_side(img, min(target_size))),
            transforms.CenterCrop(target_size),
            transforms.ToTensor(),  # [0,1] Tensor
        ]
    )

    image_tensors = []
    for filename in sorted(os.listdir(folder_path)):
        if filename.lower().endswith((".png", ".jpg", ".jpeg")):
            path = os.path.join(folder_path, filename)
            img = Image.open(path).convert(channel_mode)
            tensor = transform(img)
            image_tensors.append(tensor)

    if not image_tensors:
        raise ValueError("No image!")

    # batch tensor
    return torch.stack(image_tensors)  # (N, C, H, W)


class ImageFolderDataset(Dataset):
    def __init__(self, image_paths, transform):
        self.image_paths = image_paths
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        path = self.image_paths[idx]
        try:
            image = Image.open(path).convert("L")
            image = self.transform(image)
            return image, str(path)
        except:
            print("Failure open image.")
            return None


target_size = (256, 256)
transform = transforms.Compose(
    [
        transforms.Lambda(lambda img: resize_short_side(img, min(target_size))),  # short side resize
        transforms.CenterCrop(target_size),
        transforms.ToTensor(),  # [0,1] Tensor
    ]
)


def collate_fn(batch):
    batch = [item for item in batch if item is not None]
    if not batch:
        return None, None
    images, paths = zip(*batch)
    return torch.stack(images), paths


def wst(input_dir, transform, collate_fn, save_path, J=3, batch_size=64):
    input_dir = Path(input_dir)
    output_path = Path(save_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_size = (2, 2) if J == 3 else (4, 4)

    scattering = Scattering2D(J=J, shape=target_size).to(device)
    image_paths = list(input_dir.glob("*"))
    print(f"Found {len(image_paths)} pictures.")
    dataset = ImageFolderDataset(image_paths, transform=transform)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
    all_features, all_paths = [], []
    with torch.no_grad():
        for batch in tqdm(dataloader):
            batch_images, batch_paths = batch
            batch_images = batch_images.to(device)  # shape: [batch_size, 1, H, W]
            coeffs = scattering(batch_images)  # [batch_size, C, C', H', W']
            coeffs = coeffs.squeeze(1)
            pooled = torch.nn.functional.adaptive_avg_pool2d(coeffs, output_size=output_size)
            pooled_flattened = pooled.view(pooled.size(0), -1)
            all_features.append(pooled_flattened.cpu())
            all_paths.extend(batch_paths)
    all_features = torch.cat(all_features)
    print(all_features.shape)
    torch.save({"features": all_features, "paths": all_paths}, save_path)
    return all_features


if __name__ == "__main__":
    for i in range(10):
        cls = classes_idx[i]
        # wst(SRC_PATH + cls + "/bgan", transform, collate_fn, "../Data/Features/wst/" + cls + "/bgan.pt")
        # wst(SRC_PATH + cls + "/midj", transform, collate_fn, "../Data/Features/wst/" + cls + "/midj.pt")
        # wst(SRC_PATH + cls + "/sd_15", transform, collate_fn, "../Data/Features/wst/" + cls + "/sd_15.pt")
        # wst(SRC_PATH + cls + "/nature", transform, collate_fn, "../Data/Features/wst/" + cls + "/nature.pt")
        wst(SRC_PATH + cls + "/nature_2", transform, collate_fn, "../Data/Features/wst/" + cls + "/nature_2.pt")
