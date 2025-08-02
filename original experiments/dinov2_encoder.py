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
from transformers import AutoImageProcessor, AutoModel, AutoProcessor, AutoModelForImageClassification


device = "cuda" if torch.cuda.is_available() else "cpu"
SRC_PATH = "../Data/GenImage/"
generator_names = ["adm", "bgan", "glide", "midj", "sd_14", "sd_15", "vqdm", "wukong"]
with open("classes.json", "r", encoding="utf-8") as f:
    data = json.load(f)
classes_idx = data["1k_idx"]
classes_names = data["21k_idx"]

processor = AutoImageProcessor.from_pretrained("facebook/dinov2-base")
model = AutoModel.from_pretrained("facebook/dinov2-base").to(device)


class ImageFolderDataset(Dataset):
    def __init__(self, image_paths):
        self.image_paths = image_paths

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        path = self.image_paths[idx]
        try:
            image = Image.open(path).convert("RGB")
            return image, str(path)
        except Exception as e:
            print(f"Failure open image because of {e}")
            return None


def collate_fn(batch):
    batch = [item for item in batch if item is not None]
    if not batch:
        return None, None
    images, paths = zip(*batch)
    return list(images), paths


def dinov2_encode_2(input_dir, model, transform, collate_fn, save_path, batch_size=64):
    input_dir = Path(input_dir)
    output_path = Path(save_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    image_paths = list(input_dir.glob("*"))
    print(f"Found {len(image_paths)} pictures.")
    dataset = ImageFolderDataset(image_paths)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
    # os.makedirs(save_path,exist_ok = True)
    all_features, all_paths = [], []
    with torch.no_grad():
        for images, paths in tqdm(dataloader, desc="Image Encoding"):
            if images is None:
                continue
            # processor expects a list of PIL images
            inputs = transform(images=images, return_tensors="pt").to(device)
            outputs = model(**inputs)
            feats = outputs.last_hidden_state.mean(dim=1).cpu()
            all_features.append(feats)
            all_paths.extend(paths)
    all_features = torch.cat(all_features)
    print(all_features.shape)
    torch.save({"features": all_features, "paths": all_paths}, save_path)
    return all_features


if __name__ == "__main__":
    for i in range(10):
        _ = dinov2_encode_2(
            SRC_PATH + classes_idx[i] + "/nature_2",
            model,
            processor,
            collate_fn,
            f"../Data/Features/dinov2/{classes_idx[i]}/nature_2.pt",
        )
