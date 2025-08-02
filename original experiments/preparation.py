import os
import random
import shutil
from pathlib import Path
import json


generator_names = ["adm", "bgan", "glide", "midj", "sd_14", "sd_15", "vqdm", "wukong"]

with open("class.json", "r", encoding="utf-8") as f:
    data = json.load(f)
classes_idx = data["1k_idx"]
classes_names = data["21k_idx"]


def map1k21k(idx_1k):
    names_21k = []
    idx_21k = []
    names_1k = []
    with open("../Data/GenImage/imagenet21k_wordnet_ids.txt", "r", encoding="utf-8") as f:
        for idx, line in enumerate(f):
            line = line.strip()
            idx_21k.append(idx + 1)
            names_21k.append(line)
    with open("../Data/GenImage/map21k21k.txt", "r", encoding="utf-8") as f:
        for idx, line in enumerate(f):
            line = line.strip()
            names_1k.append(line)
            assert int(line) != 9206 and int(line) != 15028
    print(len(names_21k), len(names_1k))
    corresponding_names = [names_21k[int(names_1k[int(idx_1k[i])])] for i in range(len(idx_1k))]
    return corresponding_names


def verify_test(
    top_path,
):
    ai_path = Path(top_path + "/val/ai")
    nature_path = Path(top_path + "/val/nature")
    ai_paths = []
    nature_paths = []
    for idx, class_id in enumerate(classes_names):
        nature_pattern = f"{class_id}_*"
        nature_files = list(nature_path.glob(nature_pattern))
        # real_images = list(Path(nature_dir).glob("*"))
        try:
            assert len(nature_files) == 50
        except AssertionError as e:
            print(f"Nature files in {nature_path}/{class_id}({classes_idx[idx]}) not 50, {e}")
        nature_paths.append(nature_files)

        ai_pattern = f"{int(classes_idx[idx])}_adm_*.png"
        ai_files = list(ai_path.glob(ai_pattern))
        try:
            assert len(ai_files) == 50
        except AssertionError as e:
            print(f"AI files in {ai_path}/{classes_idx[idx]} not 50, {e}")
        ai_paths.append(ai_files)
    return ai_paths, nature_paths


def verify_train_nature(top_path):
    nature_path = Path(top_path + "/train/nature")
    nature_paths = []
    for idx, class_id in enumerate(classes_names):
        nature_pattern = f"{class_id}_*"
        nature_files = list(nature_path.glob(nature_pattern))
        # real_images = list(Path(nature_dir).glob("*"))
        if not nature_files:
            print(f"Error: Class{class_id} has no pictures in {nature_path}.")
        nature_paths.append(nature_files)
    return nature_paths
