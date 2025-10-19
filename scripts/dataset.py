import os
from pathlib import Path
from typing import List, Dict

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from PIL import Image
import timm
from transformers import AutoTokenizer

import albumentations as A
from albumentations.pytorch import ToTensorV2



def _parse_ingr_ids(s: str) -> List[str]:
    """'ingr_0000000122;ingr_0000000026;...' -> ['0000000122','0000000026', ...]"""
    if not isinstance(s, str) or s.strip() == "":
        return []
    return [t.split("_")[-1] for t in s.split(";") if t]


def _make_text(ingr_names: List[str], mass_g: float) -> str:
    # формулировка для текстовой модели
    names = ", ".join(ingr_names[:40])
    if np.isfinite(mass_g):
        return f"ingredients: {names}. mass: {int(mass_g)} g."
    return f"ingredients: {names}."


class CaloriesDataset(Dataset):
    def __init__(self, config, transforms, split: str = "train"):
        base = Path(config.BASE_DIR)
        self.img_dir = base / "images"

        # CSV
        dish = pd.read_csv(base / "dish.csv")
        ingr = pd.read_csv(base / "ingredients.csv")

        # сплит
        assert split in ("train", "test"), "split must be train|test"
        self.df = dish.query("split == @split").reset_index(drop=True)

        # маппинг id -> name
        id2name: Dict[str, str] = dict(
            zip(ingr["id"].astype(str).str.zfill(10), ingr["ingr"].astype(str))
        )

        # подготовим текст и путь к картинке
        self.df["ingr_ids"] = self.df["ingredients"].apply(_parse_ingr_ids)
        self.df["ingr_names"] = self.df["ingr_ids"].apply(
            lambda lst: [id2name.get(x, f"unknown_{x}") for x in lst]
        )
        self.df["text"] = self.df.apply(
            lambda r: _make_text(r["ingr_names"], r["total_mass"]), axis=1
        )
        self.df["image_path"] = self.df["dish_id"].apply(
            lambda d: str(self.img_dir / str(d) / "rgb.png")
        )

        self.image_cfg = timm.get_pretrained_cfg(config.IMAGE_MODEL_NAME)
        self.transforms = transforms

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        text = row["text"]
        target = float(row["total_calories"])
        mass = float(row["total_mass"])
        img_path = row["image_path"]

        # image -> tensor
        image = Image.open(img_path).convert("RGB")
        image = np.array(image)
        image = self.transforms(image=image)["image"]  # Tensor

        return {
            "image": image,
            "text": row["text"],
            "mass": torch.tensor([mass], dtype=torch.float32),
            "target": torch.tensor([target], dtype=torch.float32),
            "dish_id": str(row["dish_id"]),
            "image_path": img_path 
            }

tokenizer = AutoTokenizer.from_pretrained("huawei-noah/TinyBERT_General_4L_312D")

def collate_fn(batch):
    texts  = [b["text"] for b in batch]
    images = torch.stack([b["image"] for b in batch])
    targets = torch.stack([b["target"] for b in batch]).squeeze(1).float()
    masses  = torch.stack([b["mass"] for b in batch]).float()

    toks = tokenizer(
        texts, return_tensors="pt", padding=True, truncation=True, max_length=32,
    )

    return {
        "image": images,
        "input_ids": toks["input_ids"],
        "attention_mask": toks["attention_mask"],
        "mass": masses,
        "target": targets,
        "dish_id": [b["dish_id"] for b in batch],
        "image_path": [b["image_path"] for b in batch]
    }


def get_transforms(config, split: str = "train"):
    cfg = timm.get_pretrained_cfg(config.IMAGE_MODEL_NAME)
    H, W = cfg.input_size[1], cfg.input_size[2]

    if split == "train":
        return A.Compose(
            [
                A.SmallestMaxSize(max_size=max(H, W), p=1.0),
                A.RandomCrop(height=H, width=W, p=1.0),
                A.Affine(scale=(0.9, 1.1),
                         rotate=(-10, 10),
                         translate_percent=(-0.08, 0.08),
                         shear=(-6, 6),
                         p=0.4),
                A.ColorJitter(brightness=0.2, contrast=0.2,
                              saturation=0.15, hue=0.05, p=0.6),
                A.Normalize(mean=cfg.mean, std=cfg.std),
                ToTensorV2(p=1.0),
            ],
        )
    else:
        return A.Compose(
            [
                A.SmallestMaxSize(max_size=max(H, W), p=1.0),
                A.CenterCrop(height=H, width=W, p=1.0),
                A.Normalize(mean=cfg.mean, std=cfg.std),
                ToTensorV2(p=1.0),
            ],
        )
