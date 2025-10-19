import os
import random
from functools import partial

import numpy as np
import timm
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.utils.data import DataLoader
from transformers import AutoModel

from dataset import CaloriesDataset, get_transforms, collate_fn


def seed_everything(seed: int = 42):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True


def set_requires_grad(module: nn.Module, unfreeze_pattern: str = "", verbose=False):
    if len(unfreeze_pattern) == 0:
        for _, p in module.named_parameters():
            p.requires_grad = False
        return
    pats = unfreeze_pattern.split("|")
    for name, p in module.named_parameters():
        if any([name.startswith(t) for t in pats]):
            p.requires_grad = True
            if verbose:
                print("unfreeze:", name)
        else:
            p.requires_grad = False


class MultimodalRegressor(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.text_model = AutoModel.from_pretrained(config.TEXT_MODEL_NAME)
        txt_dim = self.text_model.config.hidden_size

        self.image_model = timm.create_model(
            config.IMAGE_MODEL_NAME, pretrained=True, num_classes=0
        )
        img_dim = self.image_model.num_features

        self.mass_fc = nn.Sequential(
            nn.Linear(1, config.MASS_HIDDEN),
            nn.ReLU(),
        )

        self.text_proj = nn.Linear(txt_dim, config.HIDDEN_DIM)
        self.image_proj = nn.Linear(img_dim, config.HIDDEN_DIM)

        self.regressor = nn.Sequential(
            nn.Linear(config.HIDDEN_DIM * 2 + config.MASS_HIDDEN, config.HIDDEN_DIM),
            nn.LayerNorm(config.HIDDEN_DIM),
            nn.ReLU(),
            nn.Dropout(config.DROPOUT),
            nn.Linear(config.HIDDEN_DIM, 1),
        )

    def forward(self, input_ids, attention_mask, image, mass):
        text_feat = self.text_model(input_ids=input_ids,
                                    attention_mask=attention_mask
                                    ).last_hidden_state[:, 0, :]
        img_feat = self.image_model(image)

        text_emb = self.text_proj(text_feat)
        img_emb  = self.image_proj(img_feat)
        mass_emb = self.mass_fc(mass)

        fused = torch.cat([text_emb, img_emb, mass_emb], dim=1)
        pred = self.regressor(fused).squeeze(1)
        return pred


@torch.no_grad()
def validate(model, loader, device):
    model.eval()
    mae_sum, n = 0.0, 0
    for batch in loader:
        inputs = {
            "input_ids": batch["input_ids"].to(device),
            "attention_mask": batch["attention_mask"].to(device),
            "image": batch["image"].to(device),
            "mass": batch["mass"].to(device),
        }
        target = batch["target"].to(device)
        pred = model(**inputs)
        mae_sum += torch.abs(pred - target).sum().item()
        n += target.numel()
    return mae_sum / max(n, 1)


def train(config, device):
    seed_everything(config.SEED)

    # модель
    model = MultimodalRegressor(config).to(device)

    set_requires_grad(model.text_model,  config.TEXT_MODEL_UNFREEZE,  verbose=True)
    set_requires_grad(model.image_model, config.IMAGE_MODEL_UNFREEZE, verbose=True)

    # оптимизатор с различными LRs
    optimizer = AdamW([
        {"params": model.text_model.parameters(),  "lr": config.TEXT_LR},
        {"params": model.image_model.parameters(), "lr": config.IMAGE_LR},
        {"params": list(model.mass_fc.parameters())
                   + list(model.text_proj.parameters())
                   + list(model.image_proj.parameters())
                   + list(model.regressor.parameters()),
         "lr": config.HEAD_LR},
    ], weight_decay=config.WEIGHT_DECAY)

    tr_tfms = get_transforms(config, split="train")
    te_tfms = get_transforms(config, split="test")

    train_ds = CaloriesDataset(config, tr_tfms, split="train")
    val_ds   = CaloriesDataset(config, te_tfms, split="test")

    train_loader = DataLoader(train_ds, batch_size=config.BATCH_SIZE,
                              shuffle=True, num_workers=config.NUM_WORKERS, pin_memory=True, collate_fn=collate_fn)
    val_loader   = DataLoader(val_ds, batch_size=config.BATCH_SIZE,
                              shuffle=False, num_workers=config.NUM_WORKERS, pin_memory=True, collate_fn=collate_fn)

    scaler = torch.cuda.amp.GradScaler(enabled=config.USE_AMP)
    mae_best = float("inf")

    print("training started")
    for epoch in range(1, config.EPOCHS + 1):
        model.train()
        running_mae, seen = 0.0, 0

        for batch in train_loader:
            inputs = {
                "input_ids": batch["input_ids"].to(device),
                "attention_mask": batch["attention_mask"].to(device),
                "image": batch["image"].to(device),
                "mass": batch["mass"].to(device),
            }
            target = batch["target"].to(device)

            optimizer.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=config.USE_AMP):
                pred = model(**inputs)
                loss = torch.nn.functional.l1_loss(pred, target)  # MAE

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            running_mae += torch.abs(pred - target).sum().item()
            seen += target.numel()

        train_mae = running_mae / max(seen, 1)
        val_mae = validate(model, val_loader, device)

        print(f"Epoch {epoch:02d}/{config.EPOCHS} | train MAE: {train_mae:.2f} | val MAE: {val_mae:.2f}")

        # сохраним лучшую модель
        if val_mae < mae_best:
            mae_best = val_mae
            os.makedirs(os.path.dirname(config.SAVE_PATH), exist_ok=True)
            torch.save(model.state_dict(), config.SAVE_PATH)
            print(f"New best, saved to {config.SAVE_PATH} (MAE={mae_best:.2f})")
