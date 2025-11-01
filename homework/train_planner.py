"""
Usage:
  Colab:
    from homework.train_planner import train
    train(model_name="mlp_planner", transform_pipeline="state_only")

  CLI:
    python3 -m homework.train_planner --model-name mlp_planner --transform-pipeline state_only
"""
from __future__ import annotations

import argparse
import math
import os
from pathlib import Path
from typing import Dict, Any

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from .models import MODEL_FACTORY, save_model
# Dataset utilities — these names are consistent with the README hints.
# If your repo exposes different names, adjust the imports below.
from .datasets.road_dataset import RoadDataset
from .datasets import road_transforms


def _get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


from inspect import signature
from .datasets import road_transforms
from .datasets.road_dataset import RoadDataset

from inspect import signature
from torch.utils.data import random_split
from .datasets.road_dataset import RoadDataset
from .datasets import road_transforms

from pathlib import Path
from typing import Dict, List
import torch
from torch.utils.data import ConcatDataset, random_split
from .datasets.road_dataset import RoadDataset

def _discover_episode_paths(root: str | Path = "drive_data") -> List[Path]:
    """
    Finds episode paths inside drive_data/. Supports directory episodes or single files (e.g., .npz).
    """
    root = Path(root)
    assert root.exists(), f"{root} not found. Did you unzip drive_data.zip in the repo root?"

    eps: List[Path] = []

    # Common layouts:
    # - drive_data/<episode_dir>/
    # - drive_data/<episode_file>.npz  (or similar)
    for p in sorted(root.iterdir()):
        if p.name.startswith("."):
            continue
        if p.is_dir():
            eps.append(p)
        elif p.is_file() and p.suffix.lower() in {".npz", ".pt", ".pth"}:
            eps.append(p)

    if not eps:
        raise RuntimeError(f"No episodes found under {root}. Check your dataset unzip step.")
    return eps

def _make_datasets(transform_pipeline: str = "state_only") -> Dict[str, torch.utils.data.Dataset]:
    """
    Build train/val datasets for the RoadDataset signature:
      RoadDataset(episode_path: str, transform_pipeline: str = 'default')
    """
    episodes = _discover_episode_paths("drive_data")

    # 90/10 split
    n = len(episodes)
    n_train = max(1, int(0.9 * n))
    train_eps = episodes[:n_train]
    val_eps   = episodes[n_train:] if n_train < n else episodes[-1:]  # ensure non-empty val

    # Build per-episode datasets
    train_parts = [
        RoadDataset(episode_path=str(ep), transform_pipeline=transform_pipeline)
        for ep in train_eps
    ]
    val_parts = [
        RoadDataset(episode_path=str(ep), transform_pipeline=transform_pipeline)
        for ep in val_eps
    ]

    # Concatenate
    train_ds = train_parts[0] if len(train_parts) == 1 else ConcatDataset(train_parts)
    val_ds   = val_parts[0]   if len(val_parts)   == 1 else ConcatDataset(val_parts)

    return {"train": train_ds, "val": val_ds}

def _collate_fn(batch):
    """
    Batch elements are dicts with keys like:
      - track_left:  (n_track, 2)
      - track_right: (n_track, 2)
      - waypoints:   (n_waypoints, 2)
      - waypoints_mask: (n_waypoints,)  [optional]
      - image: (3, 96, 128)  [if using ViT]
    We stack them if present.
    """
    out: Dict[str, Any] = {}
    keys = batch[0].keys()
    for k in keys:
        if isinstance(batch[0][k], torch.Tensor):
            out[k] = torch.stack([b[k] for b in batch], dim=0)
        else:
            out[k] = [b[k] for b in batch]
    return out


def _build_model(model_name: str) -> torch.nn.Module:
    if model_name not in MODEL_FACTORY:
        raise ValueError(f"Unknown model_name '{model_name}'. Options: {list(MODEL_FACTORY.keys())}")
    # Use defaults specified in models.py
    return MODEL_FACTORY[model_name]()


def _compute_loss(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor | None) -> torch.Tensor:
    """
    MSE over waypoints; honor optional boolean mask (valid target points).
    pred, target: (B, n_waypoints, 2)
    mask: (B, n_waypoints) or None
    """
    if mask is not None:
        # expand mask to x/y
        m = mask.unsqueeze(-1).float()  # (B, n_waypoints, 1)
        diff = (pred - target) * m
        denom = m.sum().clamp_min(1.0)
        return (diff ** 2).sum() / denom
    else:
        return nn.functional.mse_loss(pred, target)


def _epoch(model, loader, optimizer, device, train: bool) -> Dict[str, float]:
    model.train(train)
    loss_meter = 0.0
    n_samples = 0

    for batch in loader:
        # Move tensors
        for k, v in list(batch.items()):
            if isinstance(v, torch.Tensor):
                batch[k] = v.to(device, non_blocking=True)

        if train:
            optimizer.zero_grad(set_to_none=True)

        # Choose inputs based on model type
        if isinstance(model, (MODEL_FACTORY["mlp_planner"], MODEL_FACTORY["transformer_planner"])):
            pred = model(track_left=batch["track_left"], track_right=batch["track_right"])
        elif isinstance(model, MODEL_FACTORY["vit_planner"]):
            pred = model(image=batch["image"])
        else:
            # Fallback check via names
            name = type(model).__name__.lower()
            if "vit" in name:
                pred = model(image=batch["image"])
            else:
                pred = model(track_left=batch["track_left"], track_right=batch["track_right"])

        target = batch["waypoints"]
        mask = batch.get("waypoints_mask", None)

        loss = _compute_loss(pred, target, mask)
        bs = target.size(0)
        loss_meter += loss.item() * bs
        n_samples += bs

        if train:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()

    return {"loss": loss_meter / max(1, n_samples)}


def train(
    model_name: str = "mlp_planner",
    transform_pipeline: str = "state_only",
    num_workers: int = 4,
    lr: float = 1e-3,
    batch_size: int = 128,
    num_epoch: int = 40,
    weight_decay: float = 1e-4,
    save_best: bool = True,
) -> str:
    """
    Main training entrypoint used from Colab.
    Returns: path to saved model weights (.th)
    """
    device = _get_device()
    datasets = _make_datasets(transform_pipeline=transform_pipeline)

    train_loader = DataLoader(
        datasets["train"], batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True, collate_fn=_collate_fn
    )
    val_loader = DataLoader(
        datasets["val"], batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True, collate_fn=_collate_fn
    )

    model = _build_model(model_name).to(device)

    # AdamW is a solid default
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    best_val = math.inf
    best_path = ""

    for epoch in range(1, num_epoch + 1):
        train_stats = _epoch(model, train_loader, optimizer, device, train=True)
        val_stats   = _epoch(model, val_loader,   optimizer, device, train=False)

        print(f"[{epoch:03d}/{num_epoch}]  "
              f"train_loss={train_stats['loss']:.4f}  "
              f"val_loss={val_stats['loss']:.4f}")

        if save_best and val_stats["loss"] < best_val:
            best_val = val_stats["loss"]
            best_path = save_model(model)
            print(f"  ↳ saved: {best_path}")

    # Save final if best not requested
    if not save_best:
        best_path = save_model(model)
        print(f"  ↳ saved(final): {best_path}")

    return best_path


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model-name", type=str, default="mlp_planner",
                   choices=list(MODEL_FACTORY.keys()))
    p.add_argument("--transform-pipeline", type=str, default="state_only",
                   help="state_only (tracks), image_only (for ViT)")
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--num-epoch", type=int, default=40)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    args = p.parse_args()

    train(
        model_name=args.model_name,
        transform_pipeline=args.transform_pipeline,
        num_workers=args.num_workers,
        lr=args.lr,
        batch_size=args.batch_size,
        num_epoch=args.num_epoch,
        weight_decay=args.weight_decay,
    )


if __name__ == "__main__":
    main()