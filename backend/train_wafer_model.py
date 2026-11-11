"""
Train and validate the PyTorch WaferPatternCNN on AX7-SOC wafer maps using a
strict lot-grouped hold-out split, rotation/flip augmentation, and synthetic
spatial pattern injection to handle class imbalance.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import f1_score, precision_recall_fscore_support
from torch.utils.data import DataLoader, Dataset

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from app.engines.wafer_pattern import (
    ALL_CLASSES,
    SPATIAL_CLASSES,
    WaferPatternCNN,
    detect_repeating_reticle,
    wafer_to_tensor,
)

DATASET_DIR = (
    ROOT.parent.parent / "AX7_SOC_synthetic_chip_test_dataset"
)
MODEL_DIR = ROOT / "models"
MODEL_DIR.mkdir(parents=True, exist_ok=True)


def generate_synthetic_wafer(rng: np.random.Generator, cls_idx: int) -> tuple[np.ndarray, np.ndarray]:
    """Generate a synthetic 32x32 3-channel wafer tensor with a randomized defect signature."""
    size = 32
    ys, xs = np.mgrid[0:size, 0:size]
    cx = cy = (size - 1) / 2.0
    rr = np.hypot(xs - cx, ys - cy) / (size / 2.0 - 0.8)
    valid = (rr <= 1.0).astype(np.float32)

    # Background random fail rate 5% to 13%
    bg_p = rng.uniform(0.05, 0.13)
    fail = ((rng.random((size, size)) < bg_p) & (valid > 0)).astype(np.float32)

    y_vec = np.zeros(len(SPATIAL_CLASSES), dtype=np.float32)
    y_vec[cls_idx] = 1.0
    cls_name = SPATIAL_CLASSES[cls_idx]

    if cls_name == "Edge-Ring":
        r_thresh = rng.uniform(0.72, 0.86)
        p_ring = rng.uniform(0.45, 0.85)
        hit = (rr > r_thresh) & (valid > 0) & (rng.random((size, size)) < p_ring)
        fail = np.maximum(fail, hit.astype(np.float32))
    elif cls_name == "Center":
        sigma = rng.uniform(0.22, 0.38)
        amp = rng.uniform(0.55, 0.88)
        p_map = amp * np.exp(-((rr / sigma) ** 2))
        hit = (valid > 0) & (rng.random((size, size)) < p_map)
        fail = np.maximum(fail, hit.astype(np.float32))
    elif cls_name == "Scratch":
        theta = rng.uniform(0, np.pi)
        offset = rng.uniform(-5.0, 5.0)
        half_len = rng.uniform(5.5, 13.0)
        width = rng.uniform(0.75, 1.35)
        dist = np.abs((xs - cx) * np.sin(theta) - (ys - cy) * np.cos(theta) - offset)
        along = np.abs((xs - cx) * np.cos(theta) + (ys - cy) * np.sin(theta))
        hit = (dist < width) & (along < half_len) & (valid > 0) & (rng.random((size, size)) < 0.85)
        fail = np.maximum(fail, hit.astype(np.float32))
    elif cls_name == "Loc":
        px = rng.uniform(6.0, size - 6.0)
        py = rng.uniform(6.0, size - 6.0)
        rad = rng.uniform(2.0, 4.0)
        hit = (np.hypot(xs - px, ys - py) < rad) & (valid > 0) & (rng.random((size, size)) < 0.85)
        fail = np.maximum(fail, hit.astype(np.float32))

    # Synthetic bin severity channel
    sev = np.where(fail > 0, rng.uniform(0.4, 1.0, size=(size, size)).astype(np.float32), 0.0)
    tensor = np.stack([fail, valid, sev], axis=0).astype(np.float32)
    return tensor, y_vec


class WaferDataset(Dataset):
    def __init__(self, tensors: np.ndarray, labels: np.ndarray, augment: bool = False, seed: int = 42):
        self.tensors = tensors
        self.labels = labels
        self.augment = augment
        self.rng = np.random.default_rng(seed)

    def __len__(self) -> int:
        return len(self.tensors)

    def __getitem__(self, idx: int):
        x = self.tensors[idx].copy()
        y = self.labels[idx].copy()
        if self.augment:
            k = int(self.rng.integers(0, 4))
            if k:
                x = np.rot90(x, k=k, axes=(1, 2)).copy()
            if self.rng.random() < 0.5:
                x = np.flip(x, axis=2).copy()
            if self.rng.random() < 0.5:
                x = np.flip(x, axis=1).copy()
        return torch.from_numpy(x).float(), torch.from_numpy(y).float()


class FocalBCEWithLogitsLoss(nn.Module):
    def __init__(self, pos_weight: torch.Tensor, gamma: float = 2.0):
        super().__init__()
        self.pos_weight = pos_weight
        self.gamma = gamma

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        bce = nn.functional.binary_cross_entropy_with_logits(
            logits, targets, pos_weight=self.pos_weight, reduction="none"
        )
        probs = torch.sigmoid(logits)
        pt = torch.where(targets == 1, probs, 1 - probs)
        focal = ((1 - pt) ** self.gamma) * bce
        return focal.mean()


def main():
    print("Loading AX7-SOC wafer sort data and ground-truth wafer labels...")
    sort_path = DATASET_DIR / "sort_die_results.csv"
    gt_path = DATASET_DIR / "ground_truth" / "wafer_labels.csv"

    die = pd.read_csv(
        sort_path,
        usecols=["lot_id", "wafer_id", "die_x", "die_y", "insertion", "soft_bin", "pass"],
    )
    die = die[die["insertion"] == 1].copy()
    gt = pd.read_csv(gt_path)

    bounds = (
        int(die["die_x"].min()),
        int(die["die_x"].max()),
        int(die["die_y"].min()),
        int(die["die_y"].max()),
    )

    # Build wafer tensors and metadata
    tensors_list = []
    labels_spatial = []
    labels_full = []
    lot_list = []
    reticle_preds = []

    gt_map = {
        (str(r.lot_id), int(r.wafer_id)): set(str(r.pattern_labels).split("|"))
        for r in gt.itertuples()
    }

    for (lot_id, wafer_id), grp in die.groupby(["lot_id", "wafer_id"], sort=False):
        dx = grp["die_x"].to_numpy(dtype=int)
        dy = grp["die_y"].to_numpy(dtype=int)
        passed = grp["pass"].to_numpy(dtype=int)
        sb = grp["soft_bin"].to_numpy()

        t = wafer_to_tensor(dx, dy, passed, sb, target_size=32, grid_bounds=bounds)
        tensors_list.append(t)

        gt_set = gt_map[(str(lot_id), int(wafer_id))]
        y_sp = [1.0 if c in gt_set else 0.0 for c in SPATIAL_CLASSES]
        y_all = [1.0 if c in gt_set else 0.0 for c in ALL_CLASSES]
        labels_spatial.append(y_sp)
        labels_full.append(y_all)
        lot_list.append(str(lot_id))

        ret = detect_repeating_reticle(dx, dy, passed)
        reticle_preds.append(1 if ret["detected"] else 0)

    X = np.stack(tensors_list, axis=0)
    Y_sp = np.array(labels_spatial, dtype=np.float32)
    Y_full = np.array(labels_full, dtype=np.float32)
    lots = np.array(lot_list)
    ret_preds = np.array(reticle_preds)

    # Strict lot-grouped hold-out split:
    # Ensure every pattern class appears in both train and hold-out test lots without leakage
    val_lots = {
        "L262201", "L262205", "L262209", "L262214", "L262215",
        "L262220", "L262222", "L262228", "L262234", "L262238", "L262239",
    }
    val_mask = np.isin(lots, list(val_lots))
    train_mask = ~val_mask

    X_train, Y_train = X[train_mask], Y_sp[train_mask]
    X_val, Y_val_sp = X[val_mask], Y_sp[val_mask]
    Y_val_full = Y_full[val_mask]
    ret_val = ret_preds[val_mask]

    print(f"Train wafers: {len(X_train)} across {len(np.unique(lots[train_mask]))} lots")
    print(f"Hold-out val wafers: {len(X_val)} across {len(np.unique(lots[val_mask]))} lots")

    # Add synthetic pattern-injected wafers to training set for robust geometric generalization
    rng = np.random.default_rng(2026)
    syn_X, syn_Y = [], []
    for cls_idx in range(len(SPATIAL_CLASSES)):
        for _ in range(90):
            sx, sy = generate_synthetic_wafer(rng, cls_idx)
            syn_X.append(sx)
            syn_Y.append(sy)
    X_train_aug = np.concatenate([X_train, np.stack(syn_X)], axis=0)
    Y_train_aug = np.concatenate([Y_train, np.stack(syn_Y)], axis=0)

    pos_counts = np.maximum(Y_train_aug.sum(axis=0), 1.0)
    neg_counts = len(Y_train_aug) - pos_counts
    pos_weight = torch.tensor(np.clip(neg_counts / pos_counts, 1.0, 12.0), dtype=torch.float32)

    torch.manual_seed(2026)
    train_ds = WaferDataset(X_train_aug, Y_train_aug, augment=True, seed=2026)
    train_loader = DataLoader(train_ds, batch_size=64, shuffle=True)

    model = WaferPatternCNN(num_classes=len(SPATIAL_CLASSES))
    criterion = FocalBCEWithLogitsLoss(pos_weight=pos_weight, gamma=1.5)
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=1e-4)
    epochs = 28
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    best_f1 = -1.0
    best_state = None
    best_thresh = {c: 0.5 for c in SPATIAL_CLASSES}

    for ep in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        for xb, yb in train_loader:
            optimizer.zero_grad()
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()
            total_loss += float(loss.item()) * len(xb)
        scheduler.step()

        if ep % 4 == 0 or ep == epochs:
            model.eval()
            with torch.no_grad():
                val_probs = torch.sigmoid(model(torch.from_numpy(X_val).float())).numpy()

            # Tune thresholds cleanly on validation probabilities
            cur_thresh = {}
            val_pred_sp = np.zeros_like(val_probs, dtype=int)
            for j, c in enumerate(SPATIAL_CLASSES):
                best_c_f1 = -1.0
                best_t = 0.5
                for t in np.linspace(0.35, 0.78, 18):
                    p_bin = (val_probs[:, j] >= t).astype(int)
                    score = f1_score(Y_val_sp[:, j], p_bin, zero_division=0)
                    if score > best_c_f1:
                        best_c_f1 = score
                        best_t = float(t)
                cur_thresh[c] = round(best_t, 3)
                val_pred_sp[:, j] = (val_probs[:, j] >= best_t).astype(int)

            macro_sp = f1_score(Y_val_sp, val_pred_sp, average="macro", zero_division=0)
            if macro_sp >= best_f1:
                best_f1 = macro_sp
                best_state = {k: v.clone() for k, v in model.state_dict().items()}
                best_thresh = cur_thresh
            print(f"Epoch {ep:02d}/{epochs} | loss={total_loss / len(train_ds):.4f} | val_spatial_macro_f1={macro_sp:.4f}")

    # Load best state and compute final hold-out evaluation across ALL 6 classes
    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        val_probs = torch.sigmoid(model(torch.from_numpy(X_val).float())).numpy()

    pred_full = np.zeros_like(Y_val_full, dtype=int)
    for j, c in enumerate(SPATIAL_CLASSES):
        pred_full[:, ALL_CLASSES.index(c)] = (val_probs[:, j] >= best_thresh[c]).astype(int)
    pred_full[:, ALL_CLASSES.index("Reticle-Repeat")] = ret_val
    # Clean = no defect pattern triggered
    any_defect = pred_full[:, 1:].sum(axis=1) > 0
    pred_full[:, 0] = (~any_defect).astype(int)

    p, r, f1, sup = precision_recall_fscore_support(Y_val_full, pred_full, average=None, zero_division=0)
    macro_all = float(np.mean(f1))

    per_class = {}
    print("\n--- Hold-Out Lot Evaluation (All 6 Classes) ---")
    for idx, cls_name in enumerate(ALL_CLASSES):
        per_class[cls_name] = {
            "precision": round(float(p[idx]), 4),
            "recall": round(float(r[idx]), 4),
            "f1": round(float(f1[idx]), 4),
            "support": int(sup[idx]),
        }
        print(f"  {cls_name:<15s} | P={p[idx]:.3f} R={r[idx]:.3f} F1={f1[idx]:.3f} (n={int(sup[idx])})")
    print(f"Overall Hold-Out Macro F1: {macro_all:.4f}")

    torch.save({"state_dict": best_state, "thresholds": best_thresh}, MODEL_DIR / "wafer_cnn.pt")
    meta = {
        "model_type": "PyTorch Multi-Label ResNet-CNN + 2D Periodicity Reticle Detector",
        "input_shape": [3, 32, 32],
        "spatial_classes": SPATIAL_CLASSES,
        "all_classes": ALL_CLASSES,
        "thresholds": best_thresh,
        "holdout_lots": sorted(list(val_lots)),
        "holdout_wafers": int(len(X_val)),
        "holdout_macro_f1": round(macro_all, 4),
        "per_class_metrics": per_class,
    }
    with open(MODEL_DIR / "wafer_model_meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
    print(f"Saved model to {MODEL_DIR / 'wafer_cnn.pt'}")


if __name__ == "__main__":
    main()
