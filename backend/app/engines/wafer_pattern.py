"""
Engine 1: Wafer Map Pattern AI

Multi-label PyTorch Convolutional Neural Network on 32x32 3-channel wafer maps
(fail mask, valid-die mask, soft-bin group severity) with Grad-CAM spatial
attention overlays, paired with a statistical 2D stepper-field periodicity
detector for open-set repeating reticle defects.
"""
from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.stats import binomtest

SPATIAL_CLASSES = ["Edge-Ring", "Center", "Scratch", "Loc"]
ALL_CLASSES = ["Clean", "Edge-Ring", "Center", "Scratch", "Loc", "Reticle-Repeat"]

PATTERN_DESCRIPTIONS = {
    "Clean": {
        "plain_name": "Normal / Random Background",
        "cause_hint": "Only normal random defects observed.",
        "first_check": "No spatial issue detected.",
    },
    "Edge-Ring": {
        "plain_name": "Edge-Ring Pattern",
        "cause_hint": "Often caused by etch or deposition uniformity drift near the wafer perimeter.",
        "first_check": "Check etch/deposition chamber uniformity and recent preventive maintenance.",
    },
    "Center": {
        "plain_name": "Center Hotspot",
        "cause_hint": "Often linked to polishing (CMP) pad wear, spin-coat thickness, or center thermal profile.",
        "first_check": "Check CMP polisher tool health and thermal uniformity.",
    },
    "Scratch": {
        "plain_name": "Linear Scratch",
        "cause_hint": "Physical contact mark from wafer-handling robot end-effector or CMP slurry debris.",
        "first_check": "Inspect wafer handler alignment and CMP pad conditioning.",
    },
    "Loc": {
        "plain_name": "Local Cluster",
        "cause_hint": "Localized particle contamination on the wafer surface or prober chuck.",
        "first_check": "Clean prober chuck and review localized particle inspection logs.",
    },
    "Reticle-Repeat": {
        "plain_name": "Repeating Stepper Field Defect",
        "cause_hint": "A mask/reticle defect repeats at the exact same sub-position inside every exposure field.",
        "first_check": "Inspect lithography reticle / mask ID for particle or pellicle defect.",
    },
}


class ResBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, stride: int = 1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_ch)
        self.shortcut = nn.Sequential()
        if stride != 1 or in_ch != out_ch:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_ch),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = F.relu(self.bn1(self.conv1(x)), inplace=True)
        out = self.bn2(self.conv2(out))
        out = out + self.shortcut(x)
        return F.relu(out, inplace=True)


class WaferPatternCNN(nn.Module):
    """Lightweight multi-label CNN for wafer defect signature classification + Grad-CAM."""

    def __init__(self, num_classes: int = len(SPATIAL_CLASSES)):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(3, 24, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(24),
            nn.ReLU(inplace=True),
        )
        self.layer1 = ResBlock(24, 32, stride=2)   # 16x16
        self.layer2 = ResBlock(32, 64, stride=2)   # 8x8
        self.layer3 = ResBlock(64, 128, stride=2)  # 4x4 (target layer for Grad-CAM)
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.maxpool = nn.AdaptiveMaxPool2d(1)
        self.head = nn.Sequential(
            nn.Linear(256, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(64, num_classes),
        )

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feat = self.forward_features(x)
        pooled = torch.cat([self.avgpool(feat).flatten(1), self.maxpool(feat).flatten(1)], dim=1)
        return self.head(pooled)


def _bin_group_severity(soft_bin: np.ndarray, passed: np.ndarray) -> np.ndarray:
    """Map pass/soft_bin into [0, 1] channel without hardcoding product-specific bin numbers."""
    sev = np.zeros(len(passed), dtype=np.float32)
    fail_mask = passed == 0
    if not np.any(fail_mask):
        return sev
    if soft_bin is None:
        sev[fail_mask] = 1.0
        return sev
    sb_vals = pd.to_numeric(pd.Series(soft_bin), errors="coerce").fillna(-1).to_numpy()
    # Deterministic, product-agnostic severity encoding from soft_bin bucket
    sev[fail_mask] = 0.4 + 0.6 * ((np.abs(sb_vals[fail_mask]) % 7) / 6.0)
    return sev


def wafer_to_tensor(
    dx: np.ndarray,
    dy: np.ndarray,
    passed: np.ndarray,
    soft_bin: np.ndarray | None = None,
    target_size: int = 32,
    grid_bounds: tuple[int, int, int, int] | None = None,
) -> np.ndarray:
    """
    Rasterize arbitrary die (x, y) coordinates onto a standardized (3, 32, 32) wafer tensor:
      ch 0: fail map (0..1)
      ch 1: valid die mask (0..1)
      ch 2: soft-bin severity (0..1)
    """
    if len(dx) == 0:
        return np.zeros((3, target_size, target_size), dtype=np.float32)

    if grid_bounds is None:
        xmin, xmax = int(np.min(dx)), int(np.max(dx))
        ymin, ymax = int(np.min(dy)), int(np.max(dy))
    else:
        xmin, xmax, ymin, ymax = grid_bounds

    span_x = max(1, xmax - xmin + 1)
    span_y = max(1, ymax - ymin + 1)
    span = max(span_x, span_y)

    # Center the wafer cleanly inside the square grid
    pad_x = (span - span_x) / 2.0
    pad_y = (span - span_y) / 2.0

    ix = np.clip(((dx - xmin + pad_x) / span * target_size).astype(int), 0, target_size - 1)
    iy = np.clip(((dy - ymin + pad_y) / span * target_size).astype(int), 0, target_size - 1)

    fail_grid = np.zeros((target_size, target_size), dtype=np.float32)
    valid_grid = np.zeros((target_size, target_size), dtype=np.float32)
    sev_grid = np.zeros((target_size, target_size), dtype=np.float32)
    counts = np.zeros((target_size, target_size), dtype=np.float32)

    fails = (1.0 - passed).astype(np.float32)
    sev = _bin_group_severity(soft_bin, passed)

    np.add.at(fail_grid, (iy, ix), fails)
    np.add.at(valid_grid, (iy, ix), 1.0)
    np.add.at(sev_grid, (iy, ix), sev)
    np.add.at(counts, (iy, ix), 1.0)

    nonzero = counts > 0
    fail_grid[nonzero] /= counts[nonzero]
    sev_grid[nonzero] /= counts[nonzero]
    valid_grid[nonzero] = 1.0

    return np.stack([fail_grid, valid_grid, sev_grid], axis=0)


def detect_repeating_reticle(
    dx: np.ndarray,
    dy: np.ndarray,
    passed: np.ndarray,
    site: np.ndarray | None = None,
) -> dict[str, Any]:
    """
    Product-agnostic 2D stepper-field periodicity detector.
    Tests candidate stepper field pitches (3x3, 4x4, 2x3, 3x2) to detect a sub-field
    position with statistically extreme repeat failure concentration while filtering out
    2x2 probe-card site parity effects.
    """
    if len(dx) < 60:
        return {"detected": False, "confidence": 0.0, "detail": None}

    fails = (passed == 0).astype(int)
    best = {"detected": False, "confidence": 0.0, "detail": None, "p_value": 1.0, "excess": 0.0}

    # Check 2x2 probe-site parity fail rates so 2x2 probe-card wear is never confused with a 4x4 mask defect
    parity_rates = {}
    for sx in (0, 1):
        for sy in (0, 1):
            m2 = ((dx % 2) == sx) & ((dy % 2) == sy)
            if m2.any():
                parity_rates[(sx, sy)] = float(fails[m2].mean())

    candidate_pitches = [(3, 3), (4, 4), (2, 3), (3, 2)]
    n_tests = sum(px * py for px, py in candidate_pitches)

    for px, py in candidate_pitches:
        rx = dx % px
        ry = dy % py
        for u in range(px):
            for v in range(py):
                mask = (rx == u) & (ry == v)
                n_pos = int(mask.sum())
                n_rest = len(fails) - n_pos
                if n_pos < 12 or n_rest < 40:
                    continue
                pos_fail = int(fails[mask].sum())
                rest_fail = int(fails[~mask].sum())
                pos_rate = pos_fail / n_pos
                rest_rate = rest_fail / n_rest
                excess = pos_rate - rest_rate
                if pos_rate < 0.45 or excess < 0.35:
                    continue

                # Guard against 2x2 probe-site harmonic inside 4x4: must exceed same-parity sibling cells
                if px % 2 == 0 and py % 2 == 0:
                    same_parity_mask = ((dx % 2) == (u % 2)) & ((dy % 2) == (v % 2)) & (~mask)
                    if same_parity_mask.sum() >= 15:
                        sp_rate = float(fails[same_parity_mask].mean())
                        if pos_rate - sp_rate < 0.25:
                            continue

                # If site column is provided, ignore if >85% of fails at this field position belong to 1 probe site
                if site is not None and pos_fail > 0:
                    fail_sites = site[mask & (fails == 1)]
                    if len(fail_sites) > 0:
                        _, counts = np.unique(fail_sites, return_counts=True)
                        if counts.max() / len(fail_sites) > 0.85 and len(np.unique(site)) >= 2:
                            # Check whether the field position spans multiple sites or only one site
                            all_pos_sites = np.unique(site[mask])
                            if len(all_pos_sites) == 1:
                                continue

                p0 = min(max(rest_rate, 0.02), 0.95)
                pval = float(binomtest(pos_fail, n_pos, p0, alternative="greater").pvalue) * n_tests
                if pval < 1e-6 and excess > best["excess"]:
                    conf = min(0.99, 0.65 + 0.34 * min(1.0, excess / 0.65))
                    best = {
                        "detected": True,
                        "confidence": round(conf, 3),
                        "p_value": float(pval),
                        "excess": float(excess),
                        "detail": {
                            "field_size": f"{px}x{py}",
                            "field_pos": f"({u}, {v})",
                            "pos_fail_rate": round(pos_rate, 3),
                            "other_fail_rate": round(rest_rate, 3),
                            "extra_failed_dies": int(round((pos_rate - rest_rate) * n_pos)),
                        },
                    }
    return best


def compute_gradcam(
    model: WaferPatternCNN,
    tensor_3x32x32: np.ndarray,
    class_idx: int,
    dx: np.ndarray,
    dy: np.ndarray,
    grid_bounds: tuple[int, int, int, int],
) -> np.ndarray:
    """Compute normalized [0, 1] Grad-CAM attention values mapped back to each die."""
    model.eval()
    x = torch.from_numpy(tensor_3x32x32[None, ...]).float()
    x.requires_grad_(False)

    activations = None
    gradients = None

    def fwd_hook(_m, _inp, out):
        nonlocal activations
        activations = out

    def bwd_hook(_m, _grad_in, grad_out):
        nonlocal gradients
        gradients = grad_out[0]

    h1 = model.layer3.register_forward_hook(fwd_hook)
    h2 = model.layer3.register_full_backward_hook(bwd_hook)
    try:
        model.zero_grad()
        logits = model(x)
        score = logits[0, class_idx]
        score.backward()

        weights = gradients.mean(dim=(2, 3), keepdim=True)
        cam = F.relu((weights * activations).sum(dim=1, keepdim=True))
        cam_up = F.interpolate(cam, size=(32, 32), mode="bilinear", align_corners=False)[0, 0].detach().numpy()
        if cam_up.max() > 1e-8:
            cam_up = cam_up / cam_up.max()
    finally:
        h1.remove()
        h2.remove()

    xmin, xmax, ymin, ymax = grid_bounds
    span_x = max(1, xmax - xmin + 1)
    span_y = max(1, ymax - ymin + 1)
    span = max(span_x, span_y)
    pad_x = (span - span_x) / 2.0
    pad_y = (span - span_y) / 2.0
    ix = np.clip(((dx - xmin + pad_x) / span * 32).astype(int), 0, 31)
    iy = np.clip(((dy - ymin + pad_y) / span * 32).astype(int), 0, 31)
    return np.round(cam_up[iy, ix], 3)


class WaferPatternPredictor:
    """Loads the trained PyTorch WaferPatternCNN and runs batch inference + Grad-CAM."""

    def __init__(self, model_dir: str | Path):
        self.model_dir = Path(model_dir)
        self.model_path = self.model_dir / "wafer_cnn.pt"
        self.meta_path = self.model_dir / "wafer_model_meta.json"
        self.model = WaferPatternCNN(num_classes=len(SPATIAL_CLASSES))
        self.thresholds = {c: 0.50 for c in SPATIAL_CLASSES}
        self.metrics: dict[str, Any] = {}

        if self.model_path.exists():
            ckpt = torch.load(self.model_path, map_location="cpu", weights_only=True)
            if isinstance(ckpt, dict) and "state_dict" in ckpt:
                self.model.load_state_dict(ckpt["state_dict"])
                if "thresholds" in ckpt:
                    self.thresholds.update(ckpt["thresholds"])
            else:
                self.model.load_state_dict(ckpt)
            self.model.eval()
            self.is_trained = True
        else:
            self.is_trained = False

        if self.meta_path.exists():
            with open(self.meta_path, "r", encoding="utf-8") as f:
                self.metrics = json.load(f)

    def predict_batch(
        self,
        die_df: pd.DataFrame,
    ) -> tuple[pd.DataFrame, dict[tuple[str, Any], dict[str, Any]]]:
        """
        Predict spatial defect signatures for all wafers in `die_df` (first-insertion view).
        Returns:
          - wafer_patterns_df: one row per (lot_id, wafer_id)
          - wafer_cache: dict mapping (lot_id, wafer_id) -> tensor + gradcam/reticle info
        """
        if die_df.empty or "die_x" not in die_df.columns or "die_y" not in die_df.columns:
            return pd.DataFrame(), {}

        # Global die coordinate bounds across the dataset
        dx_all = pd.to_numeric(die_df["die_x"], errors="coerce")
        dy_all = pd.to_numeric(die_df["die_y"], errors="coerce")
        valid_coords = dx_all.notna() & dy_all.notna()
        if not valid_coords.any():
            return pd.DataFrame(), {}

        bounds = (
            int(dx_all[valid_coords].min()),
            int(dx_all[valid_coords].max()),
            int(dy_all[valid_coords].min()),
            int(dy_all[valid_coords].max()),
        )

        # Group wafers and build tensors
        tensors = []
        wafer_keys = []
        reticle_results = []
        die_counts = []
        fail_rates = []

        has_sb = "soft_bin" in die_df.columns
        has_site = "site" in die_df.columns
        grouped = die_df[valid_coords].groupby(["lot_id", "wafer_id"], sort=False)

        for (lot_id, wafer_id), grp in grouped:
            dx = grp["die_x"].to_numpy(dtype=int)
            dy = grp["die_y"].to_numpy(dtype=int)
            passed = grp["pass"].to_numpy(dtype=int)
            sb = grp["soft_bin"].to_numpy() if has_sb else None
            site_arr = grp["site"].to_numpy() if has_site else None

            t = wafer_to_tensor(dx, dy, passed, sb, target_size=32, grid_bounds=bounds)
            tensors.append(t)
            wafer_keys.append((str(lot_id), wafer_id))
            reticle_results.append(detect_repeating_reticle(dx, dy, passed, site=site_arr))
            die_counts.append(len(dx))
            fail_rates.append(float((passed == 0).mean()))

        if not tensors:
            return pd.DataFrame(), {}

        batch_np = np.stack(tensors, axis=0)
        probs = np.zeros((len(tensors), len(SPATIAL_CLASSES)), dtype=np.float32)

        if self.is_trained:
            self.model.eval()
            with torch.no_grad():
                for start in range(0, len(batch_np), 256):
                    xb = torch.from_numpy(batch_np[start:start + 256]).float()
                    logits = self.model(xb)
                    probs[start:start + 256] = torch.sigmoid(logits).numpy()

        rows = []
        cache: dict[tuple[str, Any], dict[str, Any]] = {}

        for i, (lot_id, wafer_id) in enumerate(wafer_keys):
            p_vec = probs[i]
            detected = []
            prob_map = {cls: round(float(p_vec[j]), 3) for j, cls in enumerate(SPATIAL_CLASSES)}

            for j, cls in enumerate(SPATIAL_CLASSES):
                if p_vec[j] >= self.thresholds.get(cls, 0.50):
                    detected.append((cls, float(p_vec[j])))

            ret_info = reticle_results[i]
            prob_map["Reticle-Repeat"] = ret_info["confidence"]
            if ret_info["detected"]:
                detected.append(("Reticle-Repeat", ret_info["confidence"]))

            # Sort detected patterns by confidence descending
            detected.sort(key=lambda x: x[1], reverse=True)
            if not detected:
                primary = "Clean"
                all_labels = ["Clean"]
                top_conf = round(float(1.0 - max(p_vec.max(), ret_info["confidence"])), 3)
            else:
                primary = detected[0][0]
                all_labels = [d[0] for d in detected]
                top_conf = round(detected[0][1], 3)

            # Novelty / unusual pattern check (high failure rate not explained by known classes)
            unexplained_drop = (fail_rates[i] > 0.22) and (primary == "Clean")
            if unexplained_drop:
                primary = "Unusual Pattern"
                all_labels = ["Unusual Pattern"]

            # Estimate excess dies lost vs ~9.3% typical baseline fail rate (or dataset baseline)
            est_pattern_lost_dies = 0
            if primary != "Clean":
                if reticle_results[i]["detected"] and primary == "Reticle-Repeat":
                    est_pattern_lost_dies = reticle_results[i]["detail"]["extra_failed_dies"]
                else:
                    baseline_rate = 0.095
                    est_pattern_lost_dies = max(0, int(round((fail_rates[i] - baseline_rate) * die_counts[i])))

            desc = PATTERN_DESCRIPTIONS.get(
                primary,
                {
                    "plain_name": "Unusual Spatial Loss",
                    "cause_hint": "Yield dropped sharply with a spatial pattern that does not match standard classes.",
                    "first_check": "Review wafer fail map manually and check recent tool/lot history.",
                },
            )

            rows.append(
                {
                    "lot_id": lot_id,
                    "wafer_id": wafer_id,
                    "primary_pattern": primary,
                    "pattern_labels": "|".join(all_labels),
                    "pattern_confidence": top_conf,
                    "plain_pattern_name": desc["plain_name"],
                    "cause_hint": desc["cause_hint"],
                    "first_check": desc["first_check"],
                    "est_pattern_lost_dies": est_pattern_lost_dies,
                    "first_pass_fail_rate": round(fail_rates[i], 4),
                    "class_probs": prob_map,
                }
            )
            cache[(lot_id, wafer_id)] = {
                "tensor": batch_np[i],
                "bounds": bounds,
                "prob_map": prob_map,
                "primary_pattern": primary,
                "reticle_detail": ret_info["detail"],
            }

        return pd.DataFrame(rows), cache

    def explain_wafer_gradcam(
        self,
        cache_entry: dict[str, Any],
        dx: np.ndarray,
        dy: np.ndarray,
    ) -> np.ndarray:
        """Return per-die Grad-CAM attention weights [0..1] for the wafer's primary pattern."""
        primary = cache_entry.get("primary_pattern", "Clean")
        if primary == "Reticle-Repeat" and cache_entry.get("reticle_detail"):
            # Exact periodic highlight for repeating stepper field defect
            field_sz = cache_entry["reticle_detail"]["field_size"]
            pos_str = cache_entry["reticle_detail"]["field_pos"]
            px, py = [int(v) for v in field_sz.split("x")]
            u, v = [int(c.strip()) for c in pos_str.strip("()").split(",")]
            return np.where(((dx % px) == u) & ((dy % py) == v), 1.0, 0.05)

        if not self.is_trained or primary not in SPATIAL_CLASSES:
            return np.zeros(len(dx), dtype=np.float32)

        cls_idx = SPATIAL_CLASSES.index(primary)
        return compute_gradcam(
            self.model,
            cache_entry["tensor"],
            cls_idx,
            dx,
            dy,
            cache_entry["bounds"],
        )
