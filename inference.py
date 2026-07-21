"""Checkpoint loading and single-sample prediction, including optional TTA."""

import json
import os

import torch
import torch.nn.functional as F
from torchvision.transforms import functional as TF

from model import (
    ALL_MODALITY_KEYS,
    IMAGE_MODALITY_KEYS,
    TABULAR_MODALITY_KEY,
    HbSensorFusionNet,
    fuse_predictions,
)
from preprocessing import prepare_modality, zero_modality

TTA_ROTATION_DEG = 8.0

_APP_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_ENCODING_MAP_PATH = os.path.join(_APP_DIR, "encoding_map.json")


def load_encoding_map(path=DEFAULT_ENCODING_MAP_PATH):
    with open(path) as f:
        return json.load(f)


def load_model_and_checkpoint(checkpoint_path, device="cpu"):
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model = HbSensorFusionNet().to(device)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    return model, ckpt


def build_batch(images, demographics, ckpt, device="cpu", yolo_models=None):
    """images: dict modality_key -> PIL.Image or None.
    demographics: dict with age, weight_kg, height_cm, race_idx, complexion_idx, gender_idx.
    ckpt: loaded checkpoint dict (for normalization stats).
    yolo_models: optional dict modality_key -> ultralytics.YOLO, used to auto-crop
    to the detected body part before the rest of preprocessing (see
    preprocessing.prepare_modality for fallback behavior on failed detection).
    Returns (batch, presence, display_images). display_images maps modality_key
    -> 512x512x3 uint8 RGB array (or None if absent) as actually fed to the
    model, for Grad-CAM overlay.
    """
    extra_feat_stats = ckpt["extra_feat_stats"]
    yolo_models = yolo_models or {}

    batch = {}
    presence = {}
    display_images = {}
    for key in IMAGE_MODALITY_KEYS:
        pil_img = images.get(key)
        if pil_img is not None:
            img_t, extra_t, display_rgb = prepare_modality(
                key, pil_img, extra_feat_stats, yolo_model=yolo_models.get(key)
            )
            presence[key] = 1.0
        else:
            img_t, extra_t, display_rgb = zero_modality(key)
            presence[key] = 0.0
        display_images[key] = display_rgb
        batch[f"img_{key}"] = img_t.unsqueeze(0).to(device)
        batch[f"extra_feats_{key}"] = extra_t.unsqueeze(0).to(device)

    age = (demographics["age"] - ckpt["age_mean"]) / ckpt["age_std"]
    weight = (demographics["weight_kg"] - ckpt["weight_mean"]) / ckpt["weight_std"]
    height = (demographics["height_cm"] - ckpt["height_mean"]) / ckpt["height_std"]
    batch["cont_feats"] = torch.tensor([[age, weight, height]], dtype=torch.float32, device=device)
    batch["race_idx"] = torch.tensor([demographics["race_idx"]], dtype=torch.long, device=device)
    batch["complexion_idx"] = torch.tensor([demographics["complexion_idx"]], dtype=torch.long, device=device)
    batch["gender_idx"] = torch.tensor([demographics["gender_idx"]], dtype=torch.long, device=device)

    presence[TABULAR_MODALITY_KEY] = 1.0
    presence_t = {k: torch.tensor([v], dtype=torch.float32, device=device) for k, v in presence.items()}
    return batch, presence_t, display_images


def _flip_batch(batch):
    b = dict(batch)
    for key in IMAGE_MODALITY_KEYS:
        b[f"img_{key}"] = torch.flip(batch[f"img_{key}"], [3])
    return b


def _rotate_batch(batch, deg):
    b = dict(batch)
    for key in IMAGE_MODALITY_KEYS:
        b[f"img_{key}"] = TF.rotate(batch[f"img_{key}"], deg)
    return b


@torch.no_grad()
def predict(model, batch, presence, ckpt, use_tta=True):
    """Returns dict with fused hb prediction (g/dL), fused std, and per-modality breakdown."""
    hb_mean, hb_std = ckpt["hb_mean"], ckpt["hb_std"]

    if use_tta:
        views = [batch, _flip_batch(batch), _rotate_batch(batch, TTA_ROTATION_DEG), _rotate_batch(batch, -TTA_ROTATION_DEG)]
        mus_per_view = {key: [] for key in IMAGE_MODALITY_KEYS}
        lvs_per_view = {key: [] for key in IMAGE_MODALITY_KEYS}
        for view_batch in views:
            for key in IMAGE_MODALITY_KEYS:
                extra = view_batch.get(f"extra_feats_{key}")
                mu, lv = model.forward_image_modality(key, view_batch[f"img_{key}"], extra)
                mus_per_view[key].append(mu)
                lvs_per_view[key].append(lv)
        mu_tab, lv_tab = model.forward_tabular(
            batch["cont_feats"], batch["race_idx"], batch["complexion_idx"], batch["gender_idx"]
        )
        mus = {key: torch.stack(mus_per_view[key], dim=0).mean(dim=0) for key in IMAGE_MODALITY_KEYS}
        log_vars = {key: torch.stack(lvs_per_view[key], dim=0).mean(dim=0) for key in IMAGE_MODALITY_KEYS}
        mus[TABULAR_MODALITY_KEY] = mu_tab
        log_vars[TABULAR_MODALITY_KEY] = lv_tab
    else:
        mus, log_vars = model(batch)

    mu_fused_n, var_fused_n = fuse_predictions(mus, log_vars, presence)

    hb_pred = (mu_fused_n * hb_std + hb_mean).item()
    hb_pred_std = (var_fused_n.sqrt() * hb_std).item()

    per_modality = {}
    for key in ALL_MODALITY_KEYS:
        mu_hb = (mus[key] * hb_std + hb_mean).item()
        var_hb = ((F.softplus(log_vars[key]) + 1e-6) * (hb_std ** 2)).item()
        per_modality[key] = {
            "hb_pred": mu_hb,
            "hb_std": var_hb ** 0.5,
            "present": bool(presence[key].item()),
        }

    return {
        "hb_pred": hb_pred,
        "hb_pred_std": hb_pred_std,
        "per_modality": per_modality,
    }
