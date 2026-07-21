"""Standalone reconstruction of HbSensorFusionNet (sensorfusion_v1_512) for inference.

Extracted from train_hb_sensorfusion_v1_512_fingertips_meta_fusion.ipynb.
Architecture must match the training notebook exactly since we load trained
state_dicts into it.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

# ---------------------------------------------------------------------------
# Config constants (must match training notebook)
# ---------------------------------------------------------------------------

IMAGE_MODALITIES = {
    "eyelid": {
        "extra_tabular_feats": [
            "eyelid_redness_index", "eyelid_lab_a_mean",
            "eyelid_red_yellow_ratio", "eyelid_hsv_s_mean",
        ],
        "apply_glare_color_constancy": False,
        "center_crop_fraction": None,
    },
    "tongue": {
        "extra_tabular_feats": [
            "tongue_lab_l_mean", "tongue_lab_a_mean", "tongue_lab_b_mean",
            "tongue_redness_index", "tongue_hsv_s_mean",
        ],
        "apply_glare_color_constancy": True,
        "center_crop_fraction": None,
    },
    "palm": {
        "extra_tabular_feats": [
            "palm_redness_index", "palm_red_yellow_ratio",
            "palm_rgb_r_mean", "palm_rgb_g_mean", "palm_rgb_b_mean",
        ],
        "apply_glare_color_constancy": False,
        "center_crop_fraction": 0.85,
    },
    "fingertips": {
        "extra_tabular_feats": [
            "fingertips_redness_index", "fingertips_red_yellow_ratio",
            "fingertips_rgb_r_mean", "fingertips_rgb_g_mean", "fingertips_rgb_b_mean",
        ],
        "apply_glare_color_constancy": False,
        "center_crop_fraction": 0.85,
    },
}

IMAGE_MODALITY_KEYS = list(IMAGE_MODALITIES.keys())
IMAGE_MODALITY_TO_IDX = {key: i for i, key in enumerate(IMAGE_MODALITY_KEYS)}
TABULAR_MODALITY_KEY = "tabular"
ALL_MODALITY_KEYS = IMAGE_MODALITY_KEYS + [TABULAR_MODALITY_KEY]

IMG_SIZE = 512

RACE_EMBED_DIM = 8
COMPLEXION_EMBED_DIM = 4
GENDER_EMBED_DIM = 2
HEAD_HIDDEN_DIM = 64
FUSION_PROJ_DIM = 64

# Confirmed from checkpoint embedding-weight shapes (fold_0).
NUM_RACE_CLASSES = 12
NUM_COMPLEXION_CLASSES = 3
NUM_GENDER_CLASSES = 2


def get_extra_feat_dim(spec):
    return len(spec["extra_tabular_feats"])


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

class SharedImageBackbone(nn.Module):
    """A single Swin-T instance, shared across every image modality's forward pass."""

    def __init__(self):
        super().__init__()
        base = models.swin_t(weights=None)
        self.features = base.features
        self.norm = base.norm
        self.permute = base.permute
        self.avgpool = base.avgpool
        self.flatten = base.flatten
        self.out_features = base.head.in_features  # 768

    def forward(self, x):
        x = self.features(x)
        x = self.norm(x)
        x = self.permute(x)
        x = self.avgpool(x)
        return self.flatten(x)


class ModalityHead(nn.Module):
    """Small independent (mu, log_var) head for one modality."""

    def __init__(self, in_dim, hidden_dim=HEAD_HIDDEN_DIM):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(in_dim, hidden_dim), nn.ReLU(True), nn.Dropout(0.3))
        self.out = nn.Linear(hidden_dim, 2)  # (mu, log_var)

    def forward(self, x):
        h = self.net(x)
        o = self.out(h)
        return o[:, 0], o[:, 1]


class HbSensorFusionNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.image_backbone = SharedImageBackbone()
        img_feat = self.image_backbone.out_features

        self.modality_embed = nn.Embedding(len(IMAGE_MODALITY_KEYS), img_feat)

        self.image_heads = nn.ModuleDict({
            key: ModalityHead(img_feat + get_extra_feat_dim(spec))
            for key, spec in IMAGE_MODALITIES.items()
        })

        self.cont_norm = nn.BatchNorm1d(3)
        self.race_emb = nn.Embedding(NUM_RACE_CLASSES, RACE_EMBED_DIM)
        self.complexion_emb = nn.Embedding(NUM_COMPLEXION_CLASSES, COMPLEXION_EMBED_DIM)
        self.gender_emb = nn.Embedding(NUM_GENDER_CLASSES, GENDER_EMBED_DIM)
        tab_in = 3 + RACE_EMBED_DIM + COMPLEXION_EMBED_DIM + GENDER_EMBED_DIM
        self.tabular_feat_extractor = nn.Sequential(nn.Linear(tab_in, FUSION_PROJ_DIM), nn.ReLU(True))
        self.tabular_head = ModalityHead(FUSION_PROJ_DIM)

    def forward_image_modality(self, key, img, extra_feats=None):
        feat = self.image_backbone(img)
        idx = torch.full((img.size(0),), IMAGE_MODALITY_TO_IDX[key], dtype=torch.long, device=img.device)
        feat = feat + self.modality_embed(idx)
        if extra_feats is not None:
            feat = torch.cat([feat, extra_feats], dim=1)
        return self.image_heads[key](feat)

    def forward_tabular(self, cont_feats, race_idx, complexion_idx, gender_idx):
        cont_normed = self.cont_norm(cont_feats)
        race_e = self.race_emb(race_idx)
        complexion_e = self.complexion_emb(complexion_idx)
        gender_e = self.gender_emb(gender_idx)
        feat = self.tabular_feat_extractor(torch.cat([cont_normed, race_e, complexion_e, gender_e], dim=1))
        return self.tabular_head(feat)

    def forward(self, batch):
        mus, log_vars = {}, {}
        for key, spec in IMAGE_MODALITIES.items():
            extra = batch.get(f"extra_feats_{key}")
            mu, lv = self.forward_image_modality(key, batch[f"img_{key}"], extra)
            mus[key], log_vars[key] = mu, lv
        mu_t, lv_t = self.forward_tabular(
            batch["cont_feats"], batch["race_idx"], batch["complexion_idx"], batch["gender_idx"]
        )
        mus[TABULAR_MODALITY_KEY], log_vars[TABULAR_MODALITY_KEY] = mu_t, lv_t
        return mus, log_vars


def fuse_predictions(mus, log_vars, presence, eps=1e-6):
    """Inverse-variance-weighted fusion over whichever modalities are present."""
    weighted_mu_sum, weight_sum = None, None
    for key in mus:
        var = F.softplus(log_vars[key]) + eps
        w = (1.0 / var) * presence[key]
        term = w * mus[key]
        weighted_mu_sum = term if weighted_mu_sum is None else weighted_mu_sum + term
        weight_sum = w if weight_sum is None else weight_sum + w
    mu_fused = weighted_mu_sum / weight_sum
    var_fused = 1.0 / weight_sum
    return mu_fused, var_fused
