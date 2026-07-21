"""Image preprocessing and handcrafted color-biomarker features.

Mirrors the eval-time pipeline from the training notebook exactly, so that a
freshly uploaded photo is transformed the same way the model was trained on.
"""

from collections import OrderedDict

import cv2
import numpy as np
import torch
from PIL import Image
from torchvision import transforms
from torchvision.transforms import functional as TF

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]
IMG_SIZE = 512

_eval_tfms = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
])

COLOR_BIOMARKER_FEATURE_NAMES = [
    "rgb_r_mean", "rgb_r_std", "rgb_g_mean", "rgb_g_std", "rgb_b_mean", "rgb_b_std",
    "hsv_h_mean", "hsv_s_mean", "hsv_v_mean",
    "lab_l_mean", "lab_a_mean", "lab_b_mean",
    "redness_index", "red_yellow_ratio",
]

COLOR_BIOMARKER_SUBSETS = {
    "eyelid": ["redness_index", "lab_a_mean", "red_yellow_ratio", "hsv_s_mean"],
    "tongue": ["lab_l_mean", "lab_a_mean", "lab_b_mean", "redness_index", "hsv_s_mean"],
    "palm": ["redness_index", "red_yellow_ratio", "rgb_r_mean", "rgb_g_mean", "rgb_b_mean"],
    "fingertips": ["redness_index", "red_yellow_ratio", "rgb_r_mean", "rgb_g_mean", "rgb_b_mean"],
}


def center_crop_fraction(pil_img, fraction):
    w, h = pil_img.size
    return TF.center_crop(pil_img, [int(round(h * fraction)), int(round(w * fraction))])


def suppress_glare(pil_img, v_thresh=220, s_thresh=60, inpaint_radius=5):
    """Inpaint near-white specular highlights (e.g. saliva glare on tongue)."""
    arr = np.asarray(pil_img.convert("RGB"))
    hsv = cv2.cvtColor(arr, cv2.COLOR_RGB2HSV)
    v, s = hsv[..., 2], hsv[..., 1]
    glare_mask = ((v >= v_thresh) & (s <= s_thresh)).astype(np.uint8) * 255
    if not glare_mask.any():
        return pil_img
    bgr = cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
    inpainted_bgr = cv2.inpaint(bgr, glare_mask, inpaint_radius, cv2.INPAINT_TELEA)
    inpainted_rgb = cv2.cvtColor(inpainted_bgr, cv2.COLOR_BGR2RGB)
    return Image.fromarray(inpainted_rgb)


def shades_of_gray(pil_img, power=6):
    """Shades-of-Gray color constancy correction."""
    arr = np.asarray(pil_img).astype(np.float32)
    img_power = np.power(arr, power)
    rgb_avg = np.power(np.mean(img_power, axis=(0, 1)), 1.0 / power)
    gray = np.mean(rgb_avg)
    scale = gray / (rgb_avg + 1e-6)
    corrected = np.clip(arr * scale[None, None, :], 0, 255).astype(np.uint8)
    return Image.fromarray(corrected)


def compute_color_biomarkers(pil_img):
    """Handcrafted RGB/HSV/LAB color-biomarker features from a raw RGB image."""
    rgb = np.asarray(pil_img.convert("RGB")).astype(np.float64)
    r_ch, g_ch, b_ch = rgb[..., 0], rgb[..., 1], rgb[..., 2]

    hsv = np.asarray(pil_img.convert("HSV")).astype(np.float64)
    h_ch, s_ch, v_ch = hsv[..., 0], hsv[..., 1], hsv[..., 2]

    # Pillow's "LAB" conversion stores a*/b* as uint8, wrapping negative signed
    # values modulo 256 -- unwrap to a signed range before averaging.
    lab_raw = np.asarray(pil_img.convert("LAB")).astype(np.int16)
    l_ch = lab_raw[..., 0].astype(np.float64)
    a_ch = np.where(lab_raw[..., 1] > 127, lab_raw[..., 1] - 256, lab_raw[..., 1]).astype(np.float64)
    lab_b_ch = np.where(lab_raw[..., 2] > 127, lab_raw[..., 2] - 256, lab_raw[..., 2]).astype(np.float64)

    feats = OrderedDict()
    feats["rgb_r_mean"] = float(r_ch.mean())
    feats["rgb_r_std"] = float(r_ch.std())
    feats["rgb_g_mean"] = float(g_ch.mean())
    feats["rgb_g_std"] = float(g_ch.std())
    feats["rgb_b_mean"] = float(b_ch.mean())
    feats["rgb_b_std"] = float(b_ch.std())
    feats["hsv_h_mean"] = float(h_ch.mean())
    feats["hsv_s_mean"] = float(s_ch.mean())
    feats["hsv_v_mean"] = float(v_ch.mean())
    feats["lab_l_mean"] = float(l_ch.mean())
    feats["lab_a_mean"] = float(a_ch.mean())
    feats["lab_b_mean"] = float(lab_b_ch.mean())
    feats["redness_index"] = feats["lab_a_mean"]
    feats["red_yellow_ratio"] = feats["rgb_r_mean"] / (feats["rgb_g_mean"] + feats["rgb_b_mean"] + 1e-6)
    return feats


def _resize_preserve_aspect(pil_img, max_dim=IMG_SIZE):
    w, h = pil_img.size
    scale = max_dim / max(w, h)
    new_w, new_h = max(1, round(w * scale)), max(1, round(h * scale))
    return pil_img.resize((new_w, new_h))


def prepare_modality(key, pil_img, extra_feat_stats, yolo_model=None):
    """Given a raw PIL image for one modality, return
    (img_tensor[3,512,512], extra_feats_tensor, display_rgb_uint8[H,W,3]).

    Mirrors the notebook's eval-time per-modality pipeline order:
    ROI localization (YOLO if available, else fixed-fraction center-crop) ->
    compute color biomarkers on that crop -> glare-correction (tongue only) ->
    resize/normalize.

    If yolo_model is given, it's used to auto-crop to the detected body part
    (eyelid/tongue/palm); on any detection failure this silently falls back to
    the raw, uncropped image rather than the old fixed-fraction crop, since a
    failed detection carries no information about where to crop. Modalities
    with no YOLO model (fingertips) keep the original fixed-fraction crop.

    The model itself is fed a square 512x512 resize (forced aspect-ratio
    distortion, same as training). display_rgb_uint8 is instead an
    aspect-ratio-preserving resize of that same (cropped, glare-corrected)
    image, used only for Grad-CAM overlay display so heatmaps aren't stretched
    into a square -- the 16x16 CAM grid maps to fractional position, which
    Resize((512,512)) preserves regardless of aspect ratio, so this stays
    spatially correct while looking undistorted.
    """
    from model import IMAGE_MODALITIES

    spec = IMAGE_MODALITIES[key]
    img = pil_img.convert("RGB")

    if yolo_model is not None:
        from segmentation import detect_and_crop
        cropped = detect_and_crop(yolo_model, img)
        if cropped is not None:
            img = cropped
        # else: no detection -- keep the raw image, no fallback crop
    else:
        fraction = spec["center_crop_fraction"]
        if fraction is not None:
            img = center_crop_fraction(img, fraction)

    raw_biomarkers = compute_color_biomarkers(img)
    subset_cols = COLOR_BIOMARKER_SUBSETS[key]
    prefixed = {f"{key}_{name}": raw_biomarkers[name] for name in subset_cols}

    z_scored = []
    for col in spec["extra_tabular_feats"]:
        mean, std = extra_feat_stats[col]
        z_scored.append((prefixed[col] - mean) / std)
    extra_feats = torch.tensor(z_scored, dtype=torch.float32)

    if spec["apply_glare_color_constancy"]:
        img = suppress_glare(img)
        img = shades_of_gray(img)

    display_img = _resize_preserve_aspect(img, IMG_SIZE)
    display_rgb = np.asarray(display_img.convert("RGB")).astype(np.uint8)

    img_tensor = _eval_tfms(img)
    return img_tensor, extra_feats, display_rgb


def zero_modality(key):
    """Tensors to use for a modality the user did not upload a photo for."""
    from model import IMAGE_MODALITIES

    spec = IMAGE_MODALITIES[key]
    img_tensor = torch.zeros(3, IMG_SIZE, IMG_SIZE)
    extra_feats = torch.zeros(len(spec["extra_tabular_feats"]))
    return img_tensor, extra_feats, None
