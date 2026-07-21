"""YOLOv8-seg based ROI localization for eyelid/tongue/palm photos.

Each modality has a trained single-class YOLOv8n-seg model that localizes the
relevant body part in an uploaded photo. This is a best-effort preprocessing
step: if a model fails to find anything (or errors out), callers fall back to
the raw, uncropped photo -- detection is never required for prediction to
proceed. By default nothing about detection is shown in the UI, only the
resulting cropped photo; an optional debug/demo view (see app.py) can draw the
detected box for illustration without changing what's actually fed to the model.
"""

import os

import numpy as np
from PIL import ImageDraw
from ultralytics import YOLO

_SEG_DIR = os.path.dirname(os.path.abspath(__file__))

YOLO_WEIGHTS = {
    "eyelid": os.path.join(_SEG_DIR, "yolo_weights", "eyelid.pt"),
    "tongue": os.path.join(_SEG_DIR, "yolo_weights", "tongue.pt"),
    "palm": os.path.join(_SEG_DIR, "yolo_weights", "palm.pt"),
}


def load_yolo_models(device="cpu"):
    """Loads all available YOLO weights, skipping any whose file is missing.
    Returns dict modality_key -> ultralytics.YOLO instance."""
    models = {}
    for key, path in YOLO_WEIGHTS.items():
        if os.path.isfile(path):
            try:
                model = YOLO(path)
                model.to(device)
                models[key] = model
            except Exception:
                pass
    return models


def detect_box(model, pil_img, conf=0.25):
    """Runs detection and returns (x1, y1, x2, y2, confidence) of the highest-
    confidence box, or None if nothing was detected or anything went wrong."""
    try:
        results = model.predict(np.asarray(pil_img.convert("RGB")), conf=conf, verbose=False)
    except Exception:
        return None

    if not results:
        return None
    boxes = results[0].boxes
    if boxes is None or len(boxes) == 0:
        return None

    confs = boxes.conf.detach().cpu().numpy()
    best_idx = int(confs.argmax())
    x1, y1, x2, y2 = boxes.xyxy[best_idx].detach().cpu().numpy().tolist()
    return (x1, y1, x2, y2, float(confs[best_idx]))


def pad_box(box, img_size, pad_frac=0.10):
    """Expands a box by pad_frac on each side for context, clipped to img_size."""
    x1, y1, x2, y2 = box
    w, h = img_size
    box_w, box_h = x2 - x1, y2 - y1
    pad_x, pad_y = box_w * pad_frac, box_h * pad_frac
    x1 = max(0, int(round(x1 - pad_x)))
    y1 = max(0, int(round(y1 - pad_y)))
    x2 = min(w, int(round(x2 + pad_x)))
    y2 = min(h, int(round(y2 + pad_y)))
    return x1, y1, x2, y2


def crop_to_box(pil_img, box):
    x1, y1, x2, y2 = box
    if x2 <= x1 or y2 <= y1:
        return None
    return pil_img.convert("RGB").crop((x1, y1, x2, y2))


def detect_and_crop(model, pil_img, conf=0.25, pad_frac=0.10):
    """Convenience wrapper: detect + pad + crop in one call. Returns None on any
    detection failure -- callers should fall back to the raw image in that case."""
    detection = detect_box(model, pil_img, conf=conf)
    if detection is None:
        return None
    x1, y1, x2, y2, _conf = detection
    padded = pad_box((x1, y1, x2, y2), pil_img.size, pad_frac)
    return crop_to_box(pil_img, padded)


def draw_box(pil_img, box, color=(255, 60, 60), width=6):
    """Returns a copy of pil_img with the given (x1,y1,x2,y2) box drawn on it --
    purely for illustration, never fed into the model."""
    img = pil_img.convert("RGB").copy()
    draw = ImageDraw.Draw(img)
    draw.rectangle(list(box), outline=color, width=width)
    return img
