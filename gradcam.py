"""Grad-CAM for the shared Swin-T image backbone.

HbSensorFusionNet has no single final conv layer like a plain CNN, so we
target the last spatial feature map produced by the backbone -- the output of
`image_backbone.features`, shape (B, H, W, C) with H=W=IMG_SIZE/32=16,
C=768 -- right before the final norm/permute/avgpool that collapses it to a
per-image feature vector. Gradients are taken w.r.t. that modality's own
predicted mean (mu), since the backbone and heads are otherwise independent
per modality (no cross-modality attention to account for).
"""

import cv2
import numpy as np
import torch
import torch.nn.functional as F


class GradCAM:
    def __init__(self, model):
        self.model = model
        self.activations = None
        self.gradients = None
        target_layer = model.image_backbone.features
        self._fwd_handle = target_layer.register_forward_hook(self._save_activation)
        self._bwd_handle = target_layer.register_full_backward_hook(self._save_gradient)

    def _save_activation(self, module, inp, out):
        self.activations = out

    def _save_gradient(self, module, grad_in, grad_out):
        self.gradients = grad_out[0]

    def remove(self):
        self._fwd_handle.remove()
        self._bwd_handle.remove()

    def compute(self, key, img_tensor, extra_feats):
        """img_tensor: (1,3,512,512). extra_feats: (1,D) or None.
        Returns a (16,16) numpy array normalized to [0,1]."""
        self.model.zero_grad(set_to_none=True)
        mu, _log_var = self.model.forward_image_modality(key, img_tensor, extra_feats)
        mu.sum().backward()

        weights = self.gradients.mean(dim=(1, 2), keepdim=True)  # (1,1,1,C), channels-last
        cam = F.relu((self.activations * weights).sum(dim=-1))  # (1,H,W)
        cam = cam.squeeze(0).detach().cpu().numpy()
        if cam.max() > 1e-8:
            cam = cam / cam.max()
        return cam


def overlay_heatmap(display_rgb_uint8, cam, alpha=0.45):
    """display_rgb_uint8: (H,W,3) uint8. cam: small (h,w) array in [0,1].
    Returns an (H,W,3) uint8 blended heatmap overlay."""
    h, w = display_rgb_uint8.shape[:2]
    cam_resized = cv2.resize(cam, (w, h), interpolation=cv2.INTER_CUBIC)
    cam_resized = np.clip(cam_resized, 0, 1)
    heatmap_bgr = cv2.applyColorMap((cam_resized * 255).astype(np.uint8), cv2.COLORMAP_JET)
    heatmap_rgb = cv2.cvtColor(heatmap_bgr, cv2.COLOR_BGR2RGB)
    overlay = display_rgb_uint8.astype(np.float32) * (1 - alpha) + heatmap_rgb.astype(np.float32) * alpha
    return np.clip(overlay, 0, 255).astype(np.uint8)


def compute_gradcam_overlays(model, batch, presence, display_images, modality_keys, alpha=0.45):
    """Returns dict modality_key -> (H,W,3) uint8 overlay, for each present modality
    in modality_keys. Must be called outside a torch.no_grad() context."""
    cam_engine = GradCAM(model)
    was_training = model.training
    model.eval()
    overlays = {}
    try:
        for key in modality_keys:
            if presence[key].item() == 0.0 or display_images.get(key) is None:
                continue
            img_tensor = batch[f"img_{key}"].clone().requires_grad_(True)
            extra_feats = batch.get(f"extra_feats_{key}")
            cam = cam_engine.compute(key, img_tensor, extra_feats)
            overlays[key] = overlay_heatmap(display_images[key], cam, alpha=alpha)
    finally:
        cam_engine.remove()
        model.zero_grad(set_to_none=True)
        model.train(was_training)
    return overlays
