"""Grad-CAM for the CNN component.

Grad-CAM highlights image regions that influenced the CNN's ASD score. It does NOT show that those regions are
clinically established ASD biomarkers, and it explains the CNN head -- not the final XGBoost decision.
"""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from ml import config
from ml.models import LAST_CONV_LAYER


class GradCamExtractor:
    """Single forward pass returning (cnn_features, P_cnn(ASD), heatmap)."""

    def __init__(self, cnn_model, last_conv_layer: str | None = None):
        import keras
        self.model = cnn_model
        name = last_conv_layer or LAST_CONV_LAYER.get(config.CNN_BACKBONE, "conv5_block3_out")
        try:
            conv = cnn_model.get_layer(name).output
        except ValueError:                                   # fall back: last 4-D activation
            conv = next(l.output for l in reversed(cnn_model.layers)
                        if len(getattr(l.output, "shape", ())) == 4)
        self.grad_model = keras.Model(cnn_model.inputs, [conv, cnn_model.get_layer(config.CNN_FEATURE_DIM_LAYER).output,
                                                         cnn_model.output])

    def run(self, batch: np.ndarray):
        import tensorflow as tf
        x = tf.convert_to_tensor(batch, dtype=tf.float32)
        with tf.GradientTape() as tape:
            conv, feats, prob = self.grad_model(x, training=False)
            p = prob[:, 0]
            # explain the class the CNN head predicts (ASD score, or its complement for Non-ASD)
            score = tf.where(p >= 0.5, p, 1.0 - p)
        grads = tape.gradient(score, conv)
        weights = tf.reduce_mean(grads, axis=(1, 2), keepdims=True)
        cam = tf.nn.relu(tf.reduce_sum(weights * conv, axis=-1))
        cam = cam.numpy()
        heatmaps = []
        for c in cam:
            m = c.max()
            heatmaps.append(c / m if m > 1e-12 else np.zeros_like(c))
        return feats.numpy(), p.numpy(), np.stack(heatmaps)


def overlay_heatmap(image_rgb: np.ndarray, heatmap: np.ndarray, alpha: float = 0.4):
    """Return (heatmap_rgb, overlay_rgb) at the image resolution (uint8)."""
    h, w = image_rgb.shape[:2]
    hm = cv2.resize(heatmap.astype(np.float32), (w, h), interpolation=cv2.INTER_CUBIC)
    hm = np.clip(hm, 0, 1)
    colored = cv2.cvtColor(cv2.applyColorMap((hm * 255).astype(np.uint8), cv2.COLORMAP_JET), cv2.COLOR_BGR2RGB)
    overlay = cv2.addWeighted(image_rgb.astype(np.uint8), 1 - alpha, colored, alpha, 0)
    return colored, overlay


def save_gradcam_outputs(image_rgb, heatmap, stem: str, out_dir: Path | None = None) -> dict:
    """Write original / heatmap / overlay PNGs plus a 3-panel figure; return their paths."""
    from PIL import Image
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out_dir = Path(out_dir or config.GRADCAM_DIR)
    out_dir.mkdir(parents=True, exist_ok=True)
    colored, overlay = overlay_heatmap(image_rgb, heatmap)
    paths = {"original": out_dir / f"{stem}_original.png", "heatmap": out_dir / f"{stem}_heatmap.png",
             "overlay": out_dir / f"{stem}_overlay.png", "figure": out_dir / f"{stem}_gradcam.png"}
    Image.fromarray(image_rgb.astype(np.uint8)).save(paths["original"])
    Image.fromarray(colored).save(paths["heatmap"])
    Image.fromarray(overlay).save(paths["overlay"])
    fig, axes = plt.subplots(1, 3, figsize=(10, 3.6))
    for ax, im, t in zip(axes, (image_rgb, colored, overlay), ("Original image", "Grad-CAM heatmap", "Overlay")):
        ax.imshow(im.astype(np.uint8)); ax.set_title(t); ax.axis("off")
    fig.tight_layout(); fig.savefig(paths["figure"], dpi=130); plt.close(fig)
    return {k: str(v) for k, v in paths.items()}
