"""CNN + ViT feature fusion (configurable).

Methods: concat (default) | l2_concat (L2-normalise each stream first) | weighted_concat
Run:  python ml/fusion/feature_fusion.py
Saves features/fused_features.npy
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from ml import config  # noqa: E402


def _l2(x: np.ndarray) -> np.ndarray:
    return x / np.maximum(np.linalg.norm(x, axis=1, keepdims=True), 1e-12)


def fuse_features(cnn_features, vit_features, method: str | None = None, cnn_weight: float | None = None):
    """Fuse two (N, d1) / (N, d2) feature matrices into (N, d1 + d2)."""
    method = (method or config.FUSION_METHOD).lower()
    cnn = np.asarray(cnn_features, dtype=np.float32)
    vit = np.asarray(vit_features, dtype=np.float32)
    if cnn.ndim != 2 or vit.ndim != 2:
        raise ValueError("Features must be 2-D arrays (samples x dimensions).")
    if cnn.shape[0] != vit.shape[0]:
        raise ValueError(f"Sample count mismatch: CNN {cnn.shape[0]} vs ViT {vit.shape[0]}.")
    if method == "concat":
        return np.concatenate([cnn, vit], axis=1)
    if method == "l2_concat":
        return np.concatenate([_l2(cnn), _l2(vit)], axis=1)
    if method == "weighted_concat":
        w = config.FUSION_CNN_WEIGHT if cnn_weight is None else cnn_weight
        if not 0.0 <= w <= 1.0:
            raise ValueError("cnn_weight must be within [0, 1].")
        return np.concatenate([w * _l2(cnn), (1.0 - w) * _l2(vit)], axis=1)
    raise ValueError(f"Unknown fusion method '{method}'.")


def main():
    config.ensure_dirs()
    try:
        cnn = np.load(config.FEATURES_DIR / "cnn_features.npy")
        vit = np.load(config.FEATURES_DIR / "vit_features.npy")
    except FileNotFoundError as exc:
        print(f"[fusion] Missing feature file: {exc.filename}. Run the extraction scripts first.")
        sys.exit(1)
    fused = fuse_features(cnn, vit)
    np.save(config.FEATURES_DIR / "fused_features.npy", fused)
    print(f"[fusion] method={config.FUSION_METHOD} cnn={cnn.shape} vit={vit.shape} -> fused={fused.shape}")


if __name__ == "__main__":
    main()
