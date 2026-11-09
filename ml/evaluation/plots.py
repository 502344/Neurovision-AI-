"""Matplotlib plots. All plots are generated from real predictions/histories."""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import precision_recall_curve, roc_curve


def plot_training_history(history: dict, path, title="Training history"):
    keys = [k for k in history if not k.startswith("val_")]
    cols = 2
    rows = int(np.ceil(len(keys) / cols)) or 1
    fig, axes = plt.subplots(rows, cols, figsize=(10, 3.2 * rows), squeeze=False)
    for ax, k in zip(axes.ravel(), keys):
        ax.plot(history[k], label="train")
        if f"val_{k}" in history:
            ax.plot(history[f"val_{k}"], label="validation")
        ax.set_title(k); ax.set_xlabel("epoch"); ax.grid(alpha=.3); ax.legend()
    for ax in axes.ravel()[len(keys):]:
        ax.axis("off")
    fig.suptitle(title); fig.tight_layout(); fig.savefig(path, dpi=150); plt.close(fig)


def plot_confusion_matrix(cm, path, labels=("Non-ASD", "ASD"), title="Confusion matrix (test set)"):
    cm = np.asarray(cm)
    fig, ax = plt.subplots(figsize=(4.6, 4))
    ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(len(labels)), labels); ax.set_yticks(range(len(labels)), labels)
    ax.set_xlabel("Predicted"); ax.set_ylabel("True"); ax.set_title(title)
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, int(cm[i, j]), ha="center", va="center",
                    color="white" if cm[i, j] > cm.max() / 2 else "black")
    fig.tight_layout(); fig.savefig(path, dpi=150); plt.close(fig)


def plot_roc_curves(curves: dict, path, title="ROC curves (test set)"):
    """curves: {model_name: (y_true, y_prob)}"""
    fig, ax = plt.subplots(figsize=(5.6, 5))
    for name, (yt, yp) in curves.items():
        if len(np.unique(yt)) < 2:
            continue
        fpr, tpr, _ = roc_curve(yt, yp)
        from sklearn.metrics import auc
        ax.plot(fpr, tpr, label=f"{name} (AUC={auc(fpr, tpr):.3f})")
    ax.plot([0, 1], [0, 1], "k--", alpha=.4)
    ax.set_xlabel("False positive rate"); ax.set_ylabel("True positive rate"); ax.set_title(title)
    ax.legend(fontsize=8, loc="lower right"); ax.grid(alpha=.3)
    fig.tight_layout(); fig.savefig(path, dpi=150); plt.close(fig)


def plot_pr_curves(curves: dict, path, title="Precision-recall curves (test set)"):
    fig, ax = plt.subplots(figsize=(5.6, 5))
    for name, (yt, yp) in curves.items():
        if len(np.unique(yt)) < 2:
            continue
        p, r, _ = precision_recall_curve(yt, yp)
        ax.plot(r, p, label=name)
    ax.set_xlabel("Recall"); ax.set_ylabel("Precision"); ax.set_title(title)
    ax.legend(fontsize=8); ax.grid(alpha=.3)
    fig.tight_layout(); fig.savefig(path, dpi=150); plt.close(fig)


def plot_feature_importance(importances, path, top=30, title="Random Forest feature importance"):
    order = np.argsort(importances)[::-1][:top]
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.barh([f"f{int(i)}" for i in order][::-1], importances[order][::-1], color="#2f6f8f")
    ax.set_xlabel("Gini importance"); ax.set_title(title)
    fig.tight_layout(); fig.savefig(path, dpi=150); plt.close(fig)


def plot_topk_sweep(rows, path):
    """rows: list of dict(k, val_auc)."""
    ks = [r["k"] for r in rows]; aucs = [r["val_roc_auc"] for r in rows]
    fig, ax = plt.subplots(figsize=(5.5, 3.8))
    ax.plot(ks, aucs, "o-"); ax.set_xscale("log")
    ax.set_xlabel("TOP_K_FEATURES"); ax.set_ylabel("Validation ROC-AUC"); ax.grid(alpha=.3)
    ax.set_title("Top-K sweep (validation set only)")
    fig.tight_layout(); fig.savefig(path, dpi=150); plt.close(fig)
