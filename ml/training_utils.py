"""Shared two-stage training routine (frozen backbone -> optional fine-tuning).

Only train and validation splits are used here. The TEST split is never touched by training scripts.
"""
from __future__ import annotations

import io
import json
import sys
from contextlib import redirect_stdout
from pathlib import Path

import numpy as np

from ml import config
from ml.data import load_split, make_dataset
from ml.evaluation.metrics import binary_metrics
from ml.evaluation.plots import plot_training_history
from ml.preprocessing.dataset_validation import load_manifest


def require_manifest():
    try:
        return load_manifest()
    except FileNotFoundError as exc:
        print(f"[train] {exc}")
        sys.exit(1)


def compute_class_weights(y) -> dict:
    y = np.asarray(y).astype(int)
    n, k = len(y), 2
    counts = np.bincount(y, minlength=2).astype(float)
    if (counts == 0).any():
        return {0: 1.0, 1: 1.0}
    return {c: float(n / (k * counts[c])) for c in (0, 1)}


def _compile(model, lr):
    import keras
    model.compile(
        optimizer=keras.optimizers.Adam(lr),
        loss="binary_crossentropy",
        metrics=["accuracy", keras.metrics.Precision(name="precision"),
                 keras.metrics.Recall(name="recall"), keras.metrics.AUC(name="auc")])


def _merge(h1: dict, h2: dict | None) -> dict:
    if not h2:
        return h1
    keys = set(h1) | set(h2)
    return {k: list(h1.get(k, [])) + list(h2.get(k, [])) for k in keys}


def _callbacks(best_path: Path, patience=8):
    import keras
    return [
        keras.callbacks.ModelCheckpoint(str(best_path), monitor="val_loss", save_best_only=True),
        keras.callbacks.EarlyStopping(monitor="val_loss", patience=patience, restore_best_weights=True),
        keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=3, min_lr=1e-7),
    ]


def train_classifier(model, *, name: str, best_path: Path, frozen_layer_names=None,
                     last_layers_to_unfreeze: int | None = None, generic_plot: str | None = None) -> dict:
    """Train ``model`` on manifest train split, validate on validation split, save best model."""
    import keras

    config.ensure_dirs()
    config.set_seeds()
    manifest = require_manifest()
    tr, va = load_split(manifest, "train"), load_split(manifest, "validation")
    if tr.empty or va.empty or tr["label_idx"].nunique() < 2:
        print("[train] Need both classes in the training split and a non-empty validation split.")
        sys.exit(1)

    train_ds = make_dataset(tr["image_path"], tr["label_idx"], shuffle=True)
    val_ds = make_dataset(va["image_path"], va["label_idx"])
    cw = compute_class_weights(tr["label_idx"])
    print(f"[{name}] train={len(tr)} val={len(va)} class_weight={cw}")

    # Stage 1: frozen pretrained backbone (or ViT backbone), train the head
    frozen = set(frozen_layer_names or [])
    for layer in model.layers:
        if layer.name in frozen:
            layer.trainable = False
    if hasattr(model, "vit_backbone"):
        model.vit_backbone.trainable = False
    _compile(model, config.LEARNING_RATE)
    h1 = model.fit(train_ds, validation_data=val_ds, epochs=config.EPOCHS, class_weight=cw,
                   callbacks=_callbacks(best_path), verbose=2).history

    # Stage 2 (optional): fine-tune the top layers at a much lower learning rate
    h2 = None
    if config.FINE_TUNE_EPOCHS > 0 and config.USE_PRETRAINED:
        n = last_layers_to_unfreeze if last_layers_to_unfreeze is not None else config.FINE_TUNE_LAST_LAYERS
        if hasattr(model, "vit_backbone"):
            model.vit_backbone.trainable = True
        else:
            backbone_layers = [l for l in model.layers if l.name in frozen]
            for layer in backbone_layers[-n:]:
                if not isinstance(layer, keras.layers.BatchNormalization):   # keep BN statistics frozen
                    layer.trainable = True
        _compile(model, config.FINE_TUNE_LEARNING_RATE)
        h2 = model.fit(train_ds, validation_data=val_ds, epochs=config.FINE_TUNE_EPOCHS, class_weight=cw,
                       callbacks=_callbacks(best_path, patience=5), verbose=2).history

    history = {k: [float(x) for x in v] for k, v in _merge(h1, h2).items()}
    best = keras.models.load_model(str(best_path), compile=False) if Path(best_path).exists() else model
    if not Path(best_path).exists():
        model.save(str(best_path))

    # validation-set metrics from real predictions (test set is NOT used here)
    val_prob = best.predict(val_ds, verbose=0).ravel()
    val_metrics = binary_metrics(va["label_idx"].values, val_prob, config.DECISION_THRESHOLD)

    (config.RESULTS_DIR / f"{name}_history.json").write_text(json.dumps(history, indent=2))
    plot_training_history(history, config.RESULTS_DIR / f"training_history_{name}.png", f"{name} training history")
    if generic_plot:
        plot_training_history(history, config.RESULTS_DIR / generic_plot, f"{name} training history")
    buf = io.StringIO()
    with redirect_stdout(buf):
        model.summary(line_length=110)
    (config.RESULTS_DIR / f"{name}_architecture.txt").write_text(buf.getvalue())
    record = {"model": name, "params": int(model.count_params()), "config": config.as_dict(),
              "train_size": int(len(tr)), "val_size": int(len(va)), "class_weight": cw,
              "epochs_run": len(history.get("loss", [])), "validation_metrics": val_metrics,
              "split_strategy_file": str(config.SPLIT_MANIFEST)}
    (config.RESULTS_DIR / f"{name}_training_record.json").write_text(json.dumps(record, indent=2, default=str))
    print(f"[{name}] saved -> {best_path}\n[{name}] validation metrics: {json.dumps(val_metrics)}")
    return record
