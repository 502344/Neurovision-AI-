"""tf.data pipelines and array loaders built on the split manifest."""
from __future__ import annotations

import numpy as np
import pandas as pd

from ml import config
from ml.preprocessing.image_preprocessing import preprocess_image


def load_split(manifest: pd.DataFrame, split: str) -> pd.DataFrame:
    part = manifest[manifest["split"] == split].reset_index(drop=True)
    return part


def make_dataset(paths, labels=None, *, batch_size=None, shuffle=False, cache=True):
    """Lazily preprocess images (validate->crop->resize) into batches of uint8 RGB."""
    import tensorflow as tf

    size = config.IMAGE_SIZE
    batch_size = batch_size or config.BATCH_SIZE
    paths = np.asarray(paths, dtype=str)

    def _load(p):
        return preprocess_image(p.decode("utf-8") if isinstance(p, bytes) else str(p)).astype(np.uint8)

    def _map(p, *rest):
        img = tf.numpy_function(_load, [p], tf.uint8)
        img.set_shape((size, size, 3))
        img = tf.cast(img, tf.float32)
        return (img, *rest)

    tensors = (paths,) if labels is None else (paths, np.asarray(labels, dtype=np.float32))
    ds = tf.data.Dataset.from_tensor_slices(tensors).map(_map, num_parallel_calls=tf.data.AUTOTUNE)
    if cache:
        ds = ds.cache()
    if shuffle:
        ds = ds.shuffle(min(len(paths), 2048), seed=config.RANDOM_STATE, reshuffle_each_iteration=True)
    return ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)


def load_images_array(paths) -> np.ndarray:
    """Eagerly load preprocessed images as float32 (N, S, S, 3) in [0, 255]."""
    return np.stack([preprocess_image(p) for p in paths]).astype(np.float32)
