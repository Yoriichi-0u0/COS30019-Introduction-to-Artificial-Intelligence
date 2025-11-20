"""Utilities for loading image datasets for severity classification."""
from pathlib import Path
import tensorflow as tf


def load_image_datasets(base_dir: str = "dataset",
                        image_size: tuple[int, int] = (128, 128),
                        batch_size: int = 32,
                        shuffle: bool = True):
    """Load train, validation, and test datasets with normalization.

    Args:
        base_dir: Root directory containing ``train``, ``val``, and ``test`` folders.
        image_size: Image size to resize to (height, width).
        batch_size: Number of samples per batch.
        shuffle: Whether to shuffle the datasets.

    Returns:
        A tuple ``(train_ds, val_ds, test_ds)`` of ``tf.data.Dataset`` objects
        with images scaled to ``[0, 1]`` and RGB color channels preserved.
    """
    base_path = Path(base_dir)
    train_dir = base_path / "train"
    val_dir = base_path / "val"
    test_dir = base_path / "test"

    dataset_kwargs = dict(
        image_size=image_size,
        batch_size=batch_size,
        shuffle=shuffle,
        label_mode="int",
        color_mode="rgb",
    )

    train_ds = tf.keras.preprocessing.image_dataset_from_directory(train_dir, **dataset_kwargs)
    val_ds = tf.keras.preprocessing.image_dataset_from_directory(val_dir, **dataset_kwargs)
    test_ds = tf.keras.preprocessing.image_dataset_from_directory(test_dir, **dataset_kwargs)

    normalization = tf.keras.layers.Rescaling(1.0 / 255)

    def _with_normalization(ds: tf.data.Dataset) -> tf.data.Dataset:
        return ds.map(lambda x, y: (normalization(x), y), num_parallel_calls=tf.data.AUTOTUNE)

    train_ds = _with_normalization(train_ds).prefetch(tf.data.AUTOTUNE)
    val_ds = _with_normalization(val_ds).prefetch(tf.data.AUTOTUNE)
    test_ds = _with_normalization(test_ds).prefetch(tf.data.AUTOTUNE)

    return train_ds, val_ds, test_ds


__all__ = ["load_image_datasets"]
