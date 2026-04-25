"""
Data loading utilities for Medical MNIST using tf.data.

Medical MNIST contains 6 classes of 64×64 grayscale medical images:
  0 - AbdomenCT
  1 - BreastMRI
  2 - ChestCT
  3 - CXR (Chest X-Ray)
  4 - Hand
  5 - HeadCT

Dataset folder structure (after extraction from Kaggle):
    medical_mnist/
        AbdomenCT/   *.jpeg
        BreastMRI/   *.jpeg
        ChestCT/     *.jpeg
        CXR/         *.jpeg
        Hand/        *.jpeg
        HeadCT/      *.jpeg

Usage (local VS Code):
    ds_train, ds_val, ds_test, class_names = load_medical_mnist(
        data_dir='data/medical_mnist'   # relative to project root
    )
"""

import os
import tensorflow as tf


# ---------------------------------------------------------------------------
# Class mapping
# ---------------------------------------------------------------------------

CLASS_NAMES = ["AbdomenCT", "BreastMRI", "ChestCT", "CXR", "Hand", "HeadCT"]

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

IMAGE_SIZE  = (64, 64)   # Medical MNIST native resolution
NUM_CLASSES = 6
AUTOTUNE    = tf.data.AUTOTUNE


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------

def _parse_image(file_path: tf.Tensor, label: tf.Tensor):
    """Decode a JPEG / PNG file and normalise pixels to [0, 1]."""
    raw = tf.io.read_file(file_path)
    img = tf.image.decode_jpeg(raw, channels=1)          # grayscale
    img = tf.image.resize(img, IMAGE_SIZE)
    img = tf.cast(img, tf.float32) / 255.0
    return img, label


def _build_file_label_lists(data_dir: str):
    """
    Walk the dataset directory and collect (file_path, label_index) pairs.

    Assumes directory structure:
        data_dir/
            ClassName0/  img1.jpeg  img2.jpeg ...
            ClassName1/  ...
    """
    file_paths = []
    labels     = []

    for idx, class_name in enumerate(CLASS_NAMES):
        class_dir = os.path.join(data_dir, class_name)
        if not os.path.isdir(class_dir):
            continue
        for fname in os.listdir(class_dir):
            if fname.lower().endswith((".jpeg", ".jpg", ".png")):
                file_paths.append(os.path.join(class_dir, fname))
                labels.append(idx)

    return file_paths, labels


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def load_medical_mnist(
    data_dir: str,
    batch_size: int = 32,
    val_split: float = 0.15,
    test_split: float = 0.10,
    shuffle: bool = True,
    seed: int = 42,
    cache: bool = True,
):
    """
    Build tf.data train / validation / test pipelines for Medical MNIST.

    Args:
        data_dir   : Path to the extracted dataset root.
        batch_size : Mini-batch size.
        val_split  : Fraction of data reserved for validation.
        test_split : Fraction of data reserved for testing.
        shuffle    : Whether to shuffle the training set.
        seed       : Random seed for reproducibility.
        cache      : Whether to cache the dataset in memory.

    Returns:
        ds_train   : tf.data.Dataset (image, label) batched.
        ds_val     : tf.data.Dataset (image, label) batched.
        ds_test    : tf.data.Dataset (image, label) batched.
        class_names: List[str] mapping integer labels to class names.
    """
    file_paths, labels = _build_file_label_lists(data_dir)
    n_total = len(file_paths)

    if n_total == 0:
        raise ValueError(
            f"No images found in '{data_dir}'. "
            "Check that the directory contains sub-folders named: "
            + ", ".join(CLASS_NAMES)
        )

    # ---- Create a full dataset and shuffle once -------------------------
    ds_full = tf.data.Dataset.from_tensor_slices((file_paths, labels))
    if shuffle:
        ds_full = ds_full.shuffle(buffer_size=n_total, seed=seed)

    # ---- Compute split sizes -------------------------------------------
    n_test = int(n_total * test_split)
    n_val  = int(n_total * val_split)
    n_train = n_total - n_val - n_test

    ds_train = ds_full.take(n_train)
    ds_val   = ds_full.skip(n_train).take(n_val)
    ds_test  = ds_full.skip(n_train + n_val)

    # ---- Apply preprocessing, batching, prefetching --------------------
    def _pipeline(ds, repeat=False):
        ds = ds.map(_parse_image, num_parallel_calls=AUTOTUNE)
        if cache:
            ds = ds.cache()
        if repeat:
            ds = ds.repeat()
        ds = ds.batch(batch_size)
        ds = ds.prefetch(AUTOTUNE)
        return ds

    ds_train = _pipeline(ds_train, repeat=True)
    ds_val   = _pipeline(ds_val)
    ds_test  = _pipeline(ds_test)

    print(
        f"[DataLoader] {n_total} images found | "
        f"Train: {n_train} | Val: {n_val} | Test: {n_test}"
    )

    return ds_train, ds_val, ds_test, CLASS_NAMES


# ---------------------------------------------------------------------------
# Noise augmentation for denoising experiments
# ---------------------------------------------------------------------------

def add_noise(
    dataset: tf.data.Dataset,
    noise_factor: float = 0.3,
) -> tf.data.Dataset:
    """
    Return a new dataset where each (clean_image, label) pair becomes
    (noisy_image, clean_image) — suitable for training a denoising AE/VAE.

    Args:
        dataset      : tf.data.Dataset yielding (image, label).
        noise_factor : Standard deviation of Gaussian noise added (0–1 scale).

    Returns:
        tf.data.Dataset yielding (noisy_image, clean_image).
    """

    def _corrupt(image, label):
        noise  = tf.random.normal(tf.shape(image), stddev=noise_factor)
        noisy  = tf.clip_by_value(image + noise, 0.0, 1.0)
        return noisy, image   # (input=noisy, target=clean)

    return dataset.map(_corrupt, num_parallel_calls=AUTOTUNE)
