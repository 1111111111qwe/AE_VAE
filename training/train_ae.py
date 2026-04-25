"""
train_ae.py — Training script for the Autoencoder (AE).

Usage (local):
    python train_ae.py

Usage (Colab):
    !python train_ae.py --data_dir /content/drive/MyDrive/medical_mnist

Outputs:
    outputs/ae_weights.keras        — Saved model weights
    outputs/ae_history.npy          — Training history
    outputs/ae_reconstructions.png  — Reconstruction comparison
    outputs/ae_latent_space.png     — Latent space scatter (PCA)
    outputs/ae_loss_curves.png      — Loss curves
"""

import os
import argparse
import numpy as np
import tensorflow as tf

from models import Autoencoder
from utils  import (
    load_medical_mnist,
    add_noise,
    plot_reconstructions,
    plot_latent_space_2d,
    plot_loss_curves,
    plot_denoising,
)
from utils.visualization import plot_latent_interpolation

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

DEFAULT_CONFIG = {
    "data_dir"    : "data/medical_mnist",   # folder inside the project root
    "latent_dim"  : 32,
    "batch_size"  : 64,
    "epochs"      : 40,
    "learning_rate": 1e-3,
    "noise_factor" : 0.3,    # for denoising experiment
    "output_dir"  : "./outputs",
}


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(description="Train Autoencoder on Medical MNIST")
    parser.add_argument("--data_dir",     type=str,   default=DEFAULT_CONFIG["data_dir"])
    parser.add_argument("--latent_dim",   type=int,   default=DEFAULT_CONFIG["latent_dim"])
    parser.add_argument("--batch_size",   type=int,   default=DEFAULT_CONFIG["batch_size"])
    parser.add_argument("--epochs",       type=int,   default=DEFAULT_CONFIG["epochs"])
    parser.add_argument("--lr",           type=float, default=DEFAULT_CONFIG["learning_rate"])
    parser.add_argument("--output_dir",   type=str,   default=DEFAULT_CONFIG["output_dir"])
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Helper: extract a numpy batch from a tf.data.Dataset
# ---------------------------------------------------------------------------

def _get_sample_batch(dataset: tf.data.Dataset, n: int = 128):
    """Return the first n (images, labels) as numpy arrays."""
    images_list, labels_list = [], []
    for imgs, labs in dataset:
        images_list.append(imgs.numpy())
        labels_list.append(labs.numpy())
        if sum(len(x) for x in images_list) >= n:
            break
    images = np.concatenate(images_list, axis=0)[:n]
    labels = np.concatenate(labels_list, axis=0)[:n]
    return images, labels


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    # ------------------------------------------------------------------ #
    # 1. Data
    # ------------------------------------------------------------------ #
    print("\n[1/5] Loading data …")
    ds_train, ds_val, ds_test, class_names = load_medical_mnist(
        data_dir   = args.data_dir,
        batch_size = args.batch_size,
    )

    # Estimate steps per epoch from a quick pass over train filenames
    # (repeat() is on, so we need steps_per_epoch)
    steps_per_epoch = None   # Keras will infer if dataset is not repeated;
                             # set manually if needed.

    # ------------------------------------------------------------------ #
    # 2. Build model
    # ------------------------------------------------------------------ #
    print("[2/5] Building Autoencoder …")
    ae = Autoencoder(latent_dim=args.latent_dim)

    ae.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=args.lr))

    # Warm-up call to build layers
    dummy = tf.zeros((1, 64, 64, 1))
    _ = ae(dummy)
    ae.encoder.summary()
    ae.decoder.summary()

    # ------------------------------------------------------------------ #
    # 3. Train
    # ------------------------------------------------------------------ #
    print("[3/5] Training …")
    callbacks = [
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_reconstruction_loss", factor=0.5, patience=5, verbose=1
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor="val_reconstruction_loss", patience=10, restore_best_weights=True
        ),
    ]

    history = ae.fit(
        ds_train,
        validation_data = ds_val,
        epochs          = args.epochs,
        callbacks       = callbacks,
    )

    # Save weights
    weights_path = os.path.join(args.output_dir, "ae_weights.keras")
    ae.save_weights(weights_path)
    print(f"  ✓ Weights saved to {weights_path}")

    # Save history
    history_path = os.path.join(args.output_dir, "ae_history.npy")
    np.save(history_path, history.history)

    # ------------------------------------------------------------------ #
    # 4. Evaluate & Visualise
    # ------------------------------------------------------------------ #
    print("[4/5] Generating visualisations …")

    # --- Sample a test batch ---
    test_images, test_labels = _get_sample_batch(ds_test, n=128)
    reconstructions = ae(test_images, training=False).numpy()

    # Reconstruction grid
    plot_reconstructions(
        test_images, reconstructions,
        title     = "AE — Reconstruction",
        save_path = os.path.join(args.output_dir, "ae_reconstructions.png"),
    )

    # Latent space
    latent_vecs = ae.encode(test_images).numpy()
    plot_latent_space_2d(
        latent_vecs, test_labels,
        method     = "pca",
        title      = "AE — Latent Space (PCA)",
        class_names = class_names,
        save_path  = os.path.join(args.output_dir, "ae_latent_space_pca.png"),
    )
    plot_latent_space_2d(
        latent_vecs, test_labels,
        method     = "tsne",
        title      = "AE — Latent Space (t-SNE)",
        class_names = class_names,
        save_path  = os.path.join(args.output_dir, "ae_latent_space_tsne.png"),
    )

    # Loss curves
    plot_loss_curves(
        history.history,
        model_name = "Autoencoder",
        save_path  = os.path.join(args.output_dir, "ae_loss_curves.png"),
    )

    # Latent interpolation between two test images
    z_start = latent_vecs[0]
    z_end   = latent_vecs[1]
    plot_latent_interpolation(
        ae, z_start, z_end,
        title     = "AE — Latent Interpolation",
        save_path = os.path.join(args.output_dir, "ae_interpolation.png"),
    )

    # ------------------------------------------------------------------ #
    # 5. Denoising experiment
    # ------------------------------------------------------------------ #
    print("[5/5] Denoising experiment …")
    ds_noisy_test = add_noise(ds_test, noise_factor=DEFAULT_CONFIG["noise_factor"])
    noisy_imgs, clean_imgs = next(iter(ds_noisy_test))
    noisy_imgs = noisy_imgs.numpy()[:16]
    clean_imgs = clean_imgs.numpy()[:16]

    # The standard AE was NOT trained for denoising — feed noisy input anyway
    # to illustrate how denoising-AE would work (retrain on noisy pipeline for full effect)
    denoised = ae(noisy_imgs, training=False).numpy()

    plot_denoising(
        noisy_imgs, clean_imgs, denoised,
        save_path = os.path.join(args.output_dir, "ae_denoising.png"),
    )

    print("\n[Done] All outputs saved to:", args.output_dir)


if __name__ == "__main__":
    main()
