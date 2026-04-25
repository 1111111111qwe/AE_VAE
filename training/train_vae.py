"""
train_vae.py — Training script for the Variational Autoencoder (VAE).

Usage (local):
    python train_vae.py

Usage (Colab):
    !python train_vae.py --data_dir /content/drive/MyDrive/medical_mnist

Outputs:
    outputs/vae_weights.keras         — Saved model weights
    outputs/vae_history.npy           — Training history
    outputs/vae_reconstructions.png   — Reconstruction comparison
    outputs/vae_latent_space.png      — Latent space scatter
    outputs/vae_generated_samples.png — New samples from prior
    outputs/vae_loss_curves.png       — All loss curves
    outputs/vae_interpolation.png     — Latent interpolation
"""

import os
import argparse
import numpy as np
import tensorflow as tf

from models import VAE
from utils  import (
    load_medical_mnist,
    add_noise,
    plot_reconstructions,
    plot_latent_space_2d,
    plot_loss_curves,
    plot_generated_samples,
    plot_latent_interpolation,
)
from utils.visualization import plot_denoising


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

DEFAULT_CONFIG = {
    "data_dir"    : "data/medical_mnist",
    "latent_dim"  : 32,
    "beta"        : 1.0,    # β-VAE: try 0.5, 1.0, 2.0 for ablations
    "batch_size"  : 64,
    "epochs"      : 50,
    "learning_rate": 1e-3,
    "noise_factor" : 0.3,
    "output_dir"  : "./outputs",
    "n_generated" : 16,
}


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(description="Train VAE on Medical MNIST")
    parser.add_argument("--data_dir",   type=str,   default=DEFAULT_CONFIG["data_dir"])
    parser.add_argument("--latent_dim", type=int,   default=DEFAULT_CONFIG["latent_dim"])
    parser.add_argument("--beta",       type=float, default=DEFAULT_CONFIG["beta"])
    parser.add_argument("--batch_size", type=int,   default=DEFAULT_CONFIG["batch_size"])
    parser.add_argument("--epochs",     type=int,   default=DEFAULT_CONFIG["epochs"])
    parser.add_argument("--lr",         type=float, default=DEFAULT_CONFIG["learning_rate"])
    parser.add_argument("--output_dir", type=str,   default=DEFAULT_CONFIG["output_dir"])
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def _get_sample_batch(dataset: tf.data.Dataset, n: int = 128):
    images_list, labels_list = [], []
    for imgs, labs in dataset:
        images_list.append(imgs.numpy())
        labels_list.append(labs.numpy())
        if sum(len(x) for x in images_list) >= n:
            break
    return (
        np.concatenate(images_list, axis=0)[:n],
        np.concatenate(labels_list, axis=0)[:n],
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    # ------------------------------------------------------------------ #
    # 1. Data
    # ------------------------------------------------------------------ #
    print("\n[1/6] Loading data …")
    ds_train, ds_val, ds_test, class_names = load_medical_mnist(
        data_dir   = args.data_dir,
        batch_size = args.batch_size,
    )

    # ------------------------------------------------------------------ #
    # 2. Build VAE
    # ------------------------------------------------------------------ #
    print("[2/6] Building VAE …")
    vae = VAE(
        latent_dim      = args.latent_dim,
        beta            = args.beta,
    )

    vae.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=args.lr))

    # Warm-up
    dummy = tf.zeros((1, 64, 64, 1))
    _ = vae(dummy)
    vae.encoder.summary()
    vae.decoder.summary()

    # ------------------------------------------------------------------ #
    # 3. Train
    # ------------------------------------------------------------------ #
    print("[3/6] Training …")
    callbacks = [
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_total_loss", factor=0.5, patience=5, verbose=1
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor="val_total_loss", patience=12, restore_best_weights=True
        ),
    ]

    history = vae.fit(
        ds_train,
        validation_data = ds_val,
        epochs          = args.epochs,
        callbacks       = callbacks,
    )

    # Save
    weights_path = os.path.join(args.output_dir, "vae_weights.keras")
    vae.save_weights(weights_path)
    np.save(os.path.join(args.output_dir, "vae_history.npy"), history.history)
    print(f"  ✓ Weights saved to {weights_path}")

    # ------------------------------------------------------------------ #
    # 4. Reconstruct & visualise
    # ------------------------------------------------------------------ #
    print("[4/6] Reconstruction visualisations …")
    test_images, test_labels = _get_sample_batch(ds_test, n=128)

    x_hat, z_mean, z_log_var, z = vae(test_images, training=False)
    reconstructions = x_hat.numpy()
    z_mean_np       = z_mean.numpy()

    plot_reconstructions(
        test_images, reconstructions,
        title     = "VAE — Reconstruction",
        save_path = os.path.join(args.output_dir, "vae_reconstructions.png"),
    )

    # ------------------------------------------------------------------ #
    # 5. Latent space
    # ------------------------------------------------------------------ #
    print("[5/6] Latent space visualisations …")
    plot_latent_space_2d(
        z_mean_np, test_labels,
        method     = "pca",
        title      = "VAE — Latent Space (PCA)",
        class_names = class_names,
        save_path  = os.path.join(args.output_dir, "vae_latent_space_pca.png"),
    )
    plot_latent_space_2d(
        z_mean_np, test_labels,
        method     = "tsne",
        title      = "VAE — Latent Space (t-SNE)",
        class_names = class_names,
        save_path  = os.path.join(args.output_dir, "vae_latent_space_tsne.png"),
    )

    # Loss curves
    plot_loss_curves(
        history.history,
        model_name = "VAE",
        save_path  = os.path.join(args.output_dir, "vae_loss_curves.png"),
    )

    # Interpolation
    z_start = z_mean_np[0]
    z_end   = z_mean_np[1]
    plot_latent_interpolation(
        vae, z_start, z_end,
        title     = "VAE — Latent Interpolation",
        save_path = os.path.join(args.output_dir, "vae_interpolation.png"),
    )

    # ------------------------------------------------------------------ #
    # 6. Generate new samples from prior
    # ------------------------------------------------------------------ #
    print("[6/6] Generating new samples from prior …")
    generated = vae.generate(n_samples=DEFAULT_CONFIG["n_generated"]).numpy()

    plot_generated_samples(
        generated,
        title     = "VAE — Generated Samples (Prior Sampling)",
        save_path = os.path.join(args.output_dir, "vae_generated_samples.png"),
    )

    # Denoising with VAE
    ds_noisy_test = add_noise(ds_test, noise_factor=DEFAULT_CONFIG["noise_factor"])
    noisy_batch, clean_batch = next(iter(ds_noisy_test))
    noisy_np = noisy_batch.numpy()[:16]
    clean_np = clean_batch.numpy()[:16]
    x_hat_noisy, *_ = vae(noisy_np, training=False)
    plot_denoising(
        noisy_np, clean_np, x_hat_noisy.numpy(),
        save_path = os.path.join(args.output_dir, "vae_denoising.png"),
    )

    print("\n[Done] All outputs saved to:", args.output_dir)


if __name__ == "__main__":
    main()
