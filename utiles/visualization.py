"""
Visualization utilities for AE / VAE experiments.

All functions accept matplotlib Axes or create their own figures so they
can be embedded directly into a Jupyter notebook or saved to disk.
"""

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

try:
    from sklearn.decomposition import PCA
    from sklearn.manifold import TSNE
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False

try:
    import umap
    HAS_UMAP = True
except ImportError:
    HAS_UMAP = False


# ---------------------------------------------------------------------------
# Colour palette for classes
# ---------------------------------------------------------------------------

_CLASS_COLORS = [
    "#e63946", "#457b9d", "#2a9d8f", "#e9c46a", "#f4a261", "#a8dadc"
]

CLASS_NAMES = ["AbdomenCT", "BreastMRI", "ChestCT", "CXR", "Hand", "HeadCT"]


# ---------------------------------------------------------------------------
# 1. Reconstruction grid
# ---------------------------------------------------------------------------

def plot_reconstructions(
    originals: np.ndarray,
    reconstructions: np.ndarray,
    n: int = 8,
    title: str = "Reconstructions",
    save_path: str = None,
):
    """
    Side-by-side grid: top row = originals, bottom row = reconstructions.

    Args:
        originals        : Array of shape (N, H, W, C) in [0, 1].
        reconstructions  : Array of same shape as originals.
        n                : Number of images to display.
        title            : Figure title.
        save_path        : If given, save figure to this path.
    """
    n = min(n, len(originals))
    fig, axes = plt.subplots(2, n, figsize=(n * 2, 4))
    fig.suptitle(title, fontsize=14, fontweight="bold")

    for i in range(n):
        for row, imgs, row_title in zip(
            [0, 1], [originals, reconstructions], ["Original", "Reconstructed"]
        ):
            ax = axes[row, i]
            img = np.squeeze(imgs[i])
            ax.imshow(img, cmap="gray", vmin=0, vmax=1)
            ax.axis("off")
            if i == 0:
                ax.set_ylabel(row_title, fontsize=10, labelpad=4)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()


# ---------------------------------------------------------------------------
# 2. Latent space 2D scatter
# ---------------------------------------------------------------------------

def plot_latent_space_2d(
    latent_vectors: np.ndarray,
    labels: np.ndarray,
    method: str = "pca",
    title: str = None,
    class_names: list = None,
    save_path: str = None,
):
    """
    2D scatter of latent representations coloured by class label.

    Args:
        latent_vectors : Array of shape (N, D).
        labels         : Integer labels, shape (N,).
        method         : Dimensionality reduction: 'pca', 'tsne', or 'umap'.
                         If D == 2 already, no reduction is applied.
        title          : Figure title (auto-generated if None).
        class_names    : List of class name strings for the legend.
        save_path      : Optional path to save figure.
    """
    assert HAS_SKLEARN or method == "none", \
        "Install scikit-learn: pip install scikit-learn"

    class_names = class_names or CLASS_NAMES

    # ---- Dimensionality reduction ---------------------------------------
    D = latent_vectors.shape[1]

    if D == 2:
        coords = latent_vectors
        axis_labels = ("z₁", "z₂")
    elif method == "pca":
        pca = PCA(n_components=2)
        coords = pca.fit_transform(latent_vectors)
        var = pca.explained_variance_ratio_ * 100
        axis_labels = (f"PC1 ({var[0]:.1f}%)", f"PC2 ({var[1]:.1f}%)")
    elif method == "tsne":
        coords = TSNE(n_components=2, perplexity=30, random_state=42).fit_transform(latent_vectors)
        axis_labels = ("t-SNE 1", "t-SNE 2")
    elif method == "umap" and HAS_UMAP:
        coords = umap.UMAP(random_state=42).fit_transform(latent_vectors)
        axis_labels = ("UMAP 1", "UMAP 2")
    else:
        raise ValueError(f"Unknown method '{method}'. Choose: pca, tsne, umap.")

    # ---- Plot -----------------------------------------------------------
    fig, ax = plt.subplots(figsize=(8, 6))

    for cls_idx, (cls_name, color) in enumerate(zip(class_names, _CLASS_COLORS)):
        mask = labels == cls_idx
        ax.scatter(
            coords[mask, 0], coords[mask, 1],
            c=color, label=cls_name, alpha=0.6, s=12, edgecolors="none"
        )

    ax.set_xlabel(axis_labels[0])
    ax.set_ylabel(axis_labels[1])
    ax.legend(loc="best", markerscale=2, fontsize=8)
    ax.set_title(title or f"Latent Space ({method.upper()})", fontweight="bold")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()


# ---------------------------------------------------------------------------
# 3. Loss curves
# ---------------------------------------------------------------------------

def plot_loss_curves(
    history: dict,
    model_name: str = "Model",
    save_path: str = None,
):
    """
    Plot training (and validation) losses over epochs.

    Args:
        history    : Keras History.history dict or equivalent.
                     Expected keys (whichever exist):
                         reconstruction_loss, val_reconstruction_loss,
                         kl_loss, val_kl_loss,
                         total_loss, val_total_loss.
        model_name : Used in the figure title.
        save_path  : Optional path to save figure.
    """
    keys = list(history.keys())
    train_keys = [k for k in keys if not k.startswith("val_")]
    n_plots = len(train_keys)

    fig, axes = plt.subplots(1, n_plots, figsize=(5 * n_plots, 4))
    if n_plots == 1:
        axes = [axes]

    for ax, key in zip(axes, train_keys):
        epochs = range(1, len(history[key]) + 1)
        ax.plot(epochs, history[key], label="Train", linewidth=2)
        val_key = f"val_{key}"
        if val_key in history:
            ax.plot(epochs, history[val_key], label="Val", linewidth=2, linestyle="--")
        ax.set_title(key.replace("_", " ").title(), fontweight="bold")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Loss")
        ax.legend()
        ax.grid(True, alpha=0.3)

    fig.suptitle(f"{model_name} — Training Curves", fontsize=13, fontweight="bold")
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()


# ---------------------------------------------------------------------------
# 4. Generated samples (VAE)
# ---------------------------------------------------------------------------

def plot_generated_samples(
    samples: np.ndarray,
    title: str = "Generated Samples (VAE)",
    save_path: str = None,
):
    """
    Display a grid of VAE-generated images.

    Args:
        samples   : Array of shape (N, H, W, C) in [0, 1].
        title     : Figure title.
        save_path : Optional path to save.
    """
    n = len(samples)
    cols = int(np.ceil(np.sqrt(n)))
    rows = int(np.ceil(n / cols))

    fig, axes = plt.subplots(rows, cols, figsize=(cols * 2, rows * 2))
    axes = np.array(axes).reshape(-1)

    for i, ax in enumerate(axes):
        if i < n:
            ax.imshow(np.squeeze(samples[i]), cmap="gray", vmin=0, vmax=1)
        ax.axis("off")

    fig.suptitle(title, fontsize=13, fontweight="bold")
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()


# ---------------------------------------------------------------------------
# 5. Latent space interpolation
# ---------------------------------------------------------------------------

def plot_latent_interpolation(
    model,
    z_start: np.ndarray,
    z_end: np.ndarray,
    n_steps: int = 10,
    title: str = "Latent Space Interpolation",
    save_path: str = None,
):
    """
    Decode linearly interpolated latent vectors between z_start and z_end.

    Args:
        model   : AE or VAE model with a .decode(z) method.
        z_start : Start latent vector, shape (D,).
        z_end   : End latent vector, shape (D,).
        n_steps : Number of interpolation steps.
        title   : Figure title.
        save_path: Optional save path.
    """
    import tensorflow as tf

    alphas = np.linspace(0, 1, n_steps)
    z_interp = np.array([
        (1 - a) * z_start + a * z_end for a in alphas
    ], dtype=np.float32)

    decoded = model.decode(tf.constant(z_interp)).numpy()

    fig, axes = plt.subplots(1, n_steps, figsize=(n_steps * 2, 2.5))
    for ax, img, alpha in zip(axes, decoded, alphas):
        ax.imshow(np.squeeze(img), cmap="gray", vmin=0, vmax=1)
        ax.set_title(f"α={alpha:.1f}", fontsize=7)
        ax.axis("off")

    fig.suptitle(title, fontsize=13, fontweight="bold")
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()


# ---------------------------------------------------------------------------
# 6. Denoising comparison
# ---------------------------------------------------------------------------

def plot_denoising(
    noisy: np.ndarray,
    clean: np.ndarray,
    reconstructed: np.ndarray,
    n: int = 8,
    save_path: str = None,
):
    """
    Three-row grid: noisy input | clean target | model output.

    Args:
        noisy         : Noisy images, shape (N, H, W, C).
        clean         : Clean images, shape (N, H, W, C).
        reconstructed : Model output, shape (N, H, W, C).
        n             : Number of examples to show.
        save_path     : Optional save path.
    """
    n = min(n, len(noisy))
    fig, axes = plt.subplots(3, n, figsize=(n * 2, 6))
    fig.suptitle("Denoising Results", fontsize=14, fontweight="bold")

    row_titles = ["Noisy Input", "Clean Target", "Model Output"]
    rows_data  = [noisy, clean, reconstructed]

    for row, (data, row_title) in enumerate(zip(rows_data, row_titles)):
        for col in range(n):
            ax = axes[row, col]
            ax.imshow(np.squeeze(data[col]), cmap="gray", vmin=0, vmax=1)
            ax.axis("off")
            if col == 0:
                ax.set_ylabel(row_title, fontsize=9)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()
