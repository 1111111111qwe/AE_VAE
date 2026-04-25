"""
Loss functions for AE and VAE.

Centralised here so they can be reused independently of model classes,
e.g. for custom training loops or ablation studies.
"""

import tensorflow as tf


# ---------------------------------------------------------------------------
# Reconstruction losses
# ---------------------------------------------------------------------------

def reconstruction_loss_mse(x: tf.Tensor, x_hat: tf.Tensor) -> tf.Tensor:
    """
    Mean Squared Error reconstruction loss.
    Suitable for AE training; measures pixel-level fidelity.

    Args:
        x     : Ground-truth images, shape (B, H, W, C).
        x_hat : Reconstructed images, shape (B, H, W, C).

    Returns:
        Scalar loss.
    """
    return tf.reduce_mean(tf.square(x - x_hat))


def reconstruction_loss_bce(x: tf.Tensor, x_hat: tf.Tensor) -> tf.Tensor:
    """
    Binary Cross-Entropy reconstruction loss (sum over pixels, mean over batch).
    Preferred for VAE when outputs are in [0, 1].

    Args:
        x     : Ground-truth images, shape (B, H, W, C).
        x_hat : Reconstructed images, shape (B, H, W, C).

    Returns:
        Scalar loss.
    """
    # Sum over spatial dims, mean over batch
    per_sample = tf.reduce_sum(
        tf.keras.losses.binary_crossentropy(x, x_hat), axis=(1, 2)
    )
    return tf.reduce_mean(per_sample)


# ---------------------------------------------------------------------------
# KL Divergence
# ---------------------------------------------------------------------------

def kl_divergence_loss(z_mean: tf.Tensor, z_log_var: tf.Tensor) -> tf.Tensor:
    """
    Analytical KL divergence: D_KL( q(z|x) || N(0, I) ).

    Derived from the closed-form solution for two Gaussians:
        -0.5 * sum(1 + log_var - mean^2 - exp(log_var))

    Args:
        z_mean    : Mean of approximate posterior q(z|x), shape (B, D).
        z_log_var : Log-variance of q(z|x), shape (B, D).

    Returns:
        Scalar KL loss (mean over batch).
    """
    kl_per_sample = -0.5 * tf.reduce_sum(
        1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var),
        axis=1,
    )
    return tf.reduce_mean(kl_per_sample)


# ---------------------------------------------------------------------------
# Combined VAE loss
# ---------------------------------------------------------------------------

def vae_loss(
    x: tf.Tensor,
    x_hat: tf.Tensor,
    z_mean: tf.Tensor,
    z_log_var: tf.Tensor,
    beta: float = 1.0,
    use_bce: bool = True,
) -> dict:
    """
    Total VAE loss = reconstruction_loss + β * kl_loss.

    Args:
        x          : Original images.
        x_hat      : Reconstructed images.
        z_mean     : Latent mean.
        z_log_var  : Latent log-variance.
        beta       : Weight for the KL term (β-VAE).
        use_bce    : Use BCE reconstruction loss (True) or MSE (False).

    Returns:
        dict with keys: total_loss, reconstruction_loss, kl_loss.
    """
    recon_loss = (
        reconstruction_loss_bce(x, x_hat)
        if use_bce
        else reconstruction_loss_mse(x, x_hat)
    )
    kl_loss   = kl_divergence_loss(z_mean, z_log_var)
    total     = recon_loss + beta * kl_loss

    return {
        "total_loss": total,
        "reconstruction_loss": recon_loss,
        "kl_loss": kl_loss,
    }
