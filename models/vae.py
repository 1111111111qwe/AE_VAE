"""
Variational Autoencoder (VAE) — Probabilistic latent space.

Key difference from AE:
  - Encoder outputs (μ, log σ²) instead of a single point z.
  - Reparameterisation trick: z = μ + σ * ε  where ε ~ N(0, I)
  - Loss = Reconstruction loss + KL divergence
"""

import tensorflow as tf
from tensorflow.keras import layers, Model


# ---------------------------------------------------------------------------
# Sampling layer (reparameterisation trick)
# ---------------------------------------------------------------------------

class Sampling(layers.Layer):
    """
    Reparameterisation trick:  z = μ + σ * ε,  ε ~ N(0, I).

    This allows gradients to flow through the sampling operation.
    """

    def call(self, inputs):
        z_mean, z_log_var = inputs
        batch  = tf.shape(z_mean)[0]
        dim    = tf.shape(z_mean)[1]
        epsilon = tf.random.normal(shape=(batch, dim))
        return z_mean + tf.exp(0.5 * z_log_var) * epsilon


# ---------------------------------------------------------------------------
# VAE Encoder
# ---------------------------------------------------------------------------

class VAEEncoder(Model):
    """
    Probabilistic encoder: maps x → (μ, log σ²).

    Args:
        latent_dim (int): Dimensionality of the latent space.
    """

    def __init__(self, latent_dim: int = 32, name: str = "vae_encoder", **kwargs):
        self.deconv4 = layers.Conv2DTranspose(16, 3, strides=2, padding="same", activation="relu")
        super().__init__(name=name, **kwargs)
        self.latent_dim = latent_dim

        self.conv1    = layers.Conv2D(32,  3, strides=2, padding="same", activation="relu")
        self.conv2    = layers.Conv2D(64,  3, strides=2, padding="same", activation="relu")
        self.conv3    = layers.Conv2D(128, 3, strides=2, padding="same", activation="relu")
        self.flatten  = layers.Flatten()
        self.dense    = layers.Dense(256, activation="relu")

        self.z_mean    = layers.Dense(latent_dim, name="z_mean")
        self.z_log_var = layers.Dense(latent_dim, name="z_log_var")
        self.sampling  = Sampling()

    def call(self, x: tf.Tensor, training: bool = False):
        """
        Returns:
            z_mean    : mean of the approximate posterior  q(z|x)
            z_log_var : log-variance of q(z|x)
            z         : sampled latent vector (via reparameterisation)
        """
        x = self.conv1(x, training=training)
        x = self.conv2(x, training=training)
        x = self.conv3(x, training=training)
        x = self.flatten(x)
        x = self.dense(x, training=training)

        z_mean    = self.z_mean(x)
        z_log_var = self.z_log_var(x)
        z         = self.sampling((z_mean, z_log_var))
        return z_mean, z_log_var, z

    def get_config(self):
        config = super().get_config()
        config.update({"latent_dim": self.latent_dim})
        return config


# ---------------------------------------------------------------------------
# VAE Decoder (identical architecture to AE decoder)
# ---------------------------------------------------------------------------

class VAEDecoder(Model):
    """
    Convolutional decoder for the VAE.

    Args:
        latent_dim (int): Dimensionality of the latent space.
        output_channels (int): Number of output image channels.
    """

    def __init__(
        self,
        latent_dim: int = 32,
        output_channels: int = 1,
        name: str = "vae_decoder",
        **kwargs,
    ):
        super().__init__(name=name, **kwargs)
        self.latent_dim      = latent_dim
        self.output_channels = output_channels

        self.dense       = layers.Dense(4 * 4 * 128, activation="relu")
        self.reshape     = layers.Reshape((4, 4, 128))

        self.deconv1     = layers.Conv2DTranspose(128, 3, strides=2, padding="same", activation="relu")
        self.deconv2     = layers.Conv2DTranspose(64,  3, strides=2, padding="same", activation="relu")
        self.deconv3     = layers.Conv2DTranspose(32,  3, strides=2, padding="same", activation="relu")
        self.output_conv = layers.Conv2DTranspose(
            output_channels, 3, padding="same", activation="sigmoid"
        )

    def call(self, z: tf.Tensor, training: bool = False) -> tf.Tensor:
        x = self.dense(z)
        x = self.reshape(x)
        x = self.deconv1(x, training=training)
        x = self.deconv2(x, training=training)
        x = self.deconv3(x, training=training)
        x = self.deconv4(x, training=training)
        x = self.output_conv(x, training=training)
        return x

    def get_config(self):
        config = super().get_config()
        config.update({"latent_dim": self.latent_dim, "output_channels": self.output_channels})
        return config


# ---------------------------------------------------------------------------
# VAE
# ---------------------------------------------------------------------------

class VAE(Model):
    """
    Variational Autoencoder.

    Loss = reconstruction_loss + β * kl_loss

    The β parameter (default 1.0) controls the weight of the KL term.
    Increasing β encourages a more disentangled latent space (β-VAE).

    Args:
        latent_dim (int): Size of the latent space.
        output_channels (int): Output image channels.
        beta (float): Weight for the KL divergence term.
    """

    def __init__(
        self,
        latent_dim: int = 32,
        output_channels: int = 1,
        beta: float = 1.0,
        name: str = "vae",
        **kwargs,
    ):
        super().__init__(name=name, **kwargs)
        self.latent_dim      = latent_dim
        self.output_channels = output_channels
        self.beta            = beta

        self.encoder = VAEEncoder(latent_dim=latent_dim)
        self.decoder = VAEDecoder(latent_dim=latent_dim, output_channels=output_channels)

        # Keras metrics for logging
        self.total_loss_tracker        = tf.keras.metrics.Mean(name="total_loss")
        self.reconstruction_loss_tracker = tf.keras.metrics.Mean(name="reconstruction_loss")
        self.kl_loss_tracker           = tf.keras.metrics.Mean(name="kl_loss")

    # ------------------------------------------------------------------
    # Metrics exposed to Keras
    # ------------------------------------------------------------------

    @property
    def metrics(self):
        return [
            self.total_loss_tracker,
            self.reconstruction_loss_tracker,
            self.kl_loss_tracker,
        ]

    # ------------------------------------------------------------------
    # Forward pass
    # ------------------------------------------------------------------

    def call(self, x: tf.Tensor, training: bool = False):
        """Returns (x_hat, z_mean, z_log_var, z)."""
        z_mean, z_log_var, z = self.encoder(x, training=training)
        x_hat                = self.decoder(z, training=training)
        return x_hat, z_mean, z_log_var, z

    # ------------------------------------------------------------------
    # Convenience helpers
    # ------------------------------------------------------------------

    def encode(self, x: tf.Tensor):
        """Return (z_mean, z_log_var, z) for input x."""
        return self.encoder(x, training=False)

    def decode(self, z: tf.Tensor) -> tf.Tensor:
        """Reconstruct from latent vector z."""
        return self.decoder(z, training=False)

    def generate(self, n_samples: int = 16) -> tf.Tensor:
        """Generate new samples by sampling from the prior N(0, I)."""
        z = tf.random.normal(shape=(n_samples, self.latent_dim))
        return self.decoder(z, training=False)

    # ------------------------------------------------------------------
    # Loss computation
    # ------------------------------------------------------------------

    @staticmethod
    def _reconstruction_loss(x: tf.Tensor, x_hat: tf.Tensor) -> tf.Tensor:
        """Binary cross-entropy per pixel, summed over spatial dims, mean over batch."""
        loss = tf.reduce_sum(
            tf.keras.losses.binary_crossentropy(x, x_hat), axis=(1, 2)
        )
        return tf.reduce_mean(loss)

    @staticmethod
    def _kl_divergence(z_mean: tf.Tensor, z_log_var: tf.Tensor) -> tf.Tensor:
        """Analytical KL: D_KL( q(z|x) || N(0,I) ) = -0.5 * sum(1 + log_var - mean² - var)."""
        kl = -0.5 * tf.reduce_sum(
            1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var), axis=1
        )
        return tf.reduce_mean(kl)

    # ------------------------------------------------------------------
    # Training / test steps
    # ------------------------------------------------------------------

    def train_step(self, data):
        x, _ = data

        with tf.GradientTape() as tape:
            x_hat, z_mean, z_log_var, _ = self(x, training=True)
            reconstruction_loss = self._reconstruction_loss(x, x_hat)
            kl_loss             = self._kl_divergence(z_mean, z_log_var)
            total_loss          = reconstruction_loss + self.beta * kl_loss

        grads = tape.gradient(total_loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.trainable_variables))

        self.total_loss_tracker.update_state(total_loss)
        self.reconstruction_loss_tracker.update_state(reconstruction_loss)
        self.kl_loss_tracker.update_state(kl_loss)

        return {m.name: m.result() for m in self.metrics}

    def test_step(self, data):
        x, _ = data
        x_hat, z_mean, z_log_var, _ = self(x, training=False)
        reconstruction_loss = self._reconstruction_loss(x, x_hat)
        kl_loss             = self._kl_divergence(z_mean, z_log_var)
        total_loss          = reconstruction_loss + self.beta * kl_loss

        self.total_loss_tracker.update_state(total_loss)
        self.reconstruction_loss_tracker.update_state(reconstruction_loss)
        self.kl_loss_tracker.update_state(kl_loss)

        return {m.name: m.result() for m in self.metrics}

    def get_config(self):
        config = super().get_config()
        config.update({
            "latent_dim": self.latent_dim,
            "output_channels": self.output_channels,
            "beta": self.beta,
        })
        return config
