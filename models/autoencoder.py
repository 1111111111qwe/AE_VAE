"""
Autoencoder (AE) — Encoder-Decoder architecture.
Uses convolutional layers for spatial feature extraction.
"""

import tensorflow as tf
from tensorflow.keras import layers, Model


# ---------------------------------------------------------------------------
# Encoder
# ---------------------------------------------------------------------------

class Encoder(Model):
    """
    Convolutional encoder that maps an input image to a latent vector.

    Args:
        latent_dim (int): Dimensionality of the latent space.
    """

    def __init__(self, latent_dim: int = 32, name: str = "encoder", **kwargs):
        self.deconv4 = layers.Conv2DTranspose(16, 3, strides=2, padding="same", activation="relu")
        super().__init__(name=name, **kwargs)
        self.latent_dim = latent_dim

        self.conv1 = layers.Conv2D(32, 3, strides=2, padding="same", activation="relu")
        self.conv2 = layers.Conv2D(64, 3, strides=2, padding="same", activation="relu")
        self.conv3 = layers.Conv2D(128, 3, strides=2, padding="same", activation="relu")
        self.flatten = layers.Flatten()
        self.dense  = layers.Dense(latent_dim, name="z")

    def call(self, x: tf.Tensor, training: bool = False) -> tf.Tensor:
        x = self.conv1(x, training=training)
        x = self.conv2(x, training=training)
        x = self.conv3(x, training=training)
        x = self.flatten(x)
        z = self.dense(x)
        return z

    def get_config(self):
        config = super().get_config()
        config.update({"latent_dim": self.latent_dim})
        return config


# ---------------------------------------------------------------------------
# Decoder
# ---------------------------------------------------------------------------

class Decoder(Model):
    """
    Convolutional decoder that maps a latent vector back to an image.

    Args:
        latent_dim (int): Dimensionality of the latent space.
        output_channels (int): Number of output image channels (1 = grayscale).
    """

    def __init__(
        self,
        latent_dim: int = 32,
        output_channels: int = 1,
        name: str = "decoder",
        **kwargs,
    ):
        super().__init__(name=name, **kwargs)
        self.latent_dim = latent_dim
        self.output_channels = output_channels

    
        self.dense   = layers.Dense(4 * 4 * 128, activation="relu")
        self.reshape = layers.Reshape((4, 4, 128))

        self.deconv1 = layers.Conv2DTranspose(128, 3, strides=2, padding="same", activation="relu")
        self.deconv2 = layers.Conv2DTranspose(64,  3, strides=2, padding="same", activation="relu")
        self.deconv3 = layers.Conv2DTranspose(32,  3, strides=2, padding="same", activation="relu")
        self.deconv4 = layers.Conv2DTranspose(16,  3, strides=2, padding="same", activation="relu")
        self.output_conv = layers.Conv2DTranspose(output_channels, 3, padding="same", activation="sigmoid")

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
# Autoencoder (AE)
# ---------------------------------------------------------------------------

class Autoencoder(Model):
    """
    Full Autoencoder: Encoder + Decoder.

    The model minimises Mean Squared Error (MSE) between input and reconstruction.

    Args:
        latent_dim (int): Size of the bottleneck representation.
        output_channels (int): Number of output channels (default 1 for grayscale).
    """

    def __init__(
        self,
        latent_dim: int = 32,
        output_channels: int = 1,
        name: str = "autoencoder",
        **kwargs,
    ):
        super().__init__(name=name, **kwargs)
        self.latent_dim = latent_dim
        self.output_channels = output_channels

        self.encoder = Encoder(latent_dim=latent_dim)
        self.decoder = Decoder(latent_dim=latent_dim, output_channels=output_channels)

    # ------------------------------------------------------------------
    # Forward pass
    # ------------------------------------------------------------------

    def call(self, x: tf.Tensor, training: bool = False) -> tf.Tensor:
        """Encode then decode an input image."""
        z    = self.encoder(x, training=training)
        x_hat = self.decoder(z, training=training)
        return x_hat

    # ------------------------------------------------------------------
    # Convenience helpers
    # ------------------------------------------------------------------

    def encode(self, x: tf.Tensor) -> tf.Tensor:
        """Return the latent representation of x."""
        return self.encoder(x, training=False)

    def decode(self, z: tf.Tensor) -> tf.Tensor:
        """Reconstruct an image from latent vector z."""
        return self.decoder(z, training=False)

    # ------------------------------------------------------------------
    # Training step
    # ------------------------------------------------------------------

    def train_step(self, data):
        x, _ = data  # labels unused for reconstruction

        with tf.GradientTape() as tape:
            x_hat = self(x, training=True)
            loss  = tf.reduce_mean(tf.square(x - x_hat))  # MSE

        grads = tape.gradient(loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.trainable_variables))

        return {"reconstruction_loss": loss}

    def test_step(self, data):
        x, _ = data
        x_hat = self(x, training=False)
        loss  = tf.reduce_mean(tf.square(x - x_hat))
        return {"reconstruction_loss": loss}

    def get_config(self):
        config = super().get_config()
        config.update({"latent_dim": self.latent_dim, "output_channels": self.output_channels})
        return config
