"""
Model definitions for standard Autoencoder (AE) and Variational Autoencoder (VAE).
"""

from typing import Tuple, List, Dict, Any

import tensorflow as tf

IMG_SIZE = 64
LATENT_DIM = 32


class Sampling(tf.keras.layers.Layer):
    """Custom layer for the reparameterization trick in VAEs."""
    def call(self, inputs: Tuple[tf.Tensor, tf.Tensor]) -> tf.Tensor:
        mu, lv = inputs
        epsilon = tf.random.normal(shape=tf.shape(mu))
        return mu + tf.exp(0.5 * lv) * epsilon


class KLAnnealing(tf.keras.callbacks.Callback):
    """Linearly increases KL weight from 0 to 1 over warmup_epochs."""
    def __init__(self, warmup_epochs: int = 10):
        super().__init__()
        self.warmup_epochs = warmup_epochs

    def on_epoch_begin(self, epoch: int, logs: Dict[str, Any] = None):
        new_weight = min(1.0, (epoch + 1) / self.warmup_epochs)
        self.model.kl_weight.assign(new_weight)


class VAE(tf.keras.Model):
    """Variational Autoencoder model class."""
    def __init__(self, enc: tf.keras.Model, dec: tf.keras.Model, **kwargs):
        super().__init__(**kwargs)
        self.enc = enc
        self.dec = dec
        self.sample_fn = Sampling()
        self.kl_weight = tf.Variable(0.0, trainable=False, dtype=tf.float32)
        self.t_loss = tf.keras.metrics.Mean('total_loss')
        self.r_loss = tf.keras.metrics.Mean('recon_loss')
        self.k_loss = tf.keras.metrics.Mean('kl_loss')

    @property
    def metrics(self) -> List[tf.keras.metrics.Metric]:
        return [self.t_loss, self.r_loss, self.k_loss]

    def call(self, x: tf.Tensor) -> tf.Tensor:
        mu, lv = self.enc(x)
        return self.dec(self.sample_fn([mu, lv]))

    def _compute(self, x: tf.Tensor, training: bool) -> Tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
        mu, lv = self.enc(x, training=training)
        xhat = self.dec(self.sample_fn([mu, lv]), training=training)

        r = tf.reduce_mean(tf.square(x - xhat))
        kl_raw = -0.5 * tf.reduce_mean(1 + lv - tf.square(mu) - tf.exp(lv))
        kl = (kl_raw / (IMG_SIZE * IMG_SIZE)) * self.kl_weight

        return r + kl, r, kl

    def train_step(self, data: Tuple[tf.Tensor, tf.Tensor]) -> Dict[str, tf.Tensor]:
        x, _ = data
        with tf.GradientTape() as tape:
            loss, r, kl = self._compute(x, training=True)
        self.optimizer.apply_gradients(
            zip(tape.gradient(loss, self.trainable_variables), self.trainable_variables)
        )
        self.t_loss.update_state(loss)
        self.r_loss.update_state(r)
        self.k_loss.update_state(kl)
        return {m.name: m.result() for m in self.metrics}

    def test_step(self, data: Tuple[tf.Tensor, tf.Tensor]) -> Dict[str, tf.Tensor]:
        x, _ = data
        loss, r, kl = self._compute(x, training=False)
        self.t_loss.update_state(loss)
        self.r_loss.update_state(r)
        self.k_loss.update_state(kl)
        return {m.name: m.result() for m in self.metrics}


def build_encoder(variational: bool = False, name_prefix: str = '') -> tf.keras.Model:
    """Builds the encoder network."""
    inp = tf.keras.Input(shape=(IMG_SIZE, IMG_SIZE, 1))

    x = tf.keras.layers.Conv2D(32, 3, strides=2, padding='same')(inp)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.LeakyReLU(0.2)(x)

    x = tf.keras.layers.Conv2D(64, 3, strides=2, padding='same')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.LeakyReLU(0.2)(x)

    x = tf.keras.layers.Conv2D(128, 3, strides=2, padding='same')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.LeakyReLU(0.2)(x)

    x = tf.keras.layers.Conv2D(256, 3, strides=2, padding='same')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.LeakyReLU(0.2)(x)

    x = tf.keras.layers.Flatten()(x)
    x = tf.keras.layers.Dense(512, activation='relu')(x)
    x = tf.keras.layers.Dropout(0.2)(x)

    if variational:
        mu = tf.keras.layers.Dense(LATENT_DIM, name='z_mean')(x)
        lv = tf.keras.layers.Dense(LATENT_DIM, name='z_log_var')(x)
        lv = tf.keras.layers.Lambda(lambda t: tf.clip_by_value(t, -4, 4), name='z_log_var_clip')(lv)
        return tf.keras.Model(inp, [mu, lv], name=f'{name_prefix}_vae_enc')

    z = tf.keras.layers.Dense(LATENT_DIM, name='z')(x)
    return tf.keras.Model(inp, z, name=f'{name_prefix}_ae_enc')


def build_decoder(name_prefix: str = '') -> tf.keras.Model:
    """Builds the decoder network."""
    s = IMG_SIZE // 16
    inp = tf.keras.Input(shape=(LATENT_DIM,))
    
    x = tf.keras.layers.Dense(s * s * 256, activation='relu')(inp)
    x = tf.keras.layers.Reshape((s, s, 256))(x)

    x = tf.keras.layers.Conv2DTranspose(256, 3, strides=2, padding='same')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.LeakyReLU(0.2)(x)

    x = tf.keras.layers.Conv2DTranspose(128, 3, strides=2, padding='same')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.LeakyReLU(0.2)(x)

    x = tf.keras.layers.Conv2DTranspose(64, 3, strides=2, padding='same')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.LeakyReLU(0.2)(x)

    x = tf.keras.layers.Conv2DTranspose(32, 3, strides=2, padding='same')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.LeakyReLU(0.2)(x)

    out = tf.keras.layers.Conv2D(1, 3, activation='sigmoid', padding='same')(x)
    return tf.keras.Model(inp, out, name=f'{name_prefix}_dec')


def build_ae(region: str) -> Tuple[tf.keras.Model, tf.keras.Model, tf.keras.Model]:
    """Compiles and returns the Autoencoder."""
    enc = build_encoder(variational=False, name_prefix=region)
    dec = build_decoder(name_prefix=region)
    inp = tf.keras.Input(shape=(IMG_SIZE, IMG_SIZE, 1))
    model = tf.keras.Model(inp, dec(enc(inp)), name=f'AE_{region}')
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3), loss='mse')
    return model, enc, dec


def build_vae(region: str) -> Tuple[tf.keras.Model, tf.keras.Model, tf.keras.Model]:
    """Compiles and returns the Variational Autoencoder."""
    enc = build_encoder(variational=True, name_prefix=region)
    dec = build_decoder(name_prefix=region)
    model = VAE(enc, dec, name=f'VAE_{region}')
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3))
    return model, enc, dec