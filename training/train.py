"""
Main training pipeline.
Handles the orchestration of data loading, model compilation, training, and saving.
"""

import os
import json

import numpy as np
import tensorflow as tf

from src.data_processing import setup_data_directory, get_paths, make_region_ds
from src.model import build_ae, build_vae, KLAnnealing
from src.utils import generate_summary_csv

# --- CONFIGURATION ---
DRIVE_DIR = '/content/drive/MyDrive/medical_mnist'
LOCAL_DIR = 'data/raw/medical_mnist'  # Fits convention data/raw/
MODELS_DIR = 'models/'
PROCESSED_DIR = 'data/processed/'
EPOCHS = 30

tf.random.set_seed(42)
np.random.seed(42)


def main():
    """Main execution function."""
    os.makedirs(MODELS_DIR, exist_ok=True)
    os.makedirs(PROCESSED_DIR, exist_ok=True)

    regions = setup_data_directory(DRIVE_DIR, LOCAL_DIR)
    print('Regions found:', regions)

    region_splits = {}
    for region in regions:
        all_p = get_paths(LOCAL_DIR, region)
        np.random.shuffle(all_p)
        cut = int(0.9 * len(all_p))
        region_splits[region] = {'train': all_p[:cut], 'val': all_p[cut:]}

    results = {}

    for region in regions:
        print(f'\n--- Training Region: {region} ---')

        tr_ds = make_region_ds(region_splits[region]['train'], training=True)
        va_ds = make_region_ds(region_splits[region]['val'], training=False)

        # ── AE Training ──
        print(f'[{region}] Training Autoencoder...')
        ae, ae_enc, ae_dec = build_ae(region)

        ae_callbacks = [
            tf.keras.callbacks.EarlyStopping(
                monitor='val_loss', patience=6, min_delta=0.0001, restore_best_weights=True, verbose=0),
            tf.keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss', factor=0.5, patience=3, min_lr=1e-5, verbose=1),
        ]

        ae_hist = ae.fit(tr_ds, validation_data=va_ds, epochs=EPOCHS, callbacks=ae_callbacks, verbose=1)
        
        # Save model adhering to convention
        ae.save(os.path.join(MODELS_DIR, f'AE_{region}_v1.h5'))

        # ── VAE Training ──
        print(f'[{region}] Training Variational Autoencoder...')
        vae, vae_enc, vae_dec = build_vae(region)

        vae_callbacks = [
            tf.keras.callbacks.EarlyStopping(
                monitor='val_total_loss', patience=6, min_delta=0.0001, restore_best_weights=True, mode='min', verbose=0),
            tf.keras.callbacks.ReduceLROnPlateau(
                monitor='val_total_loss', factor=0.5, patience=3, min_lr=1e-5, mode='min', verbose=1),
            KLAnnealing(warmup_epochs=10),
        ]

        vae_hist = vae.fit(tr_ds, validation_data=va_ds, epochs=EPOCHS, callbacks=vae_callbacks, verbose=1)
        
        # Save VAE weights (custom subclasses require save_weights instead of .h5)
        vae.save_weights(os.path.join(MODELS_DIR, f'VAE_{region}_v1_weights.h5'))

        results[region] = dict(
            ae=ae, vae=vae,
            ae_hist=ae_hist, vae_hist=vae_hist,
            val_paths=region_splits[region]['val']
        )

    # Generate performance summary
    csv_path = os.path.join(PROCESSED_DIR, 'summary_per_region.csv')
    generate_summary_csv(results, regions, csv_path)

    print("\nTraining pipeline complete. Models saved in models/, summary saved in data/processed/.")


if __name__ == '__main__':
    main()