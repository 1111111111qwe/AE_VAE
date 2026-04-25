"""
Utility functions for evaluation, plotting, and summarizing results.
"""

from typing import Dict, Any, List

import numpy as np
import pandas as pd
import tensorflow as tf

from src.data_processing import make_region_ds


def generate_summary_csv(results: Dict[str, Any], regions: List[str], output_path: str):
    """
    Evaluates models to compute sample MSE and saves a summary to CSV in data/processed/.
    """
    rows = []
    for region in regions:
        r = results[region]
        ah = r['ae_hist'].history
        vh = r['vae_hist'].history

        va_ds = make_region_ds(r['val_paths'], training=False)
        orig = next(iter(va_ds))[0][:16].numpy()
        
        ae_r = r['ae'].predict(orig, verbose=0)
        vae_r = r['vae'].predict(orig, verbose=0)

        rows.append({
            'Region': region,
            'AE Val Loss': round(ah['val_loss'][-1], 4),
            'VAE Val Total': round(vh.get('val_total_loss', [0])[-1], 2),
            'VAE Val KL': round(vh.get('val_kl_loss', [0])[-1], 4),
            'AE MSE (sample)': round(float(np.mean((orig - ae_r)**2)), 4),
            'VAE MSE (sample)': round(float(np.mean((orig - vae_r)**2)), 4),
        })

    df = pd.DataFrame(rows)
    df.to_csv(output_path, index=False)
    print(f"\nSummary successfully saved to {output_path}")
    print(df.to_string(index=False))