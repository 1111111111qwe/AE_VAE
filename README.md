# AE & VAE on Medical MNIST — Local VS Code Setup

## Step 1 — Download the dataset from Kaggle

1. Go to: https://www.kaggle.com/datasets/andrewmvd/medical-mnist
2. Click **Download** → you get a ZIP file (e.g. `archive.zip`)
3. Extract it — you should see folders: `AbdomenCT`, `BreastMRI`, `ChestCT`, `CXR`, `Hand`, `HeadCT`
4. Put those folders inside `ae_vae_project/data/medical_mnist/`

Final structure should look like:
```
ae_vae_project/
    data/
        medical_mnist/
            AbdomenCT/    ← *.jpeg files inside
            BreastMRI/
            ChestCT/
            CXR/
            Hand/
            HeadCT/
    models/
    utils/
    outputs/              ← created automatically
    experiment_notebook.ipynb
    train_ae.py
    train_vae.py
    requirements.txt
```

---

## Step 2 — Install Python & create virtual environment

Open a terminal in VS Code (`Ctrl+`` ` or Terminal → New Terminal):

```bash
# Create virtual environment
python -m venv venv

# Activate it
# Windows:
venv\Scripts\activate
# Mac / Linux:
source venv/bin/activate

# Install all dependencies
pip install -r requirements.txt
```

---

## Step 3 — Run the project

### Option A — Jupyter Notebook (recommended, see plots inline)

1. In VS Code, open `experiment_notebook.ipynb`
2. Click **Select Kernel** (top right) → choose the `venv` Python interpreter
3. Run cells one by one with `Shift+Enter`

### Option B — Terminal scripts

```bash
# Train AE
python train_ae.py

# Train VAE
python train_vae.py

# Custom data path
python train_ae.py --data_dir path/to/medical_mnist --epochs 30
```

All output plots and weights are saved to `outputs/`.

---

## Project Structure

```
models/
    autoencoder.py    ← AE encoder-decoder architecture
    vae.py            ← VAE with reparameterisation trick
utils/
    data_loader.py    ← tf.data pipeline
    losses.py         ← MSE, BCE, KL divergence
    visualization.py  ← all plotting functions
train_ae.py           ← full AE training + evaluation
train_vae.py          ← full VAE training + evaluation
experiment_notebook.ipynb  ← interactive experiments
```

---

## Troubleshooting

| Problem | Fix |
|---------|-----|
| `ModuleNotFoundError: tensorflow` | Make sure venv is activated and `pip install -r requirements.txt` was run |
| `No images found` | Check the dataset path — must contain the 6 class sub-folders |
| Slow training (no GPU) | Reduce `--epochs` to 10–15 for a quick test run |
| VS Code can't find kernel | Click **Select Kernel** → **Python Environments** → pick `venv` |
