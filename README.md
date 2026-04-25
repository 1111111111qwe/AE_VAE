# Representation Learning with Autoencoders (AE & VAE)

## Setup and Installation
1. Clone the repository.
2. Create a virtual environment: `python -m venv venv`
3. Activate the environment: `source venv/bin/activate` (Linux/Mac) or `venv\Scripts\activate` (Windows).
4. Install dependencies: `pip install -r requirements.txt`

## Running the Code
1. Ensure your dataset is in Google Drive (`/content/drive/MyDrive/medical_mnist`) if using Colab, OR place the raw images directly inside `data/raw/medical_mnist` if running locally.
2. Run the training pipeline:
   ```bash
   python -m src.train