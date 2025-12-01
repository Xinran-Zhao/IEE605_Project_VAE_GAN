# IEE605_Project_VAE_GAN

Implementation of Variational Autoencoder (VAE) and Generative Adversarial Network (GAN) models with evaluation metrics.

## Features

- **VAE (Variational Autoencoder)**: A generative model that learns a latent representation of the data
- **GAN (Generative Adversarial Network)**: DCGAN implementation with generator and discriminator
- **Evaluation Metrics**:
  - **Bits Per Dimension (BPD)**: Measures compression quality for VAE
  - **Fréchet Inception Distance (FID)**: Measures quality of generated images for GAN

## Installation

```bash
pip install -r requirements.txt
```

## Usage

### Training VAE

```bash
python train_vae.py --dataset mnist --epochs 50 --latent-dim 128
```

Options:
- `--dataset`: Dataset to use (mnist, fashion_mnist, cifar10)
- `--batch-size`: Batch size (default: 64)
- `--epochs`: Number of training epochs (default: 50)
- `--lr`: Learning rate (default: 1e-3)
- `--latent-dim`: Latent space dimension (default: 128)
- `--beta`: Beta parameter for beta-VAE (default: 1.0)
- `--output-dir`: Output directory (default: ./outputs/vae)

### Training GAN

```bash
python train_gan.py --dataset mnist --epochs 100 --latent-dim 100
```

Options:
- `--dataset`: Dataset to use (mnist, fashion_mnist, cifar10)
- `--batch-size`: Batch size (default: 64)
- `--epochs`: Number of training epochs (default: 100)
- `--lr`: Learning rate (default: 2e-4)
- `--latent-dim`: Latent noise dimension (default: 100)
- `--output-dir`: Output directory (default: ./outputs/gan)

## Project Structure

```
├── models/
│   ├── __init__.py
│   ├── vae.py          # VAE implementation
│   └── gan.py          # GAN implementation
├── metrics/
│   ├── __init__.py
│   ├── bpd.py          # Bits Per Dimension metric
│   └── fid.py          # Fréchet Inception Distance metric
├── utils/
│   ├── __init__.py
│   ├── data.py         # Data loading utilities
│   └── visualization.py # Visualization utilities
├── train_vae.py        # VAE training script
├── train_gan.py        # GAN training script
└── requirements.txt    # Dependencies
```

## Evaluation Metrics

### Bits Per Dimension (BPD)

BPD measures the negative log-likelihood normalized by the number of dimensions:

```
BPD = NLL / (log(2) * D)
```

Lower BPD indicates better compression and density estimation.

### Fréchet Inception Distance (FID)

FID measures the distance between real and generated image distributions:

```
FID = ||μ_real - μ_fake||² + Tr(Σ_real + Σ_fake - 2√(Σ_real·Σ_fake))
```

Lower FID indicates generated images are more similar to real images.

## License

Apache License 2.0