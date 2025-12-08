# IEE605 Project: Comparative Analysis of Generative Models

## Description

A comprehensive implementation and evaluation of three major families of generative models: **Variational Autoencoders (VAE)**, **Generative Adversarial Networks (GAN)**, and **Normalizing Flow models (RealNVP)**. This project provides end-to-end tools for training, sample generation, and rigorous quantitative evaluation on the CIFAR-10 dataset.

**Key capabilities:**
- Train and evaluate three different generative modeling approaches
- Quantitative metrics: FID, KID, BPD, Precision, and Recall
- Trade-off analysis: rate-distortion (VAE) and fidelity-diversity (GAN)
- Memorization detection and quality assurance

## Features

- **VAE Training:** Convolutional VAE architecture with configurable latent dimension and KL divergence regularization (beta) parameter, supporting sample generation and bits-per-dimension analysis.
- **GAN Training:** Custom GAN pipeline for CIFAR-10 with Generator and Discriminator networks, including training and sample visualization.
- **RealNVP Flow Training:** Modern flow-based model supporting likelihood-based image generation and sample synthesis with bits-per-dimension evaluation.
- **Sample Generation & Visualization:** Generate and create grids of synthetic images from trained models.
- **Quality Metrics:** Automated evaluation scripts for FID (Fréchet Inception Distance) and KID (Kernel Inception Distance).
- **Rate-Distortion Analysis:** Tools to investigate trade-offs between compression and reconstruction in VAEs.
- **Fidelity-Diversity Trade-off:** Temperature-based analysis for GANs to balance image quality and diversity.
- **Memorization Detection:** Nearest neighbor analysis to detect potential mode collapse and memorization in GANs.
- **Data Utilities:** Modular data loaders and preprocessing for CIFAR-10.

## Requirements

- Python >= 3.8
- PyTorch >= 2.0.0
- CUDA-capable GPU (recommended for training)
- See `requirements.txt` for complete dependencies

## Installation

```bash
git clone https://github.com/Xinran-Zhao/IEE605_Project_VAE_GAN.git
cd IEE605_Project_VAE_GAN
pip install -r requirements.txt
```

**Note:** For best performance, ensure you have a CUDA-compatible GPU and the appropriate CUDA toolkit installed.

## Usage

### Train a VAE

```bash
python train_conv_vae.py
```
- Set hyperparameters in `train_conv_vae.py` as required.
- Check outputs and model checkpoints in `./checkpoint/beta_vae`.

### Train a GAN

```bash
python train_gan.py
```
- Adjust parameters in `train_gan.py`.
- Model checkpoint will be saved to `./checkpoint/gan_checkpoint.pt`.
- Training progress samples will be saved to `./progress_gan/`.

### Train RealNVP Flow

```bash
python train_flow.py
```
- Supports options for batch size, number of flows, hidden channels (`--batch-size`, `--num-flows`, `--hidden-channels` and others).
- Example:
  ```bash
  python train_flow.py --batch-size 128 --num-flows 14 --hidden-channels 800
  ```
- Model and optimizer checkpoints will be saved for later use.

### RealNVP Flow Evaluation (FID/KID)

```bash
python evaluate_flow_fid_kid.py
```
- Loads a trained RealNVP model, generates samples, computes FID/KID metrics, and produces visualization grids and reports.

### VAE Rate-Distortion Analysis

```bash
python generate_vae_tradeoff_curve.py
```
- Sweeps over multiple beta values to analyze rate/distortion curves and trade-offs.
- Results will be saved to `./samples_and_plots/`.

### VAE FID/KID Evaluation

```bash
python evaluate_vae_fid_kid.py
```
- Generates samples from trained VAE model and computes FID/KID metrics.
- Results and visualizations will be saved to `./vae_fid_kid_eval/`.

### GAN Fidelity-Diversity Analysis

```bash
python generate_gan_fidelity_diversity.py
```
- Evaluates GAN samples at different temperature settings to balance quality and diversity.
- Computes FID, KID, Precision, and Recall metrics.
- Results will be saved to `./fidelity_diversity_gan/`.

### GAN Nearest Neighbor Check

```bash
python gan_nearest_neighbor_check.py
```
- Analyzes generated images against the CIFAR-10 training set to detect memorization.
- Computes nearest neighbor distances and visualizes results.
- Outputs will be saved to `./gan_nearest_neighbor_results/`.

## Project Structure

```
├── models/
│   ├── ConvVAE.py         # VAE model architecture
│   ├── GAN.py             # GAN model (Generator & Discriminator)
│   └── RealNVP_flow.py    # RealNVP (flow-based generative model)
├── data/
│   └── cifar_10.py        # CIFAR-10 data loaders & transforms
├── evaluation/
│   ├── metrics.py         # FID/KID/BPD evaluation functions
│   └── tradeoff_metrics.py  # Rate-distortion metrics for VAE
├── train_conv_vae.py      # VAE training script
├── train_gan.py           # GAN training script
├── train_flow.py          # RealNVP flow-based model training script
├── generate_vae_tradeoff_curve.py  # VAE rate-distortion analyzer
├── generate_gan_fidelity_diversity.py  # GAN fidelity-diversity trade-off
├── evaluate_vae_fid_kid.py         # VAE FID/KID evaluation
├── evaluate_flow_fid_kid.py        # RealNVP FID/KID evaluation
├── gan_nearest_neighbor_check.py   # GAN memorization detection
└── requirements.txt       # Python dependencies
```

## Metrics

- **FID (Fréchet Inception Distance):** Measures the quality and diversity of generated images by comparing feature distributions; lower is better.
- **KID (Kernel Inception Distance):** An unbiased alternative to FID for evaluating image quality; lower is better.
- **Bits-per-dimension (BPD):** Measures the compression efficiency and likelihood of the model; lower is better (used for VAE and Flow models).
- **Precision & Recall:** For GANs, measures fidelity (precision) vs. diversity (recall) trade-off.
- **Rate-Distortion:** For VAEs, analyzes the trade-off between compression rate (KL divergence) and reconstruction quality (MSE).
- **Nearest Neighbor Distance:** Detects potential memorization by comparing generated samples to training data.


## License

Apache License 2.0

---

For further usage information, see the detailed comments and sections within each script.