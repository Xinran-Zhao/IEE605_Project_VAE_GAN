"""
Evaluation metrics for VAE models
"""

import torch
import torch.nn as nn
import numpy as np
import scipy.linalg
import torchvision.models as models
import torchvision.transforms as transforms
from torch.utils.data import DataLoader


def evaluate_bpd(model, test_loader, loss_function, device, kld_weight=1.0, 
                 image_channels=3, image_size=32, apply_bitdepth_correction=True):
    """
    Evaluate the bits per dimension (BPD) of the VAE model
    """
    model.eval()
    total_nll = 0
    num_samples = 0
    
    D = image_channels * image_size * image_size
    
    with torch.no_grad():
        for batch_idx, (x, _) in enumerate(test_loader):
            current_batch_size = x.size(0)
            x = x.to(device)
            
            x_recon, mu, logvar = model(x)
            loss, recon_loss, kld_loss = loss_function(x, x_recon, mu, logvar, kld_weight)
            
            total_nll += loss.item() * current_batch_size
            num_samples += current_batch_size
    
    avg_nll_nats = total_nll / num_samples
    bpd = avg_nll_nats / (D * np.log(2.0))
    
    if apply_bitdepth_correction:
        bitdepth_correction = 8.0
        bpd += bitdepth_correction
    
    print(f"\n{'='*60}")
    print(f"BPD Evaluation (Test Set)")
    print(f"{'='*60}")
    print(f"Average Negative ELBO (nats): {avg_nll_nats:.6f}")
    print(f"BPD (before correction): {avg_nll_nats / (D * np.log(2.0)):.6f}")
    if apply_bitdepth_correction:
        print(f"Bit-depth correction: +{bitdepth_correction:.6f}")
    print(f"Final BPD (upper bound on NLL): {bpd:.6f}")
    print(f"Number of test samples: {num_samples}")
    print(f"{'='*60}\n")
    print(f"Note: This BPD is an UPPER BOUND on the true NLL for VAE models.")
    
    return bpd


def evaluate_bpd_flow(model, data_loader, device, image_channels=3, image_size=32, 
                      apply_bitdepth_correction=True):
    """
    Evaluate Bits per Dimension for Flow models (exact NLL)
    """
    model.eval()
    total_nll = 0.0
    num_samples = 0
    
    D = image_channels * image_size * image_size  # Total dimensions per image
    
    print("Evaluating Flow model on dataset...")
    with torch.no_grad():
        for batch_idx, (x, _) in enumerate(data_loader):
            x = x.to(device)
            current_batch_size = x.size(0)
            
            # Get log probability from flow model
            log_px = model.log_prob(x)  # Shape: (batch_size,)
            
            # Negative log-likelihood
            nll = -log_px
            
            total_nll += nll.sum().item()
            num_samples += current_batch_size
            
            if (batch_idx + 1) % 20 == 0:
                print(f"  Processed {num_samples} samples...")
    
    # Calculate average NLL and BPD
    avg_nll = total_nll / num_samples
    bpd_raw = avg_nll / (D * np.log(2.0))
    
    # Apply bit-depth correction for dequantized models
    if apply_bitdepth_correction:
        bitdepth_correction = 8.0  # log2(256) for 8-bit images
        bpd = bpd_raw + bitdepth_correction
    else:
        bpd = bpd_raw
    
    # Print results
    print(f"\n{'='*60}")
    print(f"BPD Evaluation - Flow Model")
    print(f"{'='*60}")
    print(f"Dataset samples: {num_samples}")
    print(f"Image dimensions: {image_channels} × {image_size} × {image_size} = {D}")
    print(f"Average NLL (nats/image): {avg_nll:.4f}")
    print(f"BPD (before correction): {bpd_raw:.4f}")
    if apply_bitdepth_correction:
        print(f"Bit-depth correction: +{bitdepth_correction:.4f}")
        print(f"Final BPD: {bpd:.4f}")
    else:
        print(f"Final BPD: {bpd:.4f}")
    print(f"{'='*60}\n")
    print(f"Note: For Flow models, this is the EXACT negative log-likelihood,")
    print(f"      not an upper bound like VAE's ELBO.")
    if apply_bitdepth_correction:
        print(f"      Bit-depth correction accounts for dequantization (log2(256) = 8).")
    
    return bpd, avg_nll


class InceptionV3FeatureExtractor(nn.Module):
    """
    InceptionV3 model for feature extraction (used in FID calculation)
    """
    def __init__(self, device):
        super(InceptionV3FeatureExtractor, self).__init__()
        # Load pre-trained InceptionV3
        self.inception_model = models.inception_v3(weights='DEFAULT', transform_input=False)
        # Remove the classification layer
        self.inception_model.fc = nn.Identity()
        self.inception_model.eval()
        self.inception_model.to(device)
        
        # Transform for InceptionV3 (expects 299x299 images)
        self.transform = transforms.Compose([
            transforms.Resize(299),
            transforms.CenterCrop(299),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        self.device = device
    
    def forward(self, x):
        """
        Extract features from images
        
        Args:
            x: Input images (batch_size, channels, height, width)
        
        Returns:
            features: Extracted features
        """
        # If grayscale (1 channel), replicate to 3 channels
        if x.shape[1] == 1:
            x = x.repeat(1, 3, 1, 1)
        
        # Apply InceptionV3 preprocessing
        x = self.transform(x)
        
        # Extract features
        with torch.no_grad():
            features = self.inception_model(x)
        
        return features


def calculate_fid(real_images, generated_images, device, batch_size=50):
    """
    Calculate Fréchet Inception Distance (FID) between real and generated images
    
    FID measures the distance between feature distributions of real and generated images.
    Lower FID scores indicate better quality and diversity of generated images.
    
    Args:
        real_images: Tensor of real images (N, C, H, W)
        generated_images: Tensor of generated images (N, C, H, W)
        device: Device to run computation on
        batch_size: Batch size for feature extraction
    
    Returns:
        fid_score: FID score (lower is better)
    """
    print("Initializing InceptionV3 for FID calculation...")
    inception_extractor = InceptionV3FeatureExtractor(device)
    
    def get_features(images, extractor, batch_size):
        """Extract features from images in batches"""
        features_list = []
        num_batches = (len(images) + batch_size - 1) // batch_size
        
        for i in range(num_batches):
            start_idx = i * batch_size
            end_idx = min((i + 1) * batch_size, len(images))
            batch = images[start_idx:end_idx].to(device)
            
            features = extractor(batch)
            features_list.append(features.cpu().numpy())
        
        return np.concatenate(features_list, axis=0)
    
    print("Extracting features from real images...")
    real_features = get_features(real_images, inception_extractor, batch_size)
    
    print("Extracting features from generated images...")
    generated_features = get_features(generated_images, inception_extractor, batch_size)
    
    # Calculate mean and covariance
    mu_real = np.mean(real_features, axis=0)
    sigma_real = np.cov(real_features, rowvar=False)
    
    mu_gen = np.mean(generated_features, axis=0)
    sigma_gen = np.cov(generated_features, rowvar=False)
    
    # Calculate FID score
    diff = mu_real - mu_gen
    
    # Product might be almost singular
    covmean, _ = scipy.linalg.sqrtm(sigma_real.dot(sigma_gen), disp=False)
    
    # Numerical error might give slight imaginary component
    if np.iscomplexobj(covmean):
        covmean = covmean.real
    
    fid_score = diff.dot(diff) + np.trace(sigma_real + sigma_gen - 2 * covmean)
    
    print(f"FID Score: {fid_score:.4f}")
    return fid_score


def evaluate_reconstruction_error(model, test_loader, device, x_dim=3072, num_batches=10):
    """
    Evaluate reconstruction error (MSE) on test set
    
    Args:
        model: Trained VAE model
        test_loader: DataLoader for test set
        device: Device to run on
        x_dim: Input dimension
        num_batches: Number of batches to evaluate
    
    Returns:
        avg_mse: Average mean squared error
    """
    model.eval()
    total_mse = 0
    num_samples = 0
    
    with torch.no_grad():
        for batch_idx, (x, _) in enumerate(test_loader):
            if batch_idx >= num_batches:
                break
            
            current_batch_size = x.size(0)
            x = x.view(current_batch_size, x_dim).to(device)
            
            x_hat, _, _ = model(x)
            
            # Calculate MSE
            mse = torch.mean((x - x_hat) ** 2, dim=1).sum()
            total_mse += mse.item()
            num_samples += current_batch_size
    
    avg_mse = total_mse / num_samples
    print(f"Average Reconstruction MSE: {avg_mse:.6f}")
    
    return avg_mse

