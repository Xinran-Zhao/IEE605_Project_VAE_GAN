"""
Fréchet Inception Distance (FID) metric for GAN evaluation.

FID measures the distance between the distribution of real and generated 
images in the feature space of a pretrained Inception network.

FID = ||mu_real - mu_fake||^2 + Tr(sigma_real + sigma_fake - 2*sqrt(sigma_real * sigma_fake))

Lower FID indicates generated images are more similar to real images.
"""

import torch
import torch.nn as nn
import numpy as np
from scipy import linalg
from typing import Tuple, Optional
import warnings


class InceptionFeatureExtractor(nn.Module):
    """
    Feature extractor for FID calculation.
    
    Uses a simple CNN for feature extraction when Inception is not available,
    or wraps a pretrained Inception-v3 model.
    """
    
    def __init__(self, use_inception: bool = True, feature_dim: int = 2048):
        """
        Args:
            use_inception: Whether to use pretrained Inception-v3
            feature_dim: Output feature dimension
        """
        super().__init__()
        
        self.use_inception = use_inception
        self.feature_dim = feature_dim
        
        if use_inception:
            try:
                from torchvision.models import inception_v3, Inception_V3_Weights
                # Load pretrained Inception-v3
                inception = inception_v3(weights=Inception_V3_Weights.DEFAULT)
                inception.eval()
                
                # Remove the final classification layer
                self.features = nn.Sequential(
                    inception.Conv2d_1a_3x3,
                    inception.Conv2d_2a_3x3,
                    inception.Conv2d_2b_3x3,
                    nn.MaxPool2d(kernel_size=3, stride=2),
                    inception.Conv2d_3b_1x1,
                    inception.Conv2d_4a_3x3,
                    nn.MaxPool2d(kernel_size=3, stride=2),
                    inception.Mixed_5b,
                    inception.Mixed_5c,
                    inception.Mixed_5d,
                    inception.Mixed_6a,
                    inception.Mixed_6b,
                    inception.Mixed_6c,
                    inception.Mixed_6d,
                    inception.Mixed_6e,
                    inception.Mixed_7a,
                    inception.Mixed_7b,
                    inception.Mixed_7c,
                    nn.AdaptiveAvgPool2d((1, 1)),
                )
                self.feature_dim = 2048
            except Exception as e:
                warnings.warn(f"Failed to load Inception: {e}. Using simple CNN.")
                self.use_inception = False
        
        if not self.use_inception:
            # Simple CNN feature extractor for smaller images
            self.features = nn.Sequential(
                nn.Conv2d(3, 64, kernel_size=4, stride=2, padding=1),
                nn.LeakyReLU(0.2),
                nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
                nn.BatchNorm2d(128),
                nn.LeakyReLU(0.2),
                nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1),
                nn.BatchNorm2d(256),
                nn.LeakyReLU(0.2),
                nn.Conv2d(256, 512, kernel_size=4, stride=2, padding=1),
                nn.BatchNorm2d(512),
                nn.LeakyReLU(0.2),
                nn.AdaptiveAvgPool2d((1, 1)),
            )
            self.feature_dim = 512
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Extract features from images.
        
        Args:
            x: Input images (B, C, H, W), values in [0, 1] or [-1, 1]
            
        Returns:
            Feature vectors of shape (B, feature_dim)
        """
        # Normalize to [-1, 1] if not already
        if x.min() >= 0:
            x = 2 * x - 1
        
        # Expand grayscale to RGB if needed
        if x.size(1) == 1:
            x = x.repeat(1, 3, 1, 1)
        
        # Resize to 299x299 for Inception if needed
        if self.use_inception and (x.size(2) != 299 or x.size(3) != 299):
            x = nn.functional.interpolate(
                x, size=(299, 299), mode='bilinear', align_corners=False
            )
        
        features = self.features(x)
        return features.view(features.size(0), -1)


def compute_statistics(features: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute mean and covariance of feature vectors.
    
    Args:
        features: Feature array of shape (N, D)
        
    Returns:
        Tuple of (mean, covariance)
    """
    mu = np.mean(features, axis=0)
    sigma = np.cov(features, rowvar=False)
    return mu, sigma


def calculate_frechet_distance(mu1: np.ndarray, sigma1: np.ndarray,
                                mu2: np.ndarray, 
                                sigma2: np.ndarray) -> float:
    """
    Calculate Fréchet distance between two multivariate Gaussians.
    
    FID = ||mu1 - mu2||^2 + Tr(sigma1 + sigma2 - 2*sqrt(sigma1*sigma2))
    
    Args:
        mu1: Mean of first distribution
        sigma1: Covariance of first distribution
        mu2: Mean of second distribution
        sigma2: Covariance of second distribution
        
    Returns:
        Fréchet distance value
    """
    mu1 = np.atleast_1d(mu1)
    mu2 = np.atleast_1d(mu2)
    sigma1 = np.atleast_2d(sigma1)
    sigma2 = np.atleast_2d(sigma2)
    
    assert mu1.shape == mu2.shape, "Mean vectors have different shapes"
    assert sigma1.shape == sigma2.shape, "Covariance matrices have different shapes"
    
    diff = mu1 - mu2
    
    # Product of covariances
    covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)
    
    # Handle numerical errors
    if not np.isfinite(covmean).all():
        msg = "FID calculation produced singular product; adding epsilon to diagonal"
        warnings.warn(msg)
        offset = np.eye(sigma1.shape[0]) * 1e-6
        covmean = linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))
    
    # Handle imaginary components (numerical artifacts)
    if np.iscomplexobj(covmean):
        if not np.allclose(np.diagonal(covmean).imag, 0, atol=1e-3):
            m = np.max(np.abs(covmean.imag))
            raise ValueError(f"Imaginary component {m}")
        covmean = covmean.real
    
    # FID formula
    fid = diff.dot(diff) + np.trace(sigma1) + np.trace(sigma2) - 2 * np.trace(covmean)
    
    return float(fid)


def extract_features(images: torch.Tensor, 
                     extractor: InceptionFeatureExtractor,
                     device: torch.device,
                     batch_size: int = 50) -> np.ndarray:
    """
    Extract features from a batch of images.
    
    Args:
        images: Image tensor of shape (N, C, H, W)
        extractor: Feature extraction model
        device: Device to run extraction on
        batch_size: Batch size for processing
        
    Returns:
        Feature array of shape (N, feature_dim)
    """
    extractor.eval()
    extractor.to(device)
    
    all_features = []
    
    with torch.no_grad():
        for i in range(0, len(images), batch_size):
            batch = images[i:i + batch_size].to(device)
            features = extractor(batch)
            all_features.append(features.cpu().numpy())
    
    return np.concatenate(all_features, axis=0)


def calculate_fid(real_images: torch.Tensor, 
                  fake_images: torch.Tensor,
                  device: torch.device,
                  extractor: Optional[InceptionFeatureExtractor] = None,
                  batch_size: int = 50) -> float:
    """
    Calculate FID between real and generated images.
    
    Args:
        real_images: Real image tensor of shape (N, C, H, W)
        fake_images: Generated image tensor of shape (N, C, H, W)
        device: Device to run computation on
        extractor: Feature extractor (created if None)
        batch_size: Batch size for feature extraction
        
    Returns:
        FID score (lower is better)
    """
    if extractor is None:
        extractor = InceptionFeatureExtractor(use_inception=False)
    
    # Extract features
    real_features = extract_features(real_images, extractor, device, batch_size)
    fake_features = extract_features(fake_images, extractor, device, batch_size)
    
    # Compute statistics
    mu_real, sigma_real = compute_statistics(real_features)
    mu_fake, sigma_fake = compute_statistics(fake_features)
    
    # Calculate FID
    fid = calculate_frechet_distance(mu_real, sigma_real, mu_fake, sigma_fake)
    
    return fid


def calculate_fid_from_loader(model: nn.Module,
                               data_loader: torch.utils.data.DataLoader,
                               device: torch.device,
                               num_samples: int = 1000,
                               extractor: Optional[InceptionFeatureExtractor] = None
                               ) -> float:
    """
    Calculate FID using a generative model and a data loader.
    
    Args:
        model: Generative model with sample() method
        data_loader: DataLoader containing real images
        device: Device to run computation on
        num_samples: Number of samples to generate
        extractor: Feature extractor (created if None)
        
    Returns:
        FID score
    """
    model.eval()
    
    # Collect real images
    real_images = []
    total_collected = 0
    
    for batch_data in data_loader:
        if isinstance(batch_data, (list, tuple)):
            images = batch_data[0]
        else:
            images = batch_data
        
        real_images.append(images)
        total_collected += images.size(0)
        
        if total_collected >= num_samples:
            break
    
    real_images = torch.cat(real_images, dim=0)[:num_samples]
    
    # Generate fake images
    with torch.no_grad():
        fake_images = model.sample(num_samples, device)
    
    # Normalize fake images to [0, 1] if they're in [-1, 1]
    if fake_images.min() < 0:
        fake_images = (fake_images + 1) / 2
    
    return calculate_fid(real_images, fake_images, device, extractor)
