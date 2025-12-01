"""
Generative Adversarial Network (GAN) implementation.

This module implements a Deep Convolutional GAN (DCGAN) with:
- Generator: Maps random noise to generated images
- Discriminator: Classifies images as real or fake
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple


class Generator(nn.Module):
    """DCGAN Generator network.
    
    Maps random noise (latent vector) to generated images.
    """
    
    def __init__(self, latent_dim: int = 100, out_channels: int = 1,
                 feature_maps: int = 64, img_size: int = 32):
        """
        Args:
            latent_dim: Dimension of the latent noise vector
            out_channels: Number of output channels
            feature_maps: Base number of feature maps
            img_size: Output image size (assumed square)
        """
        super().__init__()
        
        self.latent_dim = latent_dim
        self.img_size = img_size
        
        # Initial projection from latent space
        self.project = nn.Sequential(
            nn.Linear(latent_dim, feature_maps * 8 * 4 * 4),
            nn.ReLU(True)
        )
        
        # Transposed convolution layers
        self.main = nn.Sequential(
            # Input: (feature_maps*8) x 4 x 4
            nn.ConvTranspose2d(feature_maps * 8, feature_maps * 4,
                               kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(feature_maps * 4),
            nn.ReLU(True),
            # State: (feature_maps*4) x 8 x 8
            
            nn.ConvTranspose2d(feature_maps * 4, feature_maps * 2,
                               kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(feature_maps * 2),
            nn.ReLU(True),
            # State: (feature_maps*2) x 16 x 16
            
            nn.ConvTranspose2d(feature_maps * 2, feature_maps,
                               kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(feature_maps),
            nn.ReLU(True),
            # State: (feature_maps) x 32 x 32
            
            nn.Conv2d(feature_maps, out_channels,
                      kernel_size=3, stride=1, padding=1, bias=False),
            nn.Tanh()
            # Output: (out_channels) x 32 x 32
        )
        
        self.feature_maps = feature_maps
        
        # Initialize weights
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        """Initialize weights using DCGAN guidelines."""
        if isinstance(module, (nn.Conv2d, nn.ConvTranspose2d)):
            nn.init.normal_(module.weight.data, 0.0, 0.02)
        elif isinstance(module, nn.BatchNorm2d):
            nn.init.normal_(module.weight.data, 1.0, 0.02)
            nn.init.constant_(module.bias.data, 0)
        elif isinstance(module, nn.Linear):
            nn.init.normal_(module.weight.data, 0.0, 0.02)
    
    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the generator.
        
        Args:
            z: Latent noise vector of shape (batch_size, latent_dim)
            
        Returns:
            Generated images of shape (batch_size, channels, height, width)
        """
        # Project and reshape
        x = self.project(z)
        x = x.view(x.size(0), self.feature_maps * 8, 4, 4)
        return self.main(x)
    
    def sample(self, num_samples: int, device: torch.device) -> torch.Tensor:
        """
        Generate random samples.
        
        Args:
            num_samples: Number of samples to generate
            device: Device to create samples on
            
        Returns:
            Generated images
        """
        z = torch.randn(num_samples, self.latent_dim, device=device)
        return self.forward(z)


class Discriminator(nn.Module):
    """DCGAN Discriminator network.
    
    Classifies images as real or fake.
    """
    
    def __init__(self, in_channels: int = 1, feature_maps: int = 64):
        """
        Args:
            in_channels: Number of input channels
            feature_maps: Base number of feature maps
        """
        super().__init__()
        
        self.main = nn.Sequential(
            # Input: (in_channels) x 32 x 32
            nn.Conv2d(in_channels, feature_maps, kernel_size=4, 
                      stride=2, padding=1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # State: (feature_maps) x 16 x 16
            
            nn.Conv2d(feature_maps, feature_maps * 2, kernel_size=4,
                      stride=2, padding=1, bias=False),
            nn.BatchNorm2d(feature_maps * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # State: (feature_maps*2) x 8 x 8
            
            nn.Conv2d(feature_maps * 2, feature_maps * 4, kernel_size=4,
                      stride=2, padding=1, bias=False),
            nn.BatchNorm2d(feature_maps * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # State: (feature_maps*4) x 4 x 4
            
            nn.Conv2d(feature_maps * 4, feature_maps * 8, kernel_size=4,
                      stride=2, padding=1, bias=False),
            nn.BatchNorm2d(feature_maps * 8),
            nn.LeakyReLU(0.2, inplace=True),
            # State: (feature_maps*8) x 2 x 2
        )
        
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(feature_maps * 8 * 2 * 2, 1),
            nn.Sigmoid()
        )
        
        # Initialize weights
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        """Initialize weights using DCGAN guidelines."""
        if isinstance(module, (nn.Conv2d, nn.ConvTranspose2d)):
            nn.init.normal_(module.weight.data, 0.0, 0.02)
        elif isinstance(module, nn.BatchNorm2d):
            nn.init.normal_(module.weight.data, 1.0, 0.02)
            nn.init.constant_(module.bias.data, 0)
        elif isinstance(module, nn.Linear):
            nn.init.normal_(module.weight.data, 0.0, 0.02)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the discriminator.
        
        Args:
            x: Input images of shape (batch_size, channels, height, width)
            
        Returns:
            Probability that images are real
        """
        features = self.main(x)
        return self.classifier(features)
    
    def get_features(self, x: torch.Tensor) -> torch.Tensor:
        """Get intermediate features for feature matching."""
        return self.main(x)


def gan_loss(d_real: torch.Tensor, d_fake: torch.Tensor,
             real_label: float = 1.0, 
             fake_label: float = 0.0) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Compute GAN loss for discriminator and generator.
    
    Args:
        d_real: Discriminator output for real images
        d_fake: Discriminator output for fake images
        real_label: Label for real images
        fake_label: Label for fake images
        
    Returns:
        Tuple of (discriminator_loss, generator_loss)
    """
    batch_size = d_real.size(0)
    device = d_real.device
    
    # Labels
    real_labels = torch.full((batch_size, 1), real_label, device=device)
    fake_labels = torch.full((batch_size, 1), fake_label, device=device)
    
    # Discriminator loss using functional form
    d_loss_real = F.binary_cross_entropy(d_real, real_labels)
    d_loss_fake = F.binary_cross_entropy(d_fake, fake_labels)
    d_loss = d_loss_real + d_loss_fake
    
    # Generator loss (wants discriminator to think fakes are real)
    g_loss = F.binary_cross_entropy(d_fake, real_labels)
    
    return d_loss, g_loss


def wasserstein_loss(d_real: torch.Tensor, 
                     d_fake: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Compute Wasserstein GAN loss.
    
    Args:
        d_real: Discriminator/critic output for real images
        d_fake: Discriminator/critic output for fake images
        
    Returns:
        Tuple of (critic_loss, generator_loss)
    """
    # Critic loss: maximize E[D(real)] - E[D(fake)]
    # Equivalent to minimizing -E[D(real)] + E[D(fake)]
    critic_loss = -torch.mean(d_real) + torch.mean(d_fake)
    
    # Generator loss: maximize E[D(fake)]
    # Equivalent to minimizing -E[D(fake)]
    generator_loss = -torch.mean(d_fake)
    
    return critic_loss, generator_loss
