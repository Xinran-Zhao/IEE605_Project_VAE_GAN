"""
Variational Autoencoder (VAE) implementation.

This module implements a VAE with:
- Encoder: Maps input images to latent space (mean and log variance)
- Decoder: Reconstructs images from latent samples
- Reparameterization trick for backpropagation through sampling
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple


class Encoder(nn.Module):
    """VAE Encoder network.
    
    Maps input images to latent space parameters (mean and log variance).
    """
    
    def __init__(self, in_channels: int = 1, latent_dim: int = 128, 
                 hidden_dims: list = None):
        """
        Args:
            in_channels: Number of input channels (1 for grayscale, 3 for RGB)
            latent_dim: Dimension of the latent space
            hidden_dims: List of hidden layer dimensions
        """
        super().__init__()
        
        if hidden_dims is None:
            hidden_dims = [32, 64, 128, 256]
        
        self.latent_dim = latent_dim
        
        # Build encoder layers
        modules = []
        for h_dim in hidden_dims:
            modules.append(
                nn.Sequential(
                    nn.Conv2d(in_channels, h_dim, kernel_size=3, 
                             stride=2, padding=1),
                    nn.BatchNorm2d(h_dim),
                    nn.LeakyReLU(0.2)
                )
            )
            in_channels = h_dim
        
        self.encoder = nn.Sequential(*modules)
        
        # Calculate flattened size after convolutions
        # For 32x32 input with 4 layers of stride 2: 32 -> 16 -> 8 -> 4 -> 2
        self.flatten_size = hidden_dims[-1] * 2 * 2
        
        # Latent space projections
        self.fc_mu = nn.Linear(self.flatten_size, latent_dim)
        self.fc_logvar = nn.Linear(self.flatten_size, latent_dim)
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass of the encoder.
        
        Args:
            x: Input tensor of shape (batch_size, channels, height, width)
            
        Returns:
            Tuple of (mean, log_variance) tensors
        """
        h = self.encoder(x)
        h = h.view(h.size(0), -1)
        
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        
        return mu, logvar


class Decoder(nn.Module):
    """VAE Decoder network.
    
    Reconstructs images from latent samples.
    """
    
    def __init__(self, out_channels: int = 1, latent_dim: int = 128,
                 hidden_dims: list = None):
        """
        Args:
            out_channels: Number of output channels
            latent_dim: Dimension of the latent space
            hidden_dims: List of hidden layer dimensions (in reverse order)
        """
        super().__init__()
        
        if hidden_dims is None:
            hidden_dims = [256, 128, 64, 32]
        
        self.latent_dim = latent_dim
        self.hidden_dims = hidden_dims
        
        # Project latent to spatial feature map
        self.fc = nn.Linear(latent_dim, hidden_dims[0] * 2 * 2)
        
        # Build decoder layers
        modules = []
        for i in range(len(hidden_dims) - 1):
            modules.append(
                nn.Sequential(
                    nn.ConvTranspose2d(hidden_dims[i], hidden_dims[i + 1],
                                       kernel_size=4, stride=2, padding=1),
                    nn.BatchNorm2d(hidden_dims[i + 1]),
                    nn.LeakyReLU(0.2)
                )
            )
        
        self.decoder = nn.Sequential(*modules)
        
        # Final layer to output channels
        self.final_layer = nn.Sequential(
            nn.ConvTranspose2d(hidden_dims[-1], out_channels,
                               kernel_size=4, stride=2, padding=1),
            nn.Sigmoid()  # Output in [0, 1] range
        )
    
    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the decoder.
        
        Args:
            z: Latent tensor of shape (batch_size, latent_dim)
            
        Returns:
            Reconstructed image tensor
        """
        h = self.fc(z)
        h = h.view(h.size(0), self.hidden_dims[0], 2, 2)
        h = self.decoder(h)
        return self.final_layer(h)


class VAE(nn.Module):
    """Variational Autoencoder.
    
    Combines encoder and decoder with reparameterization trick.
    """
    
    def __init__(self, in_channels: int = 1, latent_dim: int = 128,
                 hidden_dims: list = None):
        """
        Args:
            in_channels: Number of input/output channels
            latent_dim: Dimension of the latent space
            hidden_dims: List of hidden layer dimensions
        """
        super().__init__()
        
        self.latent_dim = latent_dim
        
        if hidden_dims is None:
            hidden_dims = [32, 64, 128, 256]
        
        self.encoder = Encoder(in_channels, latent_dim, hidden_dims)
        self.decoder = Decoder(in_channels, latent_dim, 
                               list(reversed(hidden_dims)))
    
    def reparameterize(self, mu: torch.Tensor, 
                       logvar: torch.Tensor) -> torch.Tensor:
        """
        Reparameterization trick for VAE.
        
        Samples from N(mu, sigma^2) by sampling eps ~ N(0, 1) and computing
        z = mu + sigma * eps
        
        Args:
            mu: Mean of the latent distribution
            logvar: Log variance of the latent distribution
            
        Returns:
            Sampled latent vector
        """
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, 
                                                  torch.Tensor]:
        """
        Forward pass of the VAE.
        
        Args:
            x: Input tensor
            
        Returns:
            Tuple of (reconstructed, mu, logvar)
        """
        mu, logvar = self.encoder(x)
        z = self.reparameterize(mu, logvar)
        reconstructed = self.decoder(z)
        return reconstructed, mu, logvar
    
    def encode(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Encode input to latent parameters."""
        return self.encoder(x)
    
    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """Decode latent samples to images."""
        return self.decoder(z)
    
    def sample(self, num_samples: int, device: torch.device) -> torch.Tensor:
        """
        Sample from the latent space and generate images.
        
        Args:
            num_samples: Number of samples to generate
            device: Device to create samples on
            
        Returns:
            Generated image tensor
        """
        z = torch.randn(num_samples, self.latent_dim, device=device)
        return self.decode(z)
    
    @staticmethod
    def loss_function(recon_x: torch.Tensor, x: torch.Tensor,
                      mu: torch.Tensor, logvar: torch.Tensor,
                      beta: float = 1.0) -> Tuple[torch.Tensor, dict]:
        """
        Compute VAE loss (reconstruction + KL divergence).
        
        Args:
            recon_x: Reconstructed images
            x: Original images
            mu: Latent means
            logvar: Latent log variances
            beta: Weight for KL divergence term (for beta-VAE)
            
        Returns:
            Tuple of (total_loss, loss_dict)
        """
        # Reconstruction loss (binary cross entropy)
        recon_loss = F.binary_cross_entropy(recon_x, x, reduction='sum')
        
        # KL divergence loss
        # KL(q(z|x) || p(z)) = -0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
        kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        
        total_loss = recon_loss + beta * kl_loss
        
        loss_dict = {
            'loss': total_loss.item(),
            'reconstruction_loss': recon_loss.item(),
            'kl_loss': kl_loss.item()
        }
        
        return total_loss, loss_dict
