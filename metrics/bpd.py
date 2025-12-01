"""
Bits Per Dimension (BPD) metric for VAE evaluation.

BPD is a commonly used metric to evaluate the density estimation 
performance of generative models like VAEs. Lower BPD indicates better 
compression/representation of the data.

BPD = NLL / (log(2) * D)

where:
- NLL: Negative log-likelihood 
- D: Number of dimensions in the data
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Tuple


def compute_elbo(recon_x: torch.Tensor, x: torch.Tensor,
                 mu: torch.Tensor, logvar: torch.Tensor,
                 reduction: str = 'sum') -> Tuple[torch.Tensor, torch.Tensor, 
                                                    torch.Tensor]:
    """
    Compute Evidence Lower Bound (ELBO) for VAE.
    
    ELBO = E_q[log p(x|z)] - KL(q(z|x) || p(z))
    
    Args:
        recon_x: Reconstructed images
        x: Original images  
        mu: Latent means
        logvar: Latent log variances
        reduction: How to reduce the loss ('sum', 'mean', 'none')
        
    Returns:
        Tuple of (elbo, reconstruction_term, kl_term)
    """
    batch_size = x.size(0)
    
    # Flatten for computation
    recon_x_flat = recon_x.view(batch_size, -1)
    x_flat = x.view(batch_size, -1)
    
    # Clamp for numerical stability instead of adding epsilon
    recon_x_flat = torch.clamp(recon_x_flat, 1e-8, 1 - 1e-8)
    
    # Reconstruction term: log p(x|z)
    # Using Bernoulli likelihood for binary data
    recon_term = torch.sum(
        x_flat * torch.log(recon_x_flat) + 
        (1 - x_flat) * torch.log(1 - recon_x_flat),
        dim=1
    )
    
    # KL divergence: KL(q(z|x) || p(z))
    # For Gaussian prior p(z) = N(0, I) and posterior q(z|x) = N(mu, sigma^2)
    kl_term = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1)
    
    # ELBO = reconstruction - KL
    elbo = recon_term - kl_term
    
    if reduction == 'sum':
        return elbo.sum(), recon_term.sum(), kl_term.sum()
    elif reduction == 'mean':
        return elbo.mean(), recon_term.mean(), kl_term.mean()
    else:
        return elbo, recon_term, kl_term


def negative_log_likelihood(recon_x: torch.Tensor, x: torch.Tensor,
                            distribution: str = 'bernoulli') -> torch.Tensor:
    """
    Compute negative log-likelihood for reconstruction.
    
    Args:
        recon_x: Reconstructed images
        x: Original images
        distribution: Assumed data distribution ('bernoulli' or 'gaussian')
        
    Returns:
        NLL per sample
    """
    batch_size = x.size(0)
    recon_x_flat = recon_x.view(batch_size, -1)
    x_flat = x.view(batch_size, -1)
    
    if distribution == 'bernoulli':
        # Bernoulli NLL: -sum(x * log(p) + (1-x) * log(1-p))
        nll = -torch.sum(
            x_flat * torch.log(recon_x_flat + 1e-8) +
            (1 - x_flat) * torch.log(1 - recon_x_flat + 1e-8),
            dim=1
        )
    elif distribution == 'gaussian':
        # Gaussian NLL with unit variance: 0.5 * sum((x - mu)^2)
        nll = 0.5 * torch.sum((x_flat - recon_x_flat) ** 2, dim=1)
    else:
        raise ValueError(f"Unknown distribution: {distribution}")
    
    return nll


def bits_per_dimension(model: nn.Module, data_loader: torch.utils.data.DataLoader,
                       device: torch.device, 
                       num_importance_samples: int = 1) -> dict:
    """
    Compute Bits Per Dimension (BPD) for a VAE model.
    
    BPD measures how many bits are needed on average to encode each 
    dimension of the data using the learned model. Lower is better.
    
    Args:
        model: VAE model with encode() and decode() methods
        data_loader: DataLoader containing evaluation data
        device: Device to run computation on
        num_importance_samples: Number of samples for importance weighted 
                                estimate (higher = more accurate)
        
    Returns:
        Dictionary with 'bpd', 'nll', 'elbo', 'kl'
    """
    model.eval()
    
    total_nll = 0.0
    total_elbo = 0.0
    total_kl = 0.0
    total_samples = 0
    total_dims = None
    
    with torch.no_grad():
        for batch_data in data_loader:
            # Handle different data loader formats
            if isinstance(batch_data, (list, tuple)):
                x = batch_data[0]
            else:
                x = batch_data
                
            x = x.to(device)
            batch_size = x.size(0)
            
            if total_dims is None:
                # Calculate number of dimensions per sample
                total_dims = np.prod(x.shape[1:])
            
            if num_importance_samples == 1:
                # Standard ELBO estimate
                recon_x, mu, logvar = model(x)
                elbo, recon_term, kl_term = compute_elbo(
                    recon_x, x, mu, logvar, reduction='sum'
                )
                
                total_elbo += elbo.item()
                total_kl += kl_term.item()
                total_nll += (-elbo.item())  # NLL >= -ELBO
                
            else:
                # Importance weighted estimate for tighter bound
                log_weights = []
                
                for _ in range(num_importance_samples):
                    recon_x, mu, logvar = model(x)
                    elbo, _, _ = compute_elbo(
                        recon_x, x, mu, logvar, reduction='none'
                    )
                    log_weights.append(elbo)
                
                # Importance weighted ELBO: log(1/K * sum(exp(log_w)))
                log_weights = torch.stack(log_weights, dim=0)  # (K, B)
                iw_elbo = torch.logsumexp(log_weights, dim=0) - np.log(
                    num_importance_samples
                )
                
                total_elbo += iw_elbo.sum().item()
                total_nll += (-iw_elbo.sum().item())
            
            total_samples += batch_size
    
    # Compute averages
    avg_nll = total_nll / total_samples
    avg_elbo = total_elbo / total_samples
    avg_kl = total_kl / total_samples if num_importance_samples == 1 else 0.0
    
    # BPD = NLL / (log(2) * D)
    bpd = avg_nll / (np.log(2) * total_dims)
    
    return {
        'bpd': bpd,
        'nll': avg_nll,
        'elbo': avg_elbo,
        'kl': avg_kl,
        'dimensions': total_dims
    }


def compute_bpd_from_loss(loss: float, num_dims: int) -> float:
    """
    Convert reconstruction loss to bits per dimension.
    
    Args:
        loss: Total reconstruction loss (NLL)
        num_dims: Number of dimensions in the data
        
    Returns:
        Bits per dimension value
    """
    return loss / (np.log(2) * num_dims)
