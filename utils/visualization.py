"""
Visualization utilities for VAE and GAN models.

Provides functions for saving generated images and plotting training losses.
"""

import torch
import numpy as np
from typing import List, Optional, Union
from pathlib import Path


def save_images(images: torch.Tensor, 
                path: Union[str, Path],
                nrow: int = 8,
                normalize: bool = True) -> None:
    """
    Save a grid of images to file.
    
    Args:
        images: Image tensor of shape (N, C, H, W)
        path: Path to save the image
        nrow: Number of images per row
        normalize: Whether to normalize images to [0, 1]
    """
    from torchvision.utils import save_image
    
    if normalize and images.min() < 0:
        # Normalize from [-1, 1] to [0, 1]
        images = (images + 1) / 2
    
    save_image(images, path, nrow=nrow, normalize=False)


def make_grid(images: torch.Tensor, 
              nrow: int = 8,
              padding: int = 2,
              normalize: bool = True) -> np.ndarray:
    """
    Create a grid of images as a numpy array.
    
    Args:
        images: Image tensor of shape (N, C, H, W)
        nrow: Number of images per row
        padding: Padding between images
        normalize: Whether to normalize images to [0, 1]
        
    Returns:
        Grid image as numpy array (H, W, C)
    """
    from torchvision.utils import make_grid as tv_make_grid
    
    if normalize and images.min() < 0:
        images = (images + 1) / 2
    
    grid = tv_make_grid(images, nrow=nrow, padding=padding, normalize=False)
    grid = grid.permute(1, 2, 0).cpu().numpy()
    
    return grid


def plot_losses(losses: dict, 
                save_path: Optional[Union[str, Path]] = None,
                title: str = 'Training Losses') -> None:
    """
    Plot training losses over epochs.
    
    Args:
        losses: Dictionary mapping loss names to lists of values
        save_path: Path to save the plot (optional)
        title: Plot title
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not installed. Skipping plot.")
        return
    
    plt.figure(figsize=(10, 6))
    
    for name, values in losses.items():
        plt.plot(values, label=name)
    
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title(title)
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
    else:
        plt.show()


def plot_samples(real_images: torch.Tensor,
                 fake_images: torch.Tensor,
                 save_path: Optional[Union[str, Path]] = None,
                 nrow: int = 8) -> None:
    """
    Plot comparison of real and generated images.
    
    Args:
        real_images: Real image tensor
        fake_images: Generated image tensor
        save_path: Path to save the plot
        nrow: Number of images per row
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not installed. Skipping plot.")
        return
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 7))
    
    real_grid = make_grid(real_images[:nrow*nrow], nrow=nrow)
    fake_grid = make_grid(fake_images[:nrow*nrow], nrow=nrow)
    
    axes[0].imshow(real_grid.squeeze(), cmap='gray' if real_grid.shape[-1] == 1 else None)
    axes[0].set_title('Real Images')
    axes[0].axis('off')
    
    axes[1].imshow(fake_grid.squeeze(), cmap='gray' if fake_grid.shape[-1] == 1 else None)
    axes[1].set_title('Generated Images')
    axes[1].axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
    else:
        plt.show()


def interpolate_latent(model: torch.nn.Module,
                       z1: torch.Tensor,
                       z2: torch.Tensor,
                       steps: int = 10) -> torch.Tensor:
    """
    Interpolate between two latent vectors and generate images.
    
    Args:
        model: Generative model with decode method
        z1: Starting latent vector
        z2: Ending latent vector
        steps: Number of interpolation steps
        
    Returns:
        Tensor of interpolated images
    """
    model.eval()
    
    # Linear interpolation
    alphas = torch.linspace(0, 1, steps).view(-1, 1)
    z1 = z1.view(1, -1)
    z2 = z2.view(1, -1)
    
    interpolated = (1 - alphas) * z1 + alphas * z2
    
    with torch.no_grad():
        if hasattr(model, 'decode'):
            images = model.decode(interpolated.to(z1.device))
        else:
            images = model(interpolated.to(z1.device))
    
    return images
