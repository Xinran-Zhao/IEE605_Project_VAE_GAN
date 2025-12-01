"""
Data loading utilities.

Provides data loaders for common datasets used in VAE and GAN training.
"""

import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from typing import Tuple, Optional


def get_data_loader(dataset_name: str = 'mnist',
                    batch_size: int = 64,
                    image_size: int = 32,
                    train: bool = True,
                    num_workers: int = 4,
                    data_dir: str = './data') -> DataLoader:
    """
    Get a data loader for the specified dataset.
    
    Args:
        dataset_name: Name of dataset ('mnist', 'fashion_mnist', 'cifar10')
        batch_size: Batch size for training
        image_size: Size to resize images to
        train: Whether to load training or test data
        num_workers: Number of data loading workers
        data_dir: Directory to download/load data from
        
    Returns:
        DataLoader for the specified dataset
    """
    dataset_name = dataset_name.lower()
    
    # Define transforms
    if dataset_name in ['mnist', 'fashion_mnist']:
        transform = transforms.Compose([
            transforms.Resize(image_size),
            transforms.ToTensor(),
        ])
    else:
        transform = transforms.Compose([
            transforms.Resize(image_size),
            transforms.ToTensor(),
        ])
    
    # Load dataset
    if dataset_name == 'mnist':
        dataset = datasets.MNIST(
            root=data_dir, train=train, download=True, transform=transform
        )
    elif dataset_name == 'fashion_mnist':
        dataset = datasets.FashionMNIST(
            root=data_dir, train=train, download=True, transform=transform
        )
    elif dataset_name == 'cifar10':
        dataset = datasets.CIFAR10(
            root=data_dir, train=train, download=True, transform=transform
        )
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")
    
    # Create data loader
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=train,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True
    )
    
    return loader


def get_dataset_info(dataset_name: str) -> dict:
    """
    Get information about a dataset.
    
    Args:
        dataset_name: Name of the dataset
        
    Returns:
        Dictionary with 'channels', 'image_size', 'num_classes'
    """
    dataset_name = dataset_name.lower()
    
    info = {
        'mnist': {'channels': 1, 'image_size': 28, 'num_classes': 10},
        'fashion_mnist': {'channels': 1, 'image_size': 28, 'num_classes': 10},
        'cifar10': {'channels': 3, 'image_size': 32, 'num_classes': 10},
    }
    
    if dataset_name not in info:
        raise ValueError(f"Unknown dataset: {dataset_name}")
    
    return info[dataset_name]
