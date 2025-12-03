"""
CIFAR-10 Dataset Loading Utilities

Preprocessing for continuous density models (VAE/Flows):
1. Uniform dequantization: Add u ~ U[0,1) to 8-bit discrete data
2. Standard normalization: Apply mean/std normalization
3. Validation split: Create validation set from training set
"""

import torch
import torchvision.transforms as transforms
from torchvision.datasets import CIFAR10
from torch.utils.data import DataLoader, random_split, Subset
import numpy as np


class UniformDequantization:
    """
    Add uniform noise u ~ U[0, 1) to dequantize 8-bit discrete images.
    This is required for training continuous density models (VAE/Flows) on discrete data.
    """
    def __call__(self, x):
        noise = torch.rand_like(x) / 256.0
        return x + noise


class StandardNormalization:
    def __init__(self):
        self.mean = torch.tensor([0.4914, 0.4822, 0.4465]).view(3, 1, 1)
        self.std = torch.tensor([0.2470, 0.2435, 0.2616]).view(3, 1, 1)
    
    def __call__(self, x):
        return (x - self.mean) / self.std
    
    def inverse(self, x):
        return x * self.std + self.mean


def get_cifar10_datasets(data_path='~/datasets', transform=None, apply_dequantization=True):
    if transform is None:
        if apply_dequantization:
            transform = transforms.Compose([
                transforms.ToTensor(),
                UniformDequantization(),
                StandardNormalization(),
            ])
        else:
            transform = transforms.Compose([
                transforms.ToTensor(),
            ])
    
    train_dataset = CIFAR10(
        root=data_path,
        train=True,
        transform=transform,
        download=True
    )
    
    test_dataset = CIFAR10(
        root=data_path,
        train=False,
        transform=transform,
        download=True
    )
    
    print(f"CIFAR-10 Dataset Loaded:")
    print(f"  Training samples: {len(train_dataset)}")
    print(f"  Test samples: {len(test_dataset)}")
    print(f"  Image shape: {train_dataset[0][0].shape}")
    if apply_dequantization:
        print(f"  Preprocessing: Uniform dequantization + Standard normalization")
    
    return train_dataset, test_dataset


def get_cifar10_loaders(batch_size=128, data_path='~/datasets', 
                        transform=None, num_workers=0, shuffle_train=True,
                        apply_dequantization=True, validation_split=0.1):
    train_dataset, test_dataset = get_cifar10_datasets(data_path, transform, apply_dequantization)
    
    val_loader = None
    if validation_split > 0:
        total_train = len(train_dataset)
        val_size = int(total_train * validation_split)
        train_size = total_train - val_size
        
        train_subset, val_subset = random_split(
            train_dataset, 
            [train_size, val_size],
            generator=torch.Generator().manual_seed(42)
        )
        
        train_loader = DataLoader(
            dataset=train_subset,
            batch_size=batch_size,
            shuffle=shuffle_train,
            num_workers=num_workers,
            drop_last=False,
            pin_memory=torch.cuda.is_available()
        )
        
        val_loader = DataLoader(
            dataset=val_subset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            drop_last=False,
            pin_memory=torch.cuda.is_available()
        )
        
        print(f"\nDataLoaders Created with Validation Split:")
        print(f"  Training samples: {train_size}")
        print(f"  Validation samples: {val_size}")
        print(f"  Test samples: {len(test_dataset)}")
    else:
        train_loader = DataLoader(
            dataset=train_dataset,
            batch_size=batch_size,
            shuffle=shuffle_train,
            num_workers=num_workers,
            drop_last=False,
            pin_memory=torch.cuda.is_available()
        )
        
        print(f"\nDataLoaders Created:")
        print(f"  Training samples: {len(train_dataset)}")
        print(f"  Test samples: {len(test_dataset)}")
    
    test_loader = DataLoader(
        dataset=test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        drop_last=False,
        pin_memory=torch.cuda.is_available()
    )
    
    print(f"  Batch size: {batch_size}")
    print(f"  Training batches: {len(train_loader)}")
    if val_loader:
        print(f"  Validation batches: {len(val_loader)}")
    print(f"  Test batches: {len(test_loader)}")
    
    return train_loader, val_loader, test_loader

# CIFAR-10 class names for reference
CIFAR10_CLASSES = [
    'airplane', 'automobile', 'bird', 'cat', 'deer',
    'dog', 'frog', 'horse', 'ship', 'truck'
]


if __name__ == "__main__":
    print("CIFAR-10 data loading...")
    train_loader, test_loader = get_cifar10_loaders(batch_size=128)
    
    images, labels = next(iter(train_loader))
    print(f"\nSample batch:")
    print(f"  Images shape: {images.shape}")
    print(f"  Labels shape: {labels.shape}")
    print(f"  Image value range: [{images.min():.3f}, {images.max():.3f}]")

