import os
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from torch.optim import Adam
from models.GAN import Generator, Discriminator

torch.manual_seed(42)
np.random.seed(42)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

batch_size = 128
lr_g = 2e-4
lr_d = 2e-4
epochs = 20
latent_dim = 256
save_interval = 1

import torchvision.transforms as transforms
from torchvision.datasets import CIFAR10
from torch.utils.data import DataLoader

# GAN needs data in [-1, 1] range to match Tanh output
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # [0,1] -> [-1,1]
])
train_dataset = CIFAR10(root='~/datasets', train=True, transform=transform, download=True)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
print(f"CIFAR-10 loaded: {len(train_dataset)} samples")


def train(G, D, g_optimizer, d_optimizer, criterion, epochs, device):
    g_losses, d_losses = [], []
    fixed_z = torch.randn(25, latent_dim, device=device)
    
    for epoch in range(epochs):
        g_loss_epoch, d_loss_epoch = 0, 0
        num_batches = 0
        
        for batch_idx, (real_imgs, _) in enumerate(train_loader):
            bs = real_imgs.size(0)
            real_imgs = real_imgs.to(device)
            real_label = torch.ones(bs, 1, device=device)
            fake_label = torch.zeros(bs, 1, device=device)
            
            d_optimizer.zero_grad()
            real_out = D(real_imgs)
            d_loss_real = criterion(real_out, real_label)
            
            z = torch.randn(bs, latent_dim, device=device)
            fake_imgs = G(z)
            fake_out = D(fake_imgs.detach())
            d_loss_fake = criterion(fake_out, fake_label)
            
            d_loss = d_loss_real + d_loss_fake
            d_loss.backward()
            d_optimizer.step()
            
            g_optimizer.zero_grad()
            fake_out = D(fake_imgs)
            g_loss = criterion(fake_out, real_label)
            g_loss.backward()
            g_optimizer.step()
            
            g_loss_epoch += g_loss.item()
            d_loss_epoch += d_loss.item()
            num_batches += 1
        
        g_losses.append(g_loss_epoch / num_batches)
        d_losses.append(d_loss_epoch / num_batches)
        
        print(f"Epoch {epoch+1}/{epochs} | D Loss: {d_losses[-1]:.4f} | G Loss: {g_losses[-1]:.4f}")
        if (epoch + 1) % save_interval == 0 or epoch == 0:
            save_samples(G, fixed_z, epoch + 1, f'progress_gan/samples_epoch_{epoch+1:03d}.png')
    
    return g_losses, d_losses


def save_samples(G, z, epoch, save_path):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    G.eval()
    with torch.no_grad():
        samples = G(z).cpu()
        samples = (samples + 1) / 2
    
    fig, axes = plt.subplots(5, 5, figsize=(10, 10))
    for i, ax in enumerate(axes.flat):
        img = samples[i].permute(1, 2, 0).numpy()
        img = np.clip(img, 0, 1)
        ax.imshow(img)
        ax.axis('off')
    plt.suptitle(f'Epoch {epoch}', fontsize=14)
    plt.tight_layout()
    plt.savefig(save_path, dpi=100)
    plt.close()
    G.train()


if __name__ == "__main__":
    os.makedirs('progress_gan', exist_ok=True)
    
    G = Generator(latent_dim=latent_dim, device=device).to(device)
    D = Discriminator(device=device).to(device)
    
    g_optimizer = Adam(G.parameters(), lr=lr_g, betas=(0.5, 0.999))
    d_optimizer = Adam(D.parameters(), lr=lr_d, betas=(0.5, 0.999))
    criterion = nn.BCEWithLogitsLoss()
    
    print(f"G params: {sum(p.numel() for p in G.parameters()):,}")
    print(f"D params: {sum(p.numel() for p in D.parameters()):,}")
    print(f"Training: {len(train_loader)} batches per epoch\n")
    
    g_losses, d_losses = train(G, D, g_optimizer, d_optimizer, criterion, epochs, device)
