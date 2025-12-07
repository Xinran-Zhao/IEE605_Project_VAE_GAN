import torch
import numpy as np


def compute_rate_distortion(model, test_loader, device):
    model.eval()
    total_kld = 0.0
    total_mse = 0.0
    num_samples = 0
    
    with torch.no_grad():
        for x, _ in test_loader:
            x = x.to(device)
            x_recon, mu, logvar = model(x)
            
            kld = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
            mse = torch.sum((x - x_recon) ** 2)
            
            total_kld += kld.item()
            total_mse += mse.item()
            num_samples += x.size(0)
    
    avg_kld = total_kld / num_samples
    avg_mse = total_mse / (num_samples * x.size(1) * x.size(2) * x.size(3))
    
    rate_bits = avg_kld / np.log(2.0)
    distortion = avg_mse
    
    return rate_bits, distortion


def save_sample_images(model, device, beta_value, epoch, save_dir):
    import matplotlib.pyplot as plt
    import os
    
    model.eval()
    os.makedirs(save_dir, exist_ok=True)
    
    with torch.no_grad():
        z = torch.randn(16, model.latent_dim).to(device)
        samples = model.decode(z).cpu()
        
        fig, axes = plt.subplots(4, 4, figsize=(8, 8))
        for i, ax in enumerate(axes.flat):
            img = samples[i].permute(1, 2, 0).numpy()
            img = np.clip(img * 0.5 + 0.5, 0, 1)
            ax.imshow(img)
            ax.axis('off')
        
        plt.suptitle(f'Beta={beta_value}, Epoch={epoch}')
        plt.tight_layout()
        plt.savefig(f'{save_dir}/beta_{beta_value}_epoch_{epoch:03d}.png', dpi=150)
        plt.close()

