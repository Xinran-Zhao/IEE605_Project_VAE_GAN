"""
GAN Training Script.

This script trains a Generative Adversarial Network on MNIST or other datasets.
"""

import argparse
import torch
import torch.optim as optim
from pathlib import Path
from tqdm import tqdm

from models import Generator, Discriminator
from models.gan import gan_loss
from metrics import calculate_fid, InceptionFeatureExtractor
from utils import get_data_loader, save_images


def train_epoch(generator, discriminator, train_loader, 
                g_optimizer, d_optimizer, device, latent_dim):
    """Train for one epoch."""
    generator.train()
    discriminator.train()
    
    total_g_loss = 0
    total_d_loss = 0
    
    for batch_data in tqdm(train_loader, desc='Training'):
        if isinstance(batch_data, (list, tuple)):
            real_images = batch_data[0]
        else:
            real_images = batch_data
        
        real_images = real_images.to(device)
        batch_size = real_images.size(0)
        
        # Normalize real images from [0, 1] to [-1, 1] for GAN (Tanh output)
        real_images = 2 * real_images - 1
        
        # Train Discriminator
        d_optimizer.zero_grad()
        
        # Real images
        d_real = discriminator(real_images)
        
        # Fake images
        z = torch.randn(batch_size, latent_dim, device=device)
        fake_images = generator(z)
        d_fake = discriminator(fake_images.detach())
        
        d_loss, _ = gan_loss(d_real, d_fake)
        d_loss.backward()
        d_optimizer.step()
        
        # Train Generator
        g_optimizer.zero_grad()
        
        z = torch.randn(batch_size, latent_dim, device=device)
        fake_images = generator(z)
        d_fake = discriminator(fake_images)
        
        _, g_loss = gan_loss(d_real.detach(), d_fake)
        g_loss.backward()
        g_optimizer.step()
        
        total_g_loss += g_loss.item()
        total_d_loss += d_loss.item()
    
    n = len(train_loader)
    return {
        'g_loss': total_g_loss / n,
        'd_loss': total_d_loss / n
    }


def evaluate(generator, test_loader, device, num_samples=1000):
    """Evaluate the generator using FID."""
    generator.eval()
    
    # Collect real images
    real_images = []
    for batch_data in test_loader:
        if isinstance(batch_data, (list, tuple)):
            images = batch_data[0]
        else:
            images = batch_data
        real_images.append(images)
        if len(real_images) * images.size(0) >= num_samples:
            break
    
    real_images = torch.cat(real_images, dim=0)[:num_samples]
    
    # Generate fake images
    with torch.no_grad():
        fake_images = generator.sample(num_samples, device)
        # Normalize from [-1, 1] to [0, 1]
        fake_images = (fake_images + 1) / 2
    
    # Calculate FID
    extractor = InceptionFeatureExtractor(use_inception=False)
    fid = calculate_fid(real_images, fake_images, device, extractor)
    
    return {'fid': fid}


def main():
    parser = argparse.ArgumentParser(description='Train GAN')
    parser.add_argument('--dataset', type=str, default='mnist',
                        choices=['mnist', 'fashion_mnist', 'cifar10'])
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--lr', type=float, default=2e-4)
    parser.add_argument('--latent-dim', type=int, default=100)
    parser.add_argument('--output-dir', type=str, default='./outputs/gan')
    parser.add_argument('--save-interval', type=int, default=10)
    parser.add_argument('--eval-interval', type=int, default=20)
    args = parser.parse_args()
    
    # Setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Data
    in_channels = 1 if args.dataset in ['mnist', 'fashion_mnist'] else 3
    train_loader = get_data_loader(args.dataset, args.batch_size, train=True)
    test_loader = get_data_loader(args.dataset, args.batch_size, train=False)
    
    # Models
    generator = Generator(latent_dim=args.latent_dim, 
                          out_channels=in_channels).to(device)
    discriminator = Discriminator(in_channels=in_channels).to(device)
    
    # Optimizers (using Adam with DCGAN hyperparameters)
    g_optimizer = optim.Adam(generator.parameters(), lr=args.lr, 
                              betas=(0.5, 0.999))
    d_optimizer = optim.Adam(discriminator.parameters(), lr=args.lr,
                              betas=(0.5, 0.999))
    
    print(f"Generator parameters: {sum(p.numel() for p in generator.parameters()):,}")
    print(f"Discriminator parameters: {sum(p.numel() for p in discriminator.parameters()):,}")
    
    # Fixed noise for visualization
    fixed_noise = torch.randn(64, args.latent_dim, device=device)
    
    # Training loop
    best_fid = float('inf')
    
    for epoch in range(1, args.epochs + 1):
        # Train
        train_metrics = train_epoch(
            generator, discriminator, train_loader,
            g_optimizer, d_optimizer, device, args.latent_dim
        )
        
        print(f"Epoch {epoch}/{args.epochs}")
        print(f"  G Loss: {train_metrics['g_loss']:.4f}, "
              f"D Loss: {train_metrics['d_loss']:.4f}")
        
        # Save samples
        if epoch % args.save_interval == 0 or epoch == 1:
            with torch.no_grad():
                samples = generator(fixed_noise)
                samples = (samples + 1) / 2  # Normalize to [0, 1]
            save_images(samples, output_dir / f'samples_epoch_{epoch}.png')
        
        # Evaluate FID
        if epoch % args.eval_interval == 0:
            eval_metrics = evaluate(generator, test_loader, device)
            print(f"  FID: {eval_metrics['fid']:.2f}")
            
            # Save best model
            if eval_metrics['fid'] < best_fid:
                best_fid = eval_metrics['fid']
                torch.save({
                    'epoch': epoch,
                    'generator_state_dict': generator.state_dict(),
                    'discriminator_state_dict': discriminator.state_dict(),
                    'g_optimizer_state_dict': g_optimizer.state_dict(),
                    'd_optimizer_state_dict': d_optimizer.state_dict(),
                    'fid': best_fid,
                }, output_dir / 'best_model.pt')
    
    # Save final model
    torch.save({
        'epoch': args.epochs,
        'generator_state_dict': generator.state_dict(),
        'discriminator_state_dict': discriminator.state_dict(),
        'g_optimizer_state_dict': g_optimizer.state_dict(),
        'd_optimizer_state_dict': d_optimizer.state_dict(),
    }, output_dir / 'final_model.pt')
    
    print(f"\nTraining complete! Best FID: {best_fid:.2f}")


if __name__ == '__main__':
    main()
