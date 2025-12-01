"""
VAE Training Script.

This script trains a Variational Autoencoder on MNIST or other datasets.
"""

import argparse
import torch
import torch.optim as optim
from pathlib import Path
from tqdm import tqdm

from models import VAE
from metrics import bits_per_dimension
from utils import get_data_loader, save_images


def train_epoch(model: VAE, train_loader, optimizer, device, beta: float = 1.0):
    """Train for one epoch."""
    model.train()
    total_loss = 0
    total_recon = 0
    total_kl = 0
    
    for batch_data in tqdm(train_loader, desc='Training'):
        if isinstance(batch_data, (list, tuple)):
            x = batch_data[0]
        else:
            x = batch_data
        
        x = x.to(device)
        optimizer.zero_grad()
        
        recon_x, mu, logvar = model(x)
        loss, loss_dict = VAE.loss_function(recon_x, x, mu, logvar, beta)
        
        loss.backward()
        optimizer.step()
        
        total_loss += loss_dict['loss']
        total_recon += loss_dict['reconstruction_loss']
        total_kl += loss_dict['kl_loss']
    
    n = len(train_loader)
    return {
        'loss': total_loss / n,
        'recon': total_recon / n,
        'kl': total_kl / n
    }


def evaluate(model: VAE, test_loader, device):
    """Evaluate the model."""
    model.eval()
    
    # Compute BPD
    bpd_results = bits_per_dimension(model, test_loader, device)
    
    return bpd_results


def main():
    parser = argparse.ArgumentParser(description='Train VAE')
    parser.add_argument('--dataset', type=str, default='mnist',
                        choices=['mnist', 'fashion_mnist', 'cifar10'])
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--latent-dim', type=int, default=128)
    parser.add_argument('--beta', type=float, default=1.0,
                        help='Beta parameter for beta-VAE')
    parser.add_argument('--output-dir', type=str, default='./outputs/vae')
    parser.add_argument('--save-interval', type=int, default=10)
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
    
    # Model
    model = VAE(in_channels=in_channels, latent_dim=args.latent_dim).to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Training loop
    best_bpd = float('inf')
    
    for epoch in range(1, args.epochs + 1):
        # Train
        train_metrics = train_epoch(model, train_loader, optimizer, device, 
                                    args.beta)
        
        # Evaluate
        eval_metrics = evaluate(model, test_loader, device)
        
        print(f"Epoch {epoch}/{args.epochs}")
        print(f"  Train Loss: {train_metrics['loss']:.2f}, "
              f"Recon: {train_metrics['recon']:.2f}, KL: {train_metrics['kl']:.2f}")
        print(f"  Test BPD: {eval_metrics['bpd']:.4f}, "
              f"NLL: {eval_metrics['nll']:.2f}")
        
        # Save samples
        if epoch % args.save_interval == 0 or epoch == 1:
            with torch.no_grad():
                samples = model.sample(64, device)
            save_images(samples, output_dir / f'samples_epoch_{epoch}.png')
        
        # Save best model
        if eval_metrics['bpd'] < best_bpd:
            best_bpd = eval_metrics['bpd']
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'bpd': best_bpd,
            }, output_dir / 'best_model.pt')
    
    # Save final model
    torch.save({
        'epoch': args.epochs,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'bpd': eval_metrics['bpd'],
    }, output_dir / 'final_model.pt')
    
    print(f"\nTraining complete! Best BPD: {best_bpd:.4f}")


if __name__ == '__main__':
    main()
