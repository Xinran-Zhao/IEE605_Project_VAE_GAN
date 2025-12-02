import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.optim import Adam
from models.ConvVAE import ConvVAE, loss_function
from data.cifar_10 import get_cifar10_loaders, StandardNormalization
import os

# Set random seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Create inverse normalization transform for visualization
inverse_norm = StandardNormalization().inverse

# Hyperparameters
batch_size = 128
learning_rate = 1e-3
epochs = 50
latent_dim = 256

# KLD weight annealing (CRITICAL for good results!)
kld_weight_start = 0.0
kld_weight_end = 0.00025  # Much smaller than 1.0!
kld_anneal_epochs = 50  # Gradually increase KLD weight over 50 epochs
# Load data with proper preprocessing and validation split
print("Loading CIFAR-10 dataset...")
train_loader, val_loader, test_loader = get_cifar10_loaders(
    batch_size=batch_size,
    data_path='~/datasets',
    num_workers=0,
    shuffle_train=True,
    apply_dequantization=True,  # Apply uniform dequantization + normalization
    validation_split=0.1  # Use 10% of training data for validation
)

def get_kld_weight(epoch):
    if epoch >= kld_anneal_epochs:
        return kld_weight_end
    return kld_weight_start + (kld_weight_end - kld_weight_start) * (epoch / kld_anneal_epochs)

def train(model, optimizer, epochs, device, save_interval=5):
    model.train()
    train_losses = []
    val_losses = []
    recon_losses_list = []
    kld_losses_list = []
    
    test_iter = iter(test_loader)
    fixed_test_images, _ = next(test_iter)
    fixed_test_images = fixed_test_images[:10].to(device)
    
    for epoch in range(epochs):
        overall_loss = 0
        overall_recon_loss = 0
        overall_kld_loss = 0
        num_batches = 0
        
        # Get current KLD weight (annealing)
        current_kld_weight = get_kld_weight(epoch)
        
        for batch_idx, (x, _) in enumerate(train_loader):
            # Images are already in (batch_size, 3, 32, 32) format
            x = x.to(device)
            
            optimizer.zero_grad()
            x_recon, mu, logvar = model(x)
            loss, recon_loss, kld_loss = loss_function(x, x_recon, mu, logvar, current_kld_weight)
            loss.backward()
            optimizer.step()
            
            overall_loss += loss.item()
            overall_recon_loss += recon_loss.item()
            overall_kld_loss += kld_loss.item()
            num_batches += 1
        
        avg_loss = overall_loss / num_batches
        avg_recon_loss = overall_recon_loss / num_batches
        avg_kld_loss = overall_kld_loss / num_batches
        
        train_losses.append(avg_loss)
        recon_losses_list.append(avg_recon_loss)
        kld_losses_list.append(avg_kld_loss)
        
        val_loss = validate(model, val_loader, device, current_kld_weight)
        val_losses.append(val_loss)
        
        print(f"Epoch {epoch + 1}/{epochs}")
        print(f"  Train Loss: {avg_loss:.6f}")
        print(f"  Val Loss: {val_loss:.6f}")
        print(f"  Recon Loss: {avg_recon_loss:.6f}")
        print(f"  KLD Loss: {avg_kld_loss:.6f}")
        print(f"  KLD Weight: {current_kld_weight:.6f}")
        
        if (epoch + 1) % save_interval == 0 or epoch == 0:
            model.eval()
            with torch.no_grad():
                reconstructed, _, _ = model(fixed_test_images)
            
            save_path = f'progress/reconstructed_epoch_{epoch+1:03d}.png'
            save_reconstruction_progress(fixed_test_images.cpu(), reconstructed.cpu(), 
                                        epoch + 1, save_path)
            print(f"  → Saved reconstruction progress: {save_path}")
            model.train()
    
    return train_losses, val_losses, recon_losses_list, kld_losses_list


def validate(model, val_loader, device, kld_weight):
    model.eval()
    overall_loss = 0
    num_batches = 0
    
    with torch.no_grad():
        for batch_idx, (x, _) in enumerate(val_loader):
            x = x.to(device)
            x_recon, mu, logvar = model(x)
            loss, _, _ = loss_function(x, x_recon, mu, logvar, kld_weight)
            overall_loss += loss.item()
            num_batches += 1
    
    model.train()
    return overall_loss / num_batches


def test(model, device):
    model.eval()
    overall_loss = 0
    num_batches = 0
    
    with torch.no_grad():
        for batch_idx, (x, _) in enumerate(test_loader):
            x = x.to(device)
            x_recon, mu, logvar = model(x)
            loss, _, _ = loss_function(x, x_recon, mu, logvar, 1)
            overall_loss += loss.item()
            num_batches += 1
    
    avg_loss = overall_loss / num_batches
    print(f"\n====> Test set loss: {avg_loss:.6f}")
    return avg_loss


def plot_losses(train_losses, val_losses, recon_losses, kld_losses, save_path='conv_training_losses.png'):
    """
    Plot training and validation losses
    
    Args:
        train_losses: list of total training losses
        val_losses: list of validation losses
        recon_losses: list of reconstruction losses
        kld_losses: list of KLD losses
        save_path: path to save the plot
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Total loss (train vs val)
    axes[0, 0].plot(range(1, len(train_losses) + 1), train_losses, 'b-', linewidth=2, label='Train')
    axes[0, 0].plot(range(1, len(val_losses) + 1), val_losses, 'r-', linewidth=2, label='Validation')
    axes[0, 0].set_xlabel('Epoch', fontsize=12)
    axes[0, 0].set_ylabel('Total Loss', fontsize=12)
    axes[0, 0].set_title('Total Loss (Train vs Validation)', fontsize=14)
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Reconstruction loss
    axes[0, 1].plot(range(1, len(recon_losses) + 1), recon_losses, 'g-', linewidth=2)
    axes[0, 1].set_xlabel('Epoch', fontsize=12)
    axes[0, 1].set_ylabel('Reconstruction Loss', fontsize=12)
    axes[0, 1].set_title('Reconstruction Loss (MSE)', fontsize=14)
    axes[0, 1].grid(True, alpha=0.3)
    
    # KLD loss
    axes[1, 0].plot(range(1, len(kld_losses) + 1), kld_losses, 'r-', linewidth=2)
    axes[1, 0].set_xlabel('Epoch', fontsize=12)
    axes[1, 0].set_ylabel('KLD Loss', fontsize=12)
    axes[1, 0].set_title('KL Divergence Loss', fontsize=14)
    axes[1, 0].grid(True, alpha=0.3)
    
    # Loss components stacked
    axes[1, 1].plot(range(1, len(train_losses) + 1), train_losses, 'b-', linewidth=2, label='Total')
    axes[1, 1].plot(range(1, len(recon_losses) + 1), recon_losses, 'g-', linewidth=2, label='Recon')
    axes[1, 1].plot(range(1, len(kld_losses) + 1), kld_losses, 'r-', linewidth=2, label='KLD')
    axes[1, 1].set_xlabel('Epoch', fontsize=12)
    axes[1, 1].set_ylabel('Loss', fontsize=12)
    axes[1, 1].set_title('Loss Components', fontsize=14)
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Training loss plots saved as '{save_path}'")
    plt.close()


def generate_samples(model, device, num_samples=25, save_path='conv_generated_samples.png'):
    model.eval()
    
    with torch.no_grad():
        z = torch.randn(num_samples, model.latent_dim).to(device)
        samples = model.decode(z).cpu()
        
        samples = inverse_norm(samples)
        
        fig, axes = plt.subplots(5, 5, figsize=(10, 10))
        for i, ax in enumerate(axes.flat):
            if i < num_samples:
                # Transpose from (C, H, W) to (H, W, C) for display
                img = samples[i].permute(1, 2, 0).numpy()
                img = np.clip(img, 0, 1)
                ax.imshow(img)
            ax.axis('off')
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Generated samples saved as '{save_path}'")
        plt.close()


def save_reconstruction_progress(images, reconstructed, epoch, save_path):

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    images = inverse_norm(images)
    reconstructed = inverse_norm(reconstructed)
    
    num_images = images.size(0)
    
    fig, axes = plt.subplots(2, num_images, figsize=(num_images * 2, 4))
    
    for i in range(num_images):
        orig_img = images[i].permute(1, 2, 0).numpy()
        orig_img = np.clip(orig_img, 0, 1)
        axes[0, i].imshow(orig_img)
        axes[0, i].axis('off')
        if i == 0:
            axes[0, i].set_ylabel('Original', fontsize=12, rotation=0, ha='right', va='center')
        
        recon_img = reconstructed[i].permute(1, 2, 0).numpy()
        recon_img = np.clip(recon_img, 0, 1)
        axes[1, i].imshow(recon_img)
        axes[1, i].axis('off')
        if i == 0:
            axes[1, i].set_ylabel('Reconstructed', fontsize=12, rotation=0, ha='right', va='center')
    
    plt.suptitle(f'Reconstruction Progress - Epoch {epoch}', fontsize=14, y=0.98)
    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches='tight')
    plt.close()


def reconstruct_images(model, device, num_images=10, save_path='conv_reconstructed_images.png'):
    model.eval()
    
    dataiter = iter(test_loader)
    images, _ = next(dataiter)
    images = images[:num_images].to(device)
    
    with torch.no_grad():
        reconstructed, _, _ = model(images)
    
    images = images.cpu()
    reconstructed = reconstructed.cpu()
    
    images = inverse_norm(images)
    reconstructed = inverse_norm(reconstructed)
    
    fig, axes = plt.subplots(2, num_images, figsize=(num_images * 2, 4))
    
    for i in range(num_images):
        orig_img = images[i].permute(1, 2, 0).numpy()
        orig_img = np.clip(orig_img, 0, 1)
        axes[0, i].imshow(orig_img)
        axes[0, i].axis('off')
        if i == 0:
            axes[0, i].set_title('Original', fontsize=12)
        
        recon_img = reconstructed[i].permute(1, 2, 0).numpy()
        recon_img = np.clip(recon_img, 0, 1)
        axes[1, i].imshow(recon_img)
        axes[1, i].axis('off')
        if i == 0:
            axes[1, i].set_title('Reconstructed', fontsize=12)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Reconstructed images saved as '{save_path}'")
    plt.close()


if __name__ == "__main__":
    import os
    os.makedirs('progress', exist_ok=True)
    
    model = ConvVAE(latent_dim=latent_dim, device=device).to(device)
    optimizer = Adam(model.parameters(), lr=learning_rate, betas=(0.9, 0.999))
    
    train_losses, val_losses, recon_losses, kld_losses = train(model, optimizer, epochs, device, save_interval=5)
    
    print("\nEvaluating on test set...")
    test_loss = test(model, device)
    
    plot_losses(train_losses, val_losses, recon_losses, kld_losses)
    
    print("\nGenerating samples from latent space...")
    generate_samples(model, device)
    
    print("\nReconstructing test images...")
    reconstruct_images(model, device)
    
    model_path = 'conv_vae_cifar10.pth'
    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'train_losses': train_losses,
        'val_losses': val_losses,
        'recon_losses': recon_losses,
        'kld_losses': kld_losses,
        'test_loss': test_loss,
        'latent_dim': latent_dim,
    }, model_path)
    print(f"\nModel saved as '{model_path}'")
    
    print(f"  Training Loss: {train_losses[-1]:.6f}")
    print(f"  Test Loss: {test_loss:.6f}")
