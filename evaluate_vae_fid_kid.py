import os
import json
import torch
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from PIL import Image
import torchvision.transforms as transforms
from torch_fidelity import calculate_metrics

from models.ConvVAE import ConvVAE


def load_vae_model(checkpoint_path, latent_dim, device):
    """
    Load trained VAE model from checkpoint
    
    Args:
        checkpoint_path: Path to checkpoint file
        latent_dim: Latent dimension size
        device: Device to load model on
    
    Returns:
        model: Loaded VAE model
    """
    print(f"Loading VAE model from: {checkpoint_path}")
    
    # Build model with same architecture as training
    model = ConvVAE(latent_dim=latent_dim, device=device)
    
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
        epoch = checkpoint.get('epoch', 'unknown')
        print(f"  Loaded checkpoint from epoch: {epoch}")
    else:
        # Direct state dict
        model.load_state_dict(checkpoint)
        print("  Loaded model state dict directly")
    
    model.to(device)
    model.eval()
    print("  Model loaded successfully!")
    
    return model


def generate_samples(model, num_samples, latent_dim, device, save_dir, batch_size=64):
    """
    Generate images from VAE model
    
    Args:
        model: Trained VAE model
        num_samples: Number of samples to generate
        latent_dim: Latent dimension size
        device: Device to run on
        save_dir: Directory to save generated images
        batch_size: Batch size for generation
    
    Returns:
        None (saves images to disk)
    """
    os.makedirs(save_dir, exist_ok=True)
    
    model.eval()
    generated_count = 0
    
    print(f"Generating {num_samples} samples from VAE...")
    
    num_batches = (num_samples + batch_size - 1) // batch_size
    
    with torch.no_grad():
        for _ in tqdm(range(num_batches), desc="Generating samples"):
            # Calculate current batch size
            current_batch_size = min(batch_size, num_samples - generated_count)
            if current_batch_size <= 0:
                break
            
            # Sample from standard normal distribution
            z = torch.randn(current_batch_size, latent_dim, device=device)
            
            # Generate images through decoder
            x = model.decode(z)  # Output is [0, 1] from Sigmoid
            
            # Clamp to valid range (should already be [0, 1])
            x = torch.clamp(x, 0, 1)
            
            # Save images
            for j in range(current_batch_size):
                img = transforms.ToPILImage()(x[j].cpu())
                img.save(os.path.join(save_dir, f"vae_gen_{generated_count:05d}.png"))
                generated_count += 1
    
    print(f"  Generated {generated_count} images saved to {save_dir}")


def visualize_samples(sample_dir, save_path, num_display=64):
    """
    Create a grid visualization of generated samples
    
    Args:
        sample_dir: Directory containing generated images
        save_path: Path to save visualization
        num_display: Number of samples to display (default: 64 = 8x8 grid)
    """
    print(f"\nCreating visualization of {num_display} samples...")
    
    # Get list of image files
    image_files = sorted([f for f in os.listdir(sample_dir) if f.endswith('.png')])[:num_display]
    
    # Calculate grid size
    grid_size = int(np.sqrt(num_display))
    
    fig, axes = plt.subplots(grid_size, grid_size, figsize=(12, 12))
    axes = axes.flatten()
    
    for idx, img_file in enumerate(image_files):
        img_path = os.path.join(sample_dir, img_file)
        img = Image.open(img_path)
        axes[idx].imshow(img)
        axes[idx].axis('off')
    
    # Turn off any extra subplots
    for idx in range(len(image_files), len(axes)):
        axes[idx].axis('off')
    
    plt.suptitle('VAE Generated Samples', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches='tight')
    plt.close()
    
    print(f"  Visualization saved to: {save_path}")


def calculate_fid_kid_metrics(real_dir, fake_dir):
    """
    Calculate FID and KID for generated images
    
    Args:
        real_dir: Directory containing real images
        fake_dir: Directory containing generated images
    
    Returns:
        dict: Metrics including FID and KID
    """
    print("\n" + "="*60)
    print("CALCULATING FID AND KID")
    print("="*60)
    print(f"Real images: {real_dir}")
    print(f"Generated images: {fake_dir}")
    
    # Count images
    num_real = len([f for f in os.listdir(real_dir) if f.endswith('.png')])
    num_fake = len([f for f in os.listdir(fake_dir) if f.endswith('.png')])
    print(f"\nReal images: {num_real}")
    print(f"Generated images: {num_fake}")
    print()
    
    print("Computing metrics (this may take several minutes)...")
    
    try:
        metrics = calculate_metrics(
            input1=fake_dir,
            input2=real_dir,
            cuda=torch.cuda.is_available(),
            isc=False,
            fid=True,
            kid=True,
            prc=False,
            verbose=True
        )
        
        return metrics
    
    except Exception as e:
        print(f"Error calculating metrics: {e}")
        return None


def main():
    # Configuration
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}\n")
    
    checkpoint_path = "./checkpoint/beta_vae/beta_0.00025/model.pth"
    latent_dim = 256
    num_samples = 5000
    output_dir = "vae_fid_kid_eval"
    generated_dir = os.path.join(output_dir, "generated_samples")
    real_dir = "fidelity_diversity_gan/real_cifar10"  # Reuse existing real images
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Check if real images exist
    if not os.path.exists(real_dir):
        print(f"ERROR: Real images directory not found: {real_dir}")
        print("Please run generate_gan_fidelity_diversity.py first to create real images.")
        return
    
    # Check if checkpoint exists
    if not os.path.exists(checkpoint_path):
        print(f"ERROR: Checkpoint not found: {checkpoint_path}")
        return
    
    print("="*60)
    print("VAE FID/KID EVALUATION")
    print("="*60)
    print(f"Checkpoint: {checkpoint_path}")
    print(f"Latent dimension: {latent_dim}")
    print(f"Number of samples: {num_samples}")
    print(f"Output directory: {output_dir}")
    print("="*60)
    print()
    
    # Step 1: Load model
    model = load_vae_model(checkpoint_path, latent_dim, device)
    print()
    
    # Step 2: Generate samples
    generate_samples(model, num_samples, latent_dim, device, generated_dir, batch_size=64)
    print()
    
    # Step 3: Visualize samples
    visualization_path = os.path.join(output_dir, "vae_generated_samples_grid.png")
    visualize_samples(generated_dir, visualization_path, num_display=64)
    print()
    
    # Step 4: Calculate FID and KID
    metrics = calculate_fid_kid_metrics(real_dir, generated_dir)
    
    if metrics is not None:
        print()
        print("="*60)
        print("RESULTS")
        print("="*60)
        
        # Extract metrics
        fid = metrics.get('frechet_inception_distance', 0)
        kid_mean = metrics.get('kernel_inception_distance_mean', 0)
        kid_std = metrics.get('kernel_inception_distance_std', 0)
        
        print(f"FID:        {fid:.4f}")
        print(f"KID (mean): {kid_mean:.6f}")
        print(f"KID (std):  {kid_std:.6f}")
        print(f"KID:        {kid_mean:.4f} ± {kid_std:.4f}")
        print("="*60)
        
        # Save results
        results = {
            'model': 'VAE',
            'checkpoint': checkpoint_path,
            'beta_weight': 0.00025,
            'latent_dim': latent_dim,
            'num_samples': num_samples,
            'fid': float(fid),
            'kid_mean': float(kid_mean),
            'kid_std': float(kid_std),
        }
        
        # Save to JSON
        json_path = os.path.join(output_dir, 'vae_fid_kid_results.json')
        with open(json_path, 'w') as f:
            json.dump(results, f, indent=4)
        print(f"\nResults saved to: {json_path}")
        
        # Save to text report
        report_path = os.path.join(output_dir, 'vae_fid_kid_report.txt')
        with open(report_path, 'w') as f:
            f.write("FID and KID Evaluation Report for VAE Model\n")
            f.write("="*60 + "\n\n")
            f.write(f"Model: Variational Autoencoder (VAE)\n")
            f.write(f"Checkpoint: {checkpoint_path}\n")
            f.write(f"Architecture:\n")
            f.write(f"  - Latent dimension: {latent_dim}\n")
            f.write(f"  - Beta weight (KLD): 0.00025\n")
            f.write(f"  - Encoder: 32x32x3 -> 4x4x512 -> {latent_dim}\n")
            f.write(f"  - Decoder: {latent_dim} -> 4x4x512 -> 32x32x3\n\n")
            f.write(f"Evaluation:\n")
            f.write(f"  - Number of generated samples: {num_samples}\n")
            f.write(f"  - Number of real samples: {len([f for f in os.listdir(real_dir) if f.endswith('.png')])}\n\n")
            f.write(f"Results:\n")
            f.write(f"  FID:        {fid:.4f}\n")
            f.write(f"  KID (mean): {kid_mean:.6f}\n")
            f.write(f"  KID (std):  {kid_std:.6f}\n")
            f.write(f"  KID:        {kid_mean:.4f} ± {kid_std:.4f}\n\n")
            f.write("="*60 + "\n")
            f.write("Note: Lower FID and KID values indicate better quality.\n")
            f.write("FID measures distribution distance assuming Gaussian features.\n")
            f.write("KID uses kernel methods and is more robust.\n")
            f.write("\nVAE generates images by sampling from latent space z ~ N(0, I).\n")
        
        print(f"Report saved to: {report_path}")
        print()
        print("✅ FID/KID evaluation complete!")
        
    else:
        print("❌ Failed to calculate metrics.")


if __name__ == "__main__":
    main()

