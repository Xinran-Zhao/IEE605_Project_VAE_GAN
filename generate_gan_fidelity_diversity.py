import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from tqdm import tqdm
from torch_fidelity import calculate_metrics
from models.GAN import Generator
import torchvision.transforms as transforms
from torchvision.datasets import CIFAR10
import shutil

# --- Setup Configuration ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

if str(device) == 'cpu':
    print("WARNING: You are running on CPU. This will be very slow.")

# --- Parameters ---
latent_dim = 256
num_samples = 5000  # 50k is academic standard, 5k is good for experiments
temperatures = [0.3, 0.5, 0.7, 0.8, 1.0, 1.2]
model_path = "models/gan_checkpoint.pt"

# --- Output Directories ---
output_base = "fidelity_diversity_gan"
os.makedirs(output_base, exist_ok=True)

# --- Prepare Real Data (Required for FID/Recall) ---
real_data_dir = os.path.join(output_base, "real_cifar10")
if not os.path.exists(real_data_dir):
    print("Preparing real CIFAR-10 dataset for FID calculation...")
    os.makedirs(real_data_dir, exist_ok=True)
    
    # Download and extract real images
    transform = transforms.Compose([transforms.ToTensor()])
    cifar_dataset = CIFAR10(root='./datasets', train=False, transform=transform, download=True)
    
    # Save real images (limit to num_samples to save time/space if needed, 
    # though using full test set is better for accuracy)
    limit_real = min(len(cifar_dataset), num_samples)
    for idx in tqdm(range(limit_real), desc="Saving real images"):
        img, _ = cifar_dataset[idx]
        img = transforms.ToPILImage()(img)
        img.save(os.path.join(real_data_dir, f"real_{idx:05d}.png"))
    print(f"Saved {limit_real} real images to {real_data_dir}")


def load_generator(model_path, latent_dim, device):
    """Load the trained GAN generator model."""
    G = Generator(latent_dim=latent_dim, device=device).to(device)
    
    if os.path.exists(model_path):
        # Handle different saving methods (whole model vs state_dict)
        checkpoint = torch.load(model_path, map_location=device)
        if isinstance(checkpoint, dict) and 'generator' in checkpoint:
            G.load_state_dict(checkpoint['generator'])
        else:
            G.load_state_dict(checkpoint)
        print(f"Loaded model from {model_path}")
    else:
        print(f"Warning: Model file {model_path} not found. Using randomly initialized model!")
        print("Please ensure your model path is correct.")
    
    G.eval()
    return G


def generate_images_with_temperature(G, num_samples, temperature, save_dir, latent_dim, device):
    """
    Generate images using a specific 'temperature' (variance scaling).
    Lower temperature = Reduced variance = potentially higher fidelity, lower diversity.
    """
    if os.path.exists(save_dir):
        # Optional: clear directory to ensure clean metrics
        shutil.rmtree(save_dir)
    os.makedirs(save_dir, exist_ok=True)
    
    G.eval()
    batch_size = 64  # Increase this if your GPU has enough VRAM
    
    print(f"Generating {num_samples} images with Temperature={temperature}...")
    
    with torch.no_grad():
        num_batches = (num_samples + batch_size - 1) // batch_size
        generated_count = 0
        
        for _ in tqdm(range(num_batches), desc=f"  Generating (T={temperature})"):
            # Calculate current batch size
            current_batch_size = min(batch_size, num_samples - generated_count)
            if current_batch_size <= 0:
                break
                
            # Sample latent vector z ~ N(0, I) and apply temperature scaling
            # T < 1.0 reduces magnitude (truncation-like effect)
            z = torch.randn(current_batch_size, latent_dim, device=device) * temperature
            
            # Generate images
            img_tensor = G(z)
            
            # Normalize from [-1, 1] to [0, 1]
            img_tensor = (img_tensor + 1) / 2
            img_tensor = torch.clamp(img_tensor, 0, 1)
            
            # Save batch
            for j in range(current_batch_size):
                img = transforms.ToPILImage()(img_tensor[j].cpu())
                img.save(os.path.join(save_dir, f"gen_{generated_count:05d}.png"))
                generated_count += 1


def calculate_fidelity_diversity_metrics(real_dir, fake_dir):
    print(f"Calculating FID for: {fake_dir}")
    try:
        metrics = calculate_metrics(
            input1=fake_dir,
            input2=real_dir,
            cuda=torch.cuda.is_available(),
            isc=False,
            fid=True,
            kid=False,
            prc=False,
            verbose=False
        )
        
        fid = metrics['frechet_inception_distance']
        print(f"  -> FID: {fid:.4f}")
        
        return fid

    except Exception as e:
        print(f"Error calculating metrics: {e}")
        return None


def plot_fidelity_diversity_curve(results, save_path):
    temperatures = [r['temperature'] for r in results]
    fids = [r['fid'] for r in results]
    
    plt.figure(figsize=(10, 6))
    plt.plot(temperatures, fids, 'o-', linewidth=3, markersize=10, color='crimson')
    
    for i, temp in enumerate(temperatures):
        plt.annotate(f'{fids[i]:.2f}', (temperatures[i], fids[i]), 
                     textcoords="offset points", xytext=(0, 10), ha='center', fontsize=10)
    
    plt.xlabel('Temperature', fontsize=14, fontweight='bold')
    plt.ylabel('FID', fontsize=14, fontweight='bold')
    plt.title('Effect of Temperature on Fidelity (FID)', fontsize=16, fontweight='bold')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"\nSaved FID curve to {save_path}")
    plt.close()


def main():
    print(f"Starting analysis on {device}...")
    
    G = load_generator(model_path, latent_dim, device)
    
    results = []
    
    for temp in temperatures:
        temp_dir = os.path.join(output_base, f"temp_{temp}")
        
        generate_images_with_temperature(G, num_samples, temp, temp_dir, latent_dim, device)

        fid = calculate_fidelity_diversity_metrics(real_data_dir, temp_dir)
        
        if fid is not None:
            results.append({
                'temperature': temp,
                'fid': fid
            })
    
    results_file = os.path.join(output_base, 'metrics_results.txt')
    with open(results_file, 'w') as f:
        f.write("Temperature\tFID\n")
        for r in results:
            f.write(f"{r['temperature']}\t{r['fid']:.4f}\n")
    print(f"\nSaved results to {results_file}")
    
    if len(results) > 1:
        plot_path = os.path.join(output_base, 'fidelity_temperature_curve.png')
        plot_fidelity_diversity_curve(results, plot_path)
        
        print("\nSUMMARY")
        print(f"{'Temperature':<12} {'FID':<10}")
        for r in results:
            print(f"{r['temperature']:<12.1f} {r['fid']:<10.2f}")
    else:
        print("Not enough data points to plot curve.")

if __name__ == "__main__":
    main()