import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from tqdm import tqdm
import torchvision.transforms as transforms
from torchvision.datasets import CIFAR10
import json

# Setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Parameters
num_display_samples = 10  # Number of generated images to display
generated_images_dir = "fidelity_diversity_gan/temp_1.0"
output_dir = "gan_nearest_neighbor_results"
os.makedirs(output_dir, exist_ok=True)

# Seed for reproducibility
torch.manual_seed(42)
np.random.seed(42)


def load_cifar10_training_data():
    """Load CIFAR-10 training dataset (50,000 images)"""
    print("Loading CIFAR-10 training dataset...")
    
    # Use same normalization as GAN training: [-1, 1]
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    
    train_dataset = CIFAR10(root='~/datasets', train=True, transform=transform, download=True)
    
    # Load all training images into memory
    all_images = []
    print("Loading all training images into memory...")
    for i in tqdm(range(len(train_dataset)), desc="Loading CIFAR-10"):
        img, _ = train_dataset[i]
        all_images.append(img)
    
    # Stack into tensor: (50000, 3, 32, 32)
    all_images = torch.stack(all_images)
    print(f"Loaded {len(all_images)} training images, shape: {all_images.shape}")
    
    return all_images, train_dataset


def load_generated_images(gen_dir, num_samples):
    """Load generated images from directory"""
    print(f"\nLoading generated images from {gen_dir}...")
    
    # Get all image files
    image_files = sorted([f for f in os.listdir(gen_dir) if f.endswith('.png')])
    
    # Randomly select num_samples images
    selected_indices = np.random.choice(len(image_files), num_samples, replace=False)
    selected_files = [image_files[i] for i in selected_indices]
    
    # Load images
    generated_images = []
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # Same as training
    ])
    
    for filename in tqdm(selected_files, desc="Loading generated images"):
        img_path = os.path.join(gen_dir, filename)
        img = Image.open(img_path).convert('RGB')
        img_tensor = transform(img)
        generated_images.append(img_tensor)
    
    generated_images = torch.stack(generated_images)
    print(f"Loaded {len(generated_images)} generated images, shape: {generated_images.shape}")
    
    return generated_images, selected_files


def compute_nearest_neighbors(generated_images, training_images):
    """
    Compute nearest neighbor for each generated image in pixel space.
    
    Args:
        generated_images: (N, 3, 32, 32) tensor
        training_images: (50000, 3, 32, 32) tensor
    
    Returns:
        nearest_indices: (N,) indices of nearest training images
        nearest_distances: (N,) euclidean distances to nearest neighbors
    """
    print("\nComputing nearest neighbors in pixel space...")
    
    # Flatten images to vectors
    gen_flat = generated_images.view(generated_images.size(0), -1).to(device)  # (N, 3072)
    train_flat = training_images.view(training_images.size(0), -1).to(device)  # (50000, 3072)
    
    nearest_indices = []
    nearest_distances = []
    
    # Process each generated image
    for i in tqdm(range(gen_flat.size(0)), desc="Finding nearest neighbors"):
        gen_img = gen_flat[i:i+1]  # (1, 3072)
        
        # Compute Euclidean distance to all training images
        # distance = sqrt(sum((gen - train)^2))
        distances = torch.sqrt(torch.sum((gen_img - train_flat) ** 2, dim=1))  # (50000,)
        
        # Find minimum distance and its index
        min_dist, min_idx = torch.min(distances, dim=0)
        
        nearest_indices.append(min_idx.item())
        nearest_distances.append(min_dist.item())
    
    print(f"\nNearest neighbor search complete!")
    return nearest_indices, nearest_distances


def visualize_nearest_neighbors(generated_images, training_images, nearest_indices, 
                                  nearest_distances, save_path):
    """
    Create visualization grid: top row = generated, bottom row = nearest neighbor
    """
    print(f"\nCreating visualization...")
    
    num_samples = len(generated_images)
    # Adjust figure size for 10 images, make it more compact
    fig, axes = plt.subplots(2, num_samples, figsize=(num_samples * 1.2, 2.4))
    
    for i in range(num_samples):
        # Top row: generated image
        gen_img = generated_images[i].cpu()
        gen_img = (gen_img + 1) / 2  # Denormalize from [-1,1] to [0,1]
        gen_img = torch.clamp(gen_img, 0, 1)
        gen_img = gen_img.permute(1, 2, 0).numpy()
        
        axes[0, i].imshow(gen_img)
        axes[0, i].axis('off')
        # Add thin border
        for spine in axes[0, i].spines.values():
            spine.set_edgecolor('#cccccc')
            spine.set_linewidth(0.5)
            spine.set_visible(True)
        
        # Bottom row: nearest training image
        nn_idx = nearest_indices[i]
        nn_img = training_images[nn_idx].cpu()
        nn_img = (nn_img + 1) / 2  # Denormalize
        nn_img = torch.clamp(nn_img, 0, 1)
        nn_img = nn_img.permute(1, 2, 0).numpy()
        
        axes[1, i].imshow(nn_img)
        axes[1, i].axis('off')
        # Add thin border
        for spine in axes[1, i].spines.values():
            spine.set_edgecolor('#cccccc')
            spine.set_linewidth(0.5)
            spine.set_visible(True)
    
    # Remove all spacing between subplots
    plt.subplots_adjust(wspace=0.02, hspace=0.02, left=0, right=1, top=1, bottom=0)
    
    plt.savefig(save_path, dpi=300, bbox_inches='tight', pad_inches=0.01)
    print(f"Saved visualization to {save_path}")
    plt.close()


def compute_statistics(distances):
    """Compute and print statistics"""
    stats = {
        'mean': float(np.mean(distances)),
        'std': float(np.std(distances)),
        'min': float(np.min(distances)),
        'max': float(np.max(distances)),
        'median': float(np.median(distances)),
        'q25': float(np.percentile(distances, 25)),
        'q75': float(np.percentile(distances, 75))
    }
    
    print("\n" + "="*60)
    print("NEAREST NEIGHBOR DISTANCE STATISTICS")
    print("="*60)
    print(f"Mean Distance:     {stats['mean']:.4f}")
    print(f"Std Deviation:     {stats['std']:.4f}")
    print(f"Median Distance:   {stats['median']:.4f}")
    print(f"Min Distance:      {stats['min']:.4f}")
    print(f"Max Distance:      {stats['max']:.4f}")
    print(f"25th Percentile:   {stats['q25']:.4f}")
    print(f"75th Percentile:   {stats['q75']:.4f}")
    print("="*60)
    
    # Interpretation
    print("\nINTERPRETATION:")
    if stats['min'] < 5.0:
        print("⚠️  Some generated images are very close to training images (min < 5.0)")
        print("    This might indicate memorization for those samples.")
    else:
        print("✓  All generated images have significant distance from training data.")
        print("    Model appears to be generalizing rather than memorizing.")
    
    if stats['mean'] > 15.0:
        print("✓  Mean distance is substantial, indicating good generalization.")
    
    print("="*60)
    
    return stats



if __name__ == "__main__":
    print("="*60)
    print("GAN Nearest Neighbor Sanity Check")
    print("="*60)
    
    # Load CIFAR-10 training data
    training_images, train_dataset = load_cifar10_training_data()
    
    # Load generated images
    generated_images, selected_files = load_generated_images(
        generated_images_dir, num_display_samples
    )
    
    # Compute nearest neighbors
    nearest_indices, nearest_distances = compute_nearest_neighbors(
        generated_images, training_images
    )
    
    # Compute statistics
    stats = compute_statistics(nearest_distances)
    
    # Save statistics to JSON
    stats_file = os.path.join(output_dir, 'nearest_neighbor_stats.json')
    with open(stats_file, 'w') as f:
        json.dump(stats, f, indent=2)
    print(f"\nSaved statistics to {stats_file}")
    
    # Visualize results
    vis_path = os.path.join(output_dir, 'gan_nearest_neighbor_check.png')
    visualize_nearest_neighbors(
        generated_images, training_images, nearest_indices, 
        nearest_distances, vis_path
    )
    print("\n" + "="*60)
    print("Analysis Complete!")
    print("="*60)
    print(f"Results saved to: {output_dir}/")
    print(f"  - gan_nearest_neighbor_check.png")
    print(f"  - distance_distribution.png")
    print(f"  - nearest_neighbor_stats.json")
    print("="*60)

