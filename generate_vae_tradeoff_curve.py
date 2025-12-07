import torch
import numpy as np
import matplotlib.pyplot as plt
import json
import os
from torch.optim import Adam
from models.ConvVAE import ConvVAE, loss_function
from data.cifar_10 import get_cifar10_loaders
from evaluation.tradeoff_metrics import compute_rate_distortion, save_sample_images


torch.manual_seed(42)
np.random.seed(42)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")

batch_size = 128
learning_rate = 1e-3
epochs = 20
latent_dim = 256

beta_values = [0.00025, 0.0005, 0.001, 0.01, 0.1, 1.0, 2]

print("Loading CIFAR-10...")
train_loader, val_loader, test_loader = get_cifar10_loaders(
    batch_size=batch_size,
    data_path='~/datasets',
    num_workers=0,
    shuffle_train=True,
    apply_dequantization=True,
    validation_split=0.1
)


def train_with_fixed_beta(beta_value):
    print(f"\n{'='*70}")
    print(f"Training with beta = {beta_value}")
    print(f"{'='*70}")
    
    model = ConvVAE(latent_dim=latent_dim, device=device).to(device)
    optimizer = Adam(model.parameters(), lr=learning_rate, betas=(0.9, 0.999))
    
    os.makedirs(f'tradeoff_models/beta_{beta_value}', exist_ok=True)
    
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        total_recon = 0
        total_kld = 0
        n_batches = 0
        
        for x, _ in train_loader:
            x = x.to(device)
            optimizer.zero_grad()
            
            x_recon, mu, logvar = model(x)
            loss, recon_loss, kld_loss = loss_function(x, x_recon, mu, logvar, beta_value)
            
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            total_recon += recon_loss.item()
            total_kld += kld_loss.item()
            n_batches += 1
        
        avg_loss = total_loss / n_batches
        avg_recon = total_recon / n_batches
        avg_kld = total_kld / n_batches
        print(f"Epoch {epoch+1}/{epochs} | Loss: {avg_loss:.4f} | Recon: {avg_recon:.4f} | KLD: {avg_kld:.4f}")
        
        if (epoch + 1) % 4 == 0:
            save_sample_images(model, device, beta_value, epoch+1, f'tradeoff_samples')
    
    model_path = f'tradeoff_models/beta_{beta_value}/model.pth'
    torch.save({
        'model_state_dict': model.state_dict(),
        'beta': beta_value,
        'latent_dim': latent_dim,
    }, model_path)
    print(f"Saved model to {model_path}")
    
    return model


def evaluate_all_models():
    results = []
    
    for beta in beta_values:
        print(f"\nEvaluating beta = {beta}...")
        
        model = ConvVAE(latent_dim=latent_dim, device=device).to(device)
        checkpoint = torch.load(f'tradeoff_models/beta_{beta}/model.pth')
        model.load_state_dict(checkpoint['model_state_dict'])
        
        rate, distortion = compute_rate_distortion(model, test_loader, device)
        
        results.append({
            'beta': beta,
            'rate': rate,
            'distortion': distortion
        })
        
        print(f"Beta={beta}: Rate={rate:.4f} bits, Distortion={distortion:.6f}")
    
    return results


def plot_tradeoff_curve(results):
    rates = [r['rate'] for r in results]
    distortions = [r['distortion'] for r in results]
    betas = [r['beta'] for r in results]
    
    plt.figure(figsize=(10, 7))
    plt.plot(rates, distortions, 'o-', linewidth=2.5, markersize=10, color='#2E86AB')
    
    for i, beta in enumerate(betas):
        plt.annotate(f'β={beta}', 
                    xy=(rates[i], distortions[i]),
                    xytext=(10, -10), 
                    textcoords='offset points',
                    fontsize=11)
    
    plt.xlabel('Rate (KL Divergence in bits)', fontsize=14, fontweight='bold')
    plt.ylabel('Distortion (MSE)', fontsize=14, fontweight='bold')
    plt.title('Rate-Distortion Trade-off Curve for VAE', fontsize=16, fontweight='bold')
    plt.grid(True, alpha=0.3, linestyle='--')
    plt.tight_layout()
    plt.savefig('tradeoff_curve.png', dpi=300, bbox_inches='tight')
    print("\nSaved curve to tradeoff_curve.png")
    plt.close()


if __name__ == "__main__":
    # ------------------------------------------------------------
    # Case1: train the models from scratch
    os.makedirs('tradeoff_models', exist_ok=True)
    os.makedirs('tradeoff_samples', exist_ok=True)
    
    for beta in beta_values:
        train_with_fixed_beta(beta)
    
    results = evaluate_all_models()
    
    with open('tradeoff_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    print("\nSaved results to tradeoff_results.json")
    # ------------------------------------------------------------
    # # Case2: already have the results, need to replot the tradeoff curve
    # with open('tradeoff_results.json', 'r') as f:
    #     results = json.load(f)
    # ------------------------------------------------------------
    plot_tradeoff_curve(results)
    print("\n" + "="*70)
    print("Rate-Distortion Analysis Complete")
    print("="*70)
    for r in results:
        print(f"β={r['beta']:>4.1f} | Rate={r['rate']:>8.4f} bits | Distortion={r['distortion']:>10.6f}")
    print("="*70)
