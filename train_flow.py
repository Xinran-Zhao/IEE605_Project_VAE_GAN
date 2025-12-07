import os
import time
import torch
import argparse
import numpy as np
import matplotlib.pyplot as plt
from torch import optim
from models.RealNVP_flow import RealNVP, count_parameters
from data.cifar_10 import get_cifar10_loaders, StandardNormalization

inverse_norm = StandardNormalization().inverse

def save_reconstruction_progress(model, fixed_images, epoch, device, save_dir="./flow_samples"):
    os.makedirs(save_dir, exist_ok=True)
    model.eval()
    
    with torch.no_grad():
        images = fixed_images.to(device)
        z, _ = model.forward(images)
        reconstructed, _ = model.inverse(z)
    
    images = images.cpu()
    reconstructed = reconstructed.cpu()
    
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
    
    plt.suptitle(f'Flow Reconstruction - Epoch {epoch}', fontsize=14, y=0.98)
    plt.tight_layout()
    save_path = os.path.join(save_dir, f'flow_recon_epoch{epoch}.png')
    plt.savefig(save_path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"Saved reconstruction to {save_path}")

def generate_samples(model, epoch, device, num_samples=25, img_size=32, save_dir="./flow_samples"):
    os.makedirs(save_dir, exist_ok=True)
    model.eval()
    
    with torch.no_grad():
        z = torch.randn(num_samples, 3, img_size, img_size).to(device)
        samples, _ = model.inverse(z)
        samples = samples.cpu()
        
        samples = inverse_norm(samples)
        
        fig, axes = plt.subplots(5, 5, figsize=(10, 10))
        for i, ax in enumerate(axes.flat):
            if i < num_samples:
                img = samples[i].permute(1, 2, 0).numpy()
                img = np.clip(img, 0, 1)
                ax.imshow(img)
            ax.axis('off')
        
        plt.suptitle(f'Flow Generated Samples - Epoch {epoch}', fontsize=14)
        plt.tight_layout()
        save_path = os.path.join(save_dir, f'flow_samples_epoch{epoch}.png')
        plt.savefig(save_path, dpi=200, bbox_inches='tight')
        plt.close()
        print(f"Saved generated samples to {save_path}")

def train_one_epoch(model, dataloader, optimizer, device, log_interval=200):
    model.train()
    running_loss = 0.0
    n = 0
    t0 = time.time()
    for i, (x, _) in enumerate(dataloader):
        x = x.to(device)
        optimizer.zero_grad()
        # compute per-example log probability (tensor shape [B])
        logpx = model.log_prob(x)
        loss = - logpx.mean()  # negative log-likelihood
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        n += 1
        if (i + 1) % log_interval == 0:
            print(f"  batch {i+1}, avg loss {running_loss / n:.4f}")
    t1 = time.time()
    avg_loss = running_loss / max(1, n)
    print(f"Epoch finished in {t1 - t0:.1f}s, avg loss {avg_loss:.4f}")
    return avg_loss

@torch.no_grad()
def evaluate(model, dataloader, device):
    model.eval()
    total_nll = 0.0
    n = 0
    for x, _ in dataloader:
        x = x.to(device)
        logpx = model.log_prob(x)  # per-example log probability
        nll = -logpx  # per-example negative log-likelihood
        total_nll += nll.sum().item()
        n += x.size(0)
    avg_nll = total_nll / n
    # report in nats per image (you can convert to bits/dim if desired)
    return avg_nll

def save_checkpoint(model, optimizer, epoch, val_nll, outdir="./checkpoints"):
    os.makedirs(outdir, exist_ok=True)
    path = os.path.join(outdir, f"realnvp_epoch{epoch}.pt")
    torch.save({
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "val_nll": val_nll
    }, path)
    print(f"Saved checkpoint: {path} (val NLL: {val_nll:.4f})")

def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
    torch.manual_seed(args.seed)
    print("Device:", device)

    # Build model
    model = RealNVP(in_channels=3, num_flows=args.num_flows, hidden_channels=args.hidden_channels,
                    use_checkerboard=not args.channel_split)
    model = model.to(device)
    total_params = count_parameters(model)
    print(f"Model parameters: {total_params:,}")

    # Get dataloaders
    train_loader, val_loader, test_loader = get_cifar10_loaders(
        batch_size=args.batch_size,
        data_path=args.data_path,
        num_workers=4,
        shuffle_train=True,
        apply_dequantization=not args.no_dequantization,
        validation_split=0.1
    )

    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    # Initialize ActNorm (data-dependent) by running one batch forward if available
    try:
        x0, _ = next(iter(train_loader))
        x0 = x0.to(device)
        with torch.no_grad():
            _ = model.forward(x0)
        print("ActNorm initialized with first training batch.")
    except Exception as e:
        print("Warning: Could not initialize ActNorm with train batch:", e)

    # Get fixed test images for reconstruction visualization (same as VAE)
    test_iter = iter(test_loader)
    fixed_test_images, _ = next(test_iter)
    fixed_test_images = fixed_test_images[:10]
    print(f"Fixed {fixed_test_images.size(0)} test images for reconstruction visualization.")

    best_val_nll = float('inf')
    
    for epoch in range(1, args.epochs + 1):
        print(f"\n=== Epoch {epoch}/{args.epochs} ===")
        train_loss = train_one_epoch(model, train_loader, optimizer, device, log_interval=200)
        
        # evaluate on validation set
        if val_loader is not None:
            val_nll = evaluate(model, val_loader, device)
            print(f"Validation NLL (nats/image): {val_nll:.4f}")
            
            # save best model
            if val_nll < best_val_nll:
                best_val_nll = val_nll
                save_checkpoint(model, optimizer, epoch, val_nll, outdir=args.checkpoint_dir)
                print("  -> Best model updated!")
        
        # generate visualization every 5 epochs or at the last epoch
        if epoch % 5 == 0 or epoch == args.epochs:
            print(f"\nGenerating visualizations for epoch {epoch}...")
            save_reconstruction_progress(model, fixed_test_images, epoch, device, save_dir=args.sample_dir)
            generate_samples(model, epoch, device, num_samples=25, save_dir=args.sample_dir)
        
        # save last epoch
        if epoch == args.epochs:
            path = os.path.join(args.checkpoint_dir, "realnvp_last.pt")
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
            }, path)
            print(f"Saved final checkpoint: {path}")

    # final evaluation on test set
    if test_loader is not None:
        test_nll = evaluate(model, test_loader, device)
        print(f"\nTest NLL (nats/image): {test_nll:.4f}")
    print(f"Training complete. Best validation NLL: {best_val_nll:.4f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-path", type=str, default="~/datasets")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--num-flows", type=int, default=14)
    parser.add_argument("--hidden-channels", type=int, default=800)
    parser.add_argument("--channel-split", action="store_true")
    parser.add_argument("--no-dequantization", action="store_true")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--checkpoint-dir", type=str, default="./checkpoints")
    parser.add_argument("--sample-dir", type=str, default="./flow_samples")
    parser.add_argument("--no-cuda", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    main(args)
