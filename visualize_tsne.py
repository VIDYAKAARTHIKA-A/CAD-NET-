import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import torch
from dataset import prepare_isic2019_loaders, CLASSES
from models import CADNet

@torch.no_grad()
def extract_embeddings(model, loader, device, max_samples=1500):
    model.eval()
    raw_feats_list, latent_feats_list, labels_list = [], [], []
    collected = 0
    
    for images, targets in loader:
        images = images.to(device)
        # Pass with moderate noise to see the DAE effect
        _, clean_feats, _, latent = model(images, noise_level=0.5)
        
        raw_feats_list.append(clean_feats.cpu().numpy())
        latent_feats_list.append(latent.cpu().numpy())
        labels_list.append(targets.numpy())
        
        collected += images.size(0)
        if collected >= max_samples:
            break
            
    raw = np.concatenate(raw_feats_list, axis=0)[:max_samples]
    latent = np.concatenate(latent_feats_list, axis=0)[:max_samples]
    labels = np.concatenate(labels_list, axis=0)[:max_samples]
    
    return raw, latent, labels


def main():
    parser = argparse.ArgumentParser(description="Generate t-SNE of representations")
    parser.add_argument("--data_dir", type=str, default="./isic2019")
    parser.add_argument("--checkpoint", type=str, default="./checkpoints/best_cadnet.pth")
    parser.add_argument("--output_path", type=str, default="./results/tsne_comparison.png")
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.output_path), exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    _, val_loader, _, _, _ = prepare_isic2019_loaders(data_dir=args.data_dir, batch_size=32)
    model = CADNet(num_classes=len(CLASSES), backbone_name="resnet50", latent_dim=256).to(device)
    checkpoint = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])

    print("Extracting features for t-SNE projection...")
    raw, latent, labels = extract_embeddings(model, val_loader, device)

    print("Running t-SNE dimensionality reduction...")
    tsne = TSNE(n_components=2, perplexity=30, random_state=42)
    raw_2d = tsne.fit_transform(raw)
    latent_2d = tsne.fit_transform(latent)

    # Plot Comparison
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    cmap = plt.get_cmap('tab10')

    for c_idx, c_name in enumerate(CLASSES):
        mask = (labels == c_idx)
        if np.sum(mask) == 0:
            continue
        axes[0].scatter(raw_2d[mask, 0], raw_2d[mask, 1], label=c_name, color=cmap(c_idx), alpha=0.7, s=25)
        axes[1].scatter(latent_2d[mask, 0], latent_2d[mask, 1], label=c_name, color=cmap(c_idx), alpha=0.7, s=25)

    axes[0].set_title('Raw Backbone Features (Susceptible to Feature Collapse)')
    axes[1].set_title('CAD-Net Purified Latent Space (Preserved Minority Clusters)')
    for ax in axes:
        ax.set_xticks([])
        ax.set_yticks([])
        ax.legend(loc='upper right', markerscale=1.5)

    plt.tight_layout()
    plt.savefig(args.output_path, dpi=300)
    plt.close()
    print(f"t-SNE comparison saved to {args.output_path}")

if __name__ == "__main__":
    main()
