import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix, f1_score, balanced_accuracy_score

import torch
from dataset import prepare_isic2019_loaders, CLASSES, CLASS_NAMES
from models import CADNet

@torch.no_grad()
def evaluate_under_noise(model, loader, noise_level, device):
    model.eval()
    all_preds, all_targets = [], []
    
    for images, targets in loader:
        images = images.to(device)
        logits, _, _, _ = model(images, noise_level=noise_level)
        preds = torch.argmax(logits, dim=1).cpu().numpy()
        all_preds.extend(preds)
        all_targets.extend(targets.numpy())
        
    all_preds = np.array(all_preds)
    all_targets = np.array(all_targets)
    
    acc = (all_preds == all_targets).mean()
    bacc = balanced_accuracy_score(all_targets, all_preds)
    macro_f1 = f1_score(all_targets, all_preds, average='macro', zero_division=0)
    
    return acc, bacc, macro_f1, all_targets, all_preds


def plot_confusion_matrix(y_true, y_pred, save_path):
    cm = confusion_matrix(y_true, y_pred, normalize='true')
    plt.figure(figsize=(9, 7))
    sns.heatmap(cm, annot=True, fmt='.2f', cmap='Blues',
                xticklabels=CLASSES, yticklabels=CLASSES)
    plt.title('Normalized Confusion Matrix (CAD-Net)')
    plt.ylabel('True Class')
    plt.xlabel('Predicted Class')
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"Confusion matrix saved to {save_path}")


def main():
    parser = argparse.ArgumentParser(description="Evaluate CAD-Net Robustness & Macro F1")
    parser.add_argument("--data_dir", type=str, default="./isic2019")
    parser.add_argument("--checkpoint", type=str, default="./checkpoints/best_cadnet.pth")
    parser.add_argument("--backbone", type=str, default="resnet50")
    parser.add_argument("--output_dir", type=str, default="./results")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Evaluating on: {device}")

    # Load data
    _, val_loader, _, _, val_df = prepare_isic2019_loaders(data_dir=args.data_dir, batch_size=32)

    # Initialize model and load weights
    model = CADNet(num_classes=len(CLASSES), backbone_name=args.backbone, latent_dim=256).to(device)
    checkpoint = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    print(f"Loaded checkpoint from {args.checkpoint}")

    # 1. Clean Evaluation & Detailed Classification Report
    acc, bacc, macro_f1, y_true, y_pred = evaluate_under_noise(model, val_loader, noise_level=0.0, device=device)
    print("\n================ CLEAN EVALUATION (Noise std = 0.0) ================")
    print(f"Overall Accuracy:  {acc * 100:.2f}%")
    print(f"Balanced Accuracy: {bacc * 100:.2f}%")
    print(f"Macro F1-Score:    {macro_f1 * 100:.2f}%")
    print("\nDetailed Per-Class Performance:")
    print(classification_report(y_true, y_pred, target_names=CLASSES, digits=4))

    # Save Confusion Matrix
    cm_path = os.path.join(args.output_dir, "confusion_matrix.png")
    plot_confusion_matrix(y_true, y_pred, cm_path)

    # 2. Paper-Style Robustness Benchmark (Noise std in {0.0, 0.1, 0.5, 1.0})
    print("\n================ ROBUSTNESS EVALUATION (Paper Protocol) ================")
    noise_levels = [0.0, 0.1, 0.5, 1.0]
    results = []

    print(f"{'Noise (std)':<12} | {'Accuracy':<10} | {'Balanced Acc':<14} | {'Macro F1':<10}")
    print("-" * 55)
    for nl in noise_levels:
        n_acc, n_bacc, n_f1, _, _ = evaluate_under_noise(model, val_loader, noise_level=nl, device=device)
        print(f"{nl:<12.1f} | {n_acc*100:<9.2f}% | {n_bacc*100:<13.2f}% | {n_f1*100:<9.2f}%")
        results.append((nl, n_acc, n_bacc, n_f1))

    # Plot Robustness Degradation Curve
    plt.figure(figsize=(8, 5))
    stds = [r[0] for r in results]
    f1s = [r[3] * 100 for r in results]
    baccs = [r[2] * 100 for r in results]
    
    plt.plot(stds, f1s, marker='o', label='Macro F1 (%)', linewidth=2)
    plt.plot(stds, baccs, marker='s', label='Balanced Accuracy (%)', linewidth=2)
    plt.title('Robustness under Embedding Perturbation (CAD-Net on ISIC 2019)')
    plt.xlabel('Inference Noise Standard Deviation (sigma)')
    plt.ylabel('Score (%)')
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend()
    plt.tight_layout()
    rob_path = os.path.join(args.output_dir, "robustness_curve.png")
    plt.savefig(rob_path, dpi=300)
    plt.close()
    print(f"\nRobustness curve saved to {rob_path}")

if __name__ == "__main__":
    main()
