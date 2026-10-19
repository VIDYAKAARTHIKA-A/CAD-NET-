import os
import argparse
import pandas as pd
import matplotlib.pyplot as plt

def main():
    parser = argparse.ArgumentParser(description="Plot Training & Validation History")
    parser.add_argument("--history_csv", type=str, default="./checkpoints/training_history.csv")
    parser.add_argument("--save_path", type=str, default="./results/training_curves.png")
    args = parser.parse_args()

    if not os.path.exists(args.history_csv):
        print(f"File not found: {args.history_csv}")
        return

    df = pd.read_csv(args.history_csv)
    os.makedirs(os.path.dirname(args.save_path), exist_ok=True)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # 1. Losses
    axes[0].plot(df['epoch'], df['train_loss'], label='Total Loss', color='black', linewidth=2)
    axes[0].plot(df['epoch'], df['train_clf'], label='Classification Loss', linestyle='--', color='blue')
    axes[0].plot(df['epoch'], df['train_recon'], label='DAE Recon Loss', linestyle=':', color='green')
    axes[0].plot(df['epoch'], df['train_proto'], label='Proto Loss', linestyle='-.', color='red')
    axes[0].set_title('Training Losses vs Epoch')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss Value')
    axes[0].grid(True, linestyle='--', alpha=0.5)
    axes[0].legend()

    # 2. Validation Metrics
    axes[1].plot(df['epoch'], df['val_macro_f1'], label='Validation Macro F1 (%)', color='purple', linewidth=2.5, marker='o')
    axes[1].plot(df['epoch'], df['val_bacc'], label='Balanced Acc (%)', color='orange', linewidth=1.8, linestyle='--')
    axes[1].plot(df['epoch'], df['val_acc'], label='Overall Acc (%)', color='gray', linewidth=1.5, linestyle=':')
    axes[1].set_title('Validation Metrics vs Epoch (Macro F1 Monitoring)')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Score (%)')
    axes[1].grid(True, linestyle='--', alpha=0.5)
    axes[1].legend()

    plt.tight_layout()
    plt.savefig(args.save_path, dpi=300)
    plt.close()
    print(f"Training curves plot saved to {args.save_path}")

if __name__ == "__main__":
    main()
