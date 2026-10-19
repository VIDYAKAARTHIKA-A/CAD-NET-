import os
import argparse
import time
import json
import csv
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import f1_score, balanced_accuracy_score
from dataset import prepare_isic2019_loaders, CLASSES
from models import CADNet
from losses import LogitAdjustedLoss, SmoothL1ReconLoss, PrototypicalGuidanceLoss

class EarlyStopping:
    """
    Early stops the training if validation metric (Macro F1) doesn't improve
    after a given patience.
    """
    def __init__(self, patience=8, min_delta=1e-4, mode='max'):
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.counter = 0
        self.best_score = -float('inf') if mode == 'max' else float('inf')
        self.early_stop = False

    def __call__(self, current_score):
        if self.mode == 'max':
            score_improved = current_score > (self.best_score + self.min_delta)
        else:
            score_improved = current_score < (self.best_score - self.min_delta)

        if score_improved:
            self.best_score = current_score
            self.counter = 0
            return True  # New best found
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
            return False  # Did not improve


def train_epoch(model, loader, optimizer, clf_loss_fn, recon_loss_fn, proto_loss_fn, 
                recon_weight, proto_weight, scaler, device):
    model.train()
    total_loss, total_clf, total_recon, total_proto = 0.0, 0.0, 0.0, 0.0
    
    for images, targets in loader:
        images, targets = images.to(device), targets.to(device)
        optimizer.zero_grad()
        
        with torch.cuda.amp.autocast(enabled=(device.type == 'cuda')):
            logits, clean_feats, recon_feats, latent = model(images)
            
            # 1. Logit-Adjusted Classification Loss (enforces class margins)
            l_clf = clf_loss_fn(logits, targets)
            # 2. Embedding Reconstruction DAE Loss
            l_recon = recon_loss_fn(recon_feats, clean_feats.detach())
            # 3. Prototypical Guidance Loss (prevents minority manifold collapse)
            l_proto = proto_loss_fn(latent, targets) if proto_weight > 0 else torch.tensor(0.0)
            
            loss = l_clf + (recon_weight * l_recon) + (proto_weight * l_proto)
            
        if device.type == 'cuda':
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()
            
        bs = images.size(0)
        total_loss += loss.item() * bs
        total_clf += l_clf.item() * bs
        total_recon += l_recon.item() * bs
        total_proto += l_proto.item() * bs
        
    n = len(loader.dataset)
    return total_loss / n, total_clf / n, total_recon / n, total_proto / n


@torch.no_grad()
def evaluate_epoch(model, loader, device):
    model.eval()
    preds, targets_list = [], []
    
    for images, targets in loader:
        images = images.to(device)
        logits, _, _, _ = model(images, noise_level=0.0)
        p = torch.argmax(logits, dim=1).cpu().numpy()
        preds.extend(p)
        targets_list.extend(targets.numpy())
        
    preds = np.array(preds)
    targets_list = np.array(targets_list)
    
    acc = (preds == targets_list).mean()
    bacc = balanced_accuracy_score(targets_list, preds)
    macro_f1 = f1_score(targets_list, preds, average='macro', zero_division=0)
    
    return acc, bacc, macro_f1


def main():
    parser = argparse.ArgumentParser(description="Train Proto-CADNet on ISIC 2019 with Early Stopping")
    parser.add_argument("--data_dir", type=str, default="./isic2019", help="Path to ISIC 2019 data folder")
    parser.add_argument("--backbone", type=str, default="resnet50", choices=["resnet50", "efficientnet_b0"])
    parser.add_argument("--epochs", type=int, default=50, help="Number of training epochs (default: 50)")
    parser.add_argument("--patience", type=int, default=8, help="Early stopping patience in epochs (default: 8)")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--noise_std", type=float, default=0.2, help="DAE Gaussian noise std")
    parser.add_argument("--recon_weight", type=float, default=0.5, help="Weight for reconstruction loss")
    parser.add_argument("--proto_weight", type=float, default=0.2, help="Weight for prototypical guidance loss")
    parser.add_argument("--tau", type=float, default=1.0, help="Logit adjustment temperature")
    parser.add_argument("--output_dir", type=str, default="./checkpoints", help="Save directory")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Executing on: {device}")

    # Prepare Data
    train_loader, val_loader, cls_num_list, train_df, val_df = prepare_isic2019_loaders(
        data_dir=args.data_dir, batch_size=args.batch_size
    )

    print("\n--- Training Set Class Distribution ---")
    for cls_name, count in zip(CLASSES, cls_num_list):
        print(f"  {cls_name:>5}: {count:>6} images ({count / len(train_df) * 100:.2f}%)")

    # Build Model & Losses
    model = CADNet(
        num_classes=len(CLASSES),
        backbone_name=args.backbone,
        latent_dim=256,
        noise_std=args.noise_std
    ).to(device)

    clf_loss_fn = LogitAdjustedLoss(cls_num_list=cls_num_list, tau=args.tau)
    recon_loss_fn = SmoothL1ReconLoss()
    proto_loss_fn = PrototypicalGuidanceLoss(num_classes=len(CLASSES), feat_dim=256).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)
    scaler = torch.cuda.amp.GradScaler(enabled=(device.type == 'cuda'))

    # Early Stopping tracker on Validation Macro F1
    early_stopping = EarlyStopping(patience=args.patience, min_delta=1e-4, mode='max')

    best_macro_f1 = 0.0
    best_ckpt_path = os.path.join(args.output_dir, "best_cadnet.pth")
    last_ckpt_path = os.path.join(args.output_dir, "last_cadnet.pth")
    history_csv = os.path.join(args.output_dir, "training_history.csv")

    # CSV Logging initialization
    with open(history_csv, mode='w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['epoch', 'lr', 'train_loss', 'train_clf', 'train_recon', 'train_proto', 
                         'val_acc', 'val_bacc', 'val_macro_f1', 'time_sec'])

    print(f"\nTraining for up to {args.epochs} epochs with Early Stopping (Patience = {args.patience})...")
    
    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        curr_lr = optimizer.param_groups[0]['lr']
        
        train_loss, train_clf, train_recon, train_proto = train_epoch(
            model, train_loader, optimizer, clf_loss_fn, recon_loss_fn, proto_loss_fn,
            args.recon_weight, args.proto_weight, scaler, device
        )
        scheduler.step()
        
        acc, bacc, macro_f1 = evaluate_epoch(model, val_loader, device)
        elapsed = time.time() - t0
        
        # Log to CSV
        with open(history_csv, mode='a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([epoch, f"{curr_lr:.6f}", f"{train_loss:.4f}", f"{train_clf:.4f}", 
                             f"{train_recon:.4f}", f"{train_proto:.4f}", 
                             f"{acc*100:.2f}", f"{bacc*100:.2f}", f"{macro_f1*100:.2f}", f"{elapsed:.1f}"])

        print(f"Epoch [{epoch:02d}/{args.epochs:02d}] ({elapsed:.1f}s, lr={curr_lr:.2e}) "
              f"Loss: {train_loss:.4f} (Clf: {train_clf:.3f}, Recon: {train_recon:.3f}, Proto: {train_proto:.3f}) | "
              f"Acc: {acc*100:.2f}% | Bal Acc: {bacc*100:.2f}% | Macro F1: {macro_f1*100:.2f}%")

        # Always save last checkpoint for safety/resumption
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'macro_f1': macro_f1,
            'prototypes': proto_loss_fn.prototypes,
            'cls_num_list': cls_num_list,
            'args': vars(args)
        }, last_ckpt_path)

        # Check Early Stopping & Save Best Checkpoint
        is_best = early_stopping(macro_f1)
        if is_best:
            best_macro_f1 = macro_f1
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'macro_f1': best_macro_f1,
                'balanced_acc': bacc,
                'acc': acc,
                'prototypes': proto_loss_fn.prototypes,
                'cls_num_list': cls_num_list,
                'args': vars(args)
            }, best_ckpt_path)
            print(f"  >>> [BEST] Saved new best checkpoint to {best_ckpt_path} (Macro F1: {best_macro_f1*100:.2f}%)")
        else:
            print(f"  --- [Patience] No improvement for {early_stopping.counter}/{args.patience} epochs.")

        if early_stopping.early_stop:
            print(f"\n[EARLY STOPPING TRIGGERED] Validation Macro F1 has not improved in {args.patience} epochs.")
            print(f"Stopping training at Epoch {epoch}. Best Macro F1 achieved: {best_macro_f1*100:.2f}%")
            break

    print(f"\nTraining pipeline finished.")
    print(f"- Best Model Checkpoint: {best_ckpt_path}")
    print(f"- Last Model Checkpoint: {last_ckpt_path}")
    print(f"- Epoch Metrics Log:     {history_csv}")

if __name__ == "__main__":
    main()
