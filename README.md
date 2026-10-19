# CAD-Net: Class-Aware Noise-Resilient Representation Learning for ISIC 2019

Implementation of the research gap identified in *"Noise-Aware Visual Representation Learning for Medical Visual Question Answering"* (Pratama et al., 2026), resolving the **Macro F1** collapse under extreme class imbalance.

---

## 1. Directory Structure

```text
cadnet_isic2019/
├── dataset.py            # ISIC 2019 dataset loader, stratified split, weighted sampling
├── models.py             # CAD-Net architecture with DAE bottleneck & classifier
├── losses.py             # Logit-Adjusted CE loss & Smooth L1 reconstruction loss
├── train.py              # Full training loop with amp, cosine LR, best Macro F1 save
├── evaluate.py           # Robustness benchmark (noise in {0, 0.1, 0.5, 1.0}), confusion matrix
├── visualize_tsne.py     # t-SNE projection comparing raw vs. DAE latent space
├── requirements.txt      # Python dependencies
└── isic2019/             # Place downloaded ISIC 2019 data here
    ├── ISIC_2019_Training_GroundTruth.csv
    └── ISIC_2019_Training_Input/
        ├── ISIC_0000000.jpg
        └── ...
```

---

## 2. Setup & Execution Steps

### Step 1: Install Dependencies
```bash
pip install -r requirements.txt
```

### Step 2: Download ISIC 2019 Dataset
Download the following from the official ISIC Archive (or Kaggle `isic-2019`):
1. `ISIC_2019_Training_Input.zip` -> Unzip into `./isic2019/ISIC_2019_Training_Input/`
2. `ISIC_2019_Training_GroundTruth.csv` -> Place in `./isic2019/`

### Step 3: Train the Model
```bash
python train.py --data_dir ./isic2019 --backbone resnet50 --epochs 25 --batch_size 32 --lr 1e-4 --tau 1.0
```

### Step 4: Evaluate Robustness & Generate Confusion Matrix
Runs the paper's benchmark protocol across $\sigma \in \{0.0, 0.1, 0.5, 1.0\}$:
```bash
python evaluate.py --data_dir ./isic2019 --checkpoint ./checkpoints/best_cadnet.pth
```
Outputs:
- Detailed per-class precision, recall, and Macro F1 table.
- Normalized confusion matrix (`results/confusion_matrix.png`).
- Robustness degradation curve (`results/robustness_curve.png`).

### Step 5: Visualize t-SNE Feature Separation
```bash
python visualize_tsne.py --data_dir ./isic2019 --checkpoint ./checkpoints/best_cadnet.pth
```
Generates `results/tsne_comparison.png` demonstrating why rare classes (DF, VASC, SCC) do not collapse into majority classes (NV).
