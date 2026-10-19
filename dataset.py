import os
import numpy as np
import pandas as pd
from PIL import Image
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torchvision import transforms

# 8 diagnostic categories of ISIC 2019
CLASSES = ['MEL', 'NV', 'BCC', 'AK', 'BKL', 'DF', 'VASC', 'SCC']
CLASS_NAMES = {
    'MEL': 'Melanoma',
    'NV': 'Melanocytic Nevus',
    'BCC': 'Basal Cell Carcinoma',
    'AK': 'Actinic Keratosis',
    'BKL': 'Benign Keratosis',
    'DF': 'Dermatofibroma',
    'VASC': 'Vascular Lesion',
    'SCC': 'Squamous Cell Carcinoma'
}

class ISIC2019Dataset(Dataset):
    def __init__(self, df, img_dir, transform=None):
        self.df = df.reset_index(drop=True)
        self.img_dir = img_dir
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_name = row['image']
        if not img_name.endswith('.jpg'):
            img_name = f"{img_name}.jpg"
        img_path = os.path.join(self.img_dir, img_name)
        
        # Load and convert image
        try:
            image = Image.open(img_path).convert('RGB')
        except Exception:
            # Fallback black image if corrupted
            image = Image.new('RGB', (224, 224), (0, 0, 0))
            
        label = int(row['label'])
        
        if self.transform:
            image = self.transform(image)
            
        return image, label


def get_transforms(img_size=224):
    train_transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.5),
        transforms.RandomRotation(degrees=30),
        transforms.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.15),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])
    
    val_transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])
    return train_transform, val_transform


def prepare_isic2019_loaders(data_dir, batch_size=32, num_workers=2, test_size=0.2, random_state=42):
    """
    Loads ISIC 2019 metadata, stratifies into train/val sets,
    computes class prior frequencies and sets up weighted sampling.
    """
    csv_path = os.path.join(data_dir, "ISIC_2019_Training_GroundTruth.csv")
    img_dir = os.path.join(data_dir, "ISIC_2019_Training_Input")
    
    if not os.path.exists(csv_path):
        raise FileNotFoundError(
            f"Ground truth CSV not found at {csv_path}. Please place "
            f"ISIC_2019_Training_GroundTruth.csv inside {data_dir}."
        )
        
    df = pd.read_csv(csv_path)
    
    # Identify labels from one-hot columns
    available_classes = [c for c in CLASSES if c in df.columns]
    if len(available_classes) != len(CLASSES):
        raise ValueError(f"Expected 8 classes {CLASSES}, found {available_classes} in CSV.")
        
    df['label'] = np.argmax(df[CLASSES].values, axis=1)
    
    # Stratified train/val split
    train_df, val_df = train_test_split(
        df, test_size=test_size, random_state=random_state, stratify=df['label']
    )
    
    # Calculate class counts
    cls_num_list = [int((train_df['label'] == c).sum()) for c in range(len(CLASSES))]
    
    # Weighted Random Sampler to balance gradient updates per batch
    class_weights = 1.0 / (np.array(cls_num_list, dtype=np.float32) + 1e-5)
    sample_weights = [class_weights[lbl] for lbl in train_df['label'].values]
    sampler = WeightedRandomSampler(weights=sample_weights, num_samples=len(train_df), replacement=True)
    
    train_transform, val_transform = get_transforms()
    train_set = ISIC2019Dataset(train_df, img_dir, transform=train_transform)
    val_set = ISIC2019Dataset(val_df, img_dir, transform=val_transform)
    
    train_loader = DataLoader(train_set, batch_size=batch_size, sampler=sampler, 
                              num_workers=num_workers, pin_memory=torch.cuda.is_available())
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False, 
                            num_workers=num_workers, pin_memory=torch.cuda.is_available())
                            
    return train_loader, val_loader, cls_num_list, train_df, val_df
