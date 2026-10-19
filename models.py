import torch
import torch.nn as nn
from torchvision import models

class DenoisingBottleneck(nn.Module):
    """
    Denoising Autoencoder Module based on the paper's bottleneck design:
    Compresses high-dimensional feature representations into a compact latent space
    and reconstructs the clean visual embedding from corrupted inputs.
    """
    def __init__(self, in_dim=2048, hidden_dim=512, latent_dim=256):
        super(DenoisingBottleneck, self).__init__()
        
        # DAE Encoder
        self.encoder = nn.Sequential(
            nn.Dropout(p=0.1),
            nn.Linear(in_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, latent_dim),
            nn.BatchNorm1d(latent_dim),
            nn.ReLU(inplace=True)
        )
        
        # DAE Decoder
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, in_dim)
        )

    def forward(self, x):
        latent = self.encoder(x)
        reconstruction = self.decoder(latent)
        return latent, reconstruction


class CADNet(nn.Module):
    """
    Class-Aware Denoising Network (CAD-Net):
    Combines deep feature extraction, controlled embedding perturbation, 
    denoising autoencoder reconstruction, and logit-adjusted classification.
    """
    def __init__(self, num_classes=8, backbone_name='resnet50', latent_dim=256, noise_std=0.2):
        super(CADNet, self).__init__()
        self.noise_std = noise_std
        
        # Pretrained Vision Backbone
        if backbone_name == 'resnet50':
            base_model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
            self.feat_dim = base_model.fc.in_features  # 2048
            self.backbone = nn.Sequential(*list(base_model.children())[:-1])
        elif backbone_name == 'efficientnet_b0':
            base_model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT)
            self.feat_dim = base_model.classifier[1].in_features  # 1280
            self.backbone = nn.Sequential(*list(base_model.children())[:-1])
        else:
            raise ValueError(f"Unsupported backbone: {backbone_name}")
            
        # Embedding Denoising Autoencoder
        self.dae = DenoisingBottleneck(in_dim=self.feat_dim, hidden_dim=512, latent_dim=latent_dim)
        
        # Classifier Head operates on the noise-purified latent embedding
        self.classifier = nn.Sequential(
            nn.Linear(latent_dim, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes)
        )

    def forward(self, x, noise_level=None):
        """
        Args:
            x: Input image tensor [B, C, H, W]
            noise_level: Optional override for noise standard deviation.
                         If None, uses self.noise_std during training and 0 during eval.
        """
        # 1. Feature extraction
        feats = self.backbone(x)
        feats = torch.flatten(feats, 1)  # [B, feat_dim]
        
        # 2. Additive Gaussian noise perturbation
        current_noise = noise_level if noise_level is not None else (self.noise_std if self.training else 0.0)
        if current_noise > 0.0:
            noise = torch.randn_like(feats) * current_noise
            corrupted_feats = feats + noise
        else:
            corrupted_feats = feats
            
        # 3. DAE forward pass
        latent, reconstructed_feats = self.dae(corrupted_feats)
        
        # 4. Classification logits
        logits = self.classifier(latent)
        
        return logits, feats, reconstructed_feats, latent
