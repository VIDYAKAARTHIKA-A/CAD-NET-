import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class LogitAdjustedLoss(nn.Module):
    """
    Logit-Adjusted Loss (Menon et al., ICLR 2021).
    Directly maximizes balanced accuracy and macro F1 by enforcing 
    class-dependent margins:
        loss = CrossEntropy(logits + tau * log(pi), y)
    where pi_c is the relative frequency of class c.
    """
    def __init__(self, cls_num_list, tau=1.0):
        super(LogitAdjustedLoss, self).__init__()
        cls_prior = np.array(cls_num_list, dtype=np.float32) / sum(cls_num_list)
        log_prior = np.log(cls_prior + 1e-9)
        self.register_buffer('log_prior', torch.tensor(log_prior, dtype=torch.float32))
        self.tau = tau

    def forward(self, logits, targets):
        adjusted_logits = logits + self.tau * self.log_prior.unsqueeze(0)
        return F.cross_entropy(adjusted_logits, targets)


class SmoothL1ReconLoss(nn.Module):
    """
    Embedding Reconstruction Loss as defined in Section 3.2 (Eq. 1) of the paper:
    Smooth L1 between reconstructed embedding and clean target embedding.
    """
    def __init__(self, beta=1.0):
        super(SmoothL1ReconLoss, self).__init__()
        self.loss_fn = nn.SmoothL1Loss(beta=beta)

    def forward(self, recon_feats, clean_feats):
        return self.loss_fn(recon_feats, clean_feats)


class PrototypicalGuidanceLoss(nn.Module):
    """
    Class-Centroid Prototypical Guidance Loss (Proto-DAE):
    Maintains momentum-updated class centroids in the latent space.
    Penalizes perturbed latent embeddings if they drift away from their
    true class centroid, explicitly preventing minority classes from collapsing
    into dense majority class clusters.
    """
    def __init__(self, num_classes=8, feat_dim=256, momentum=0.9):
        super(PrototypicalGuidanceLoss, self).__init__()
        self.num_classes = num_classes
        self.feat_dim = feat_dim
        self.momentum = momentum
        
        # Buffer to store moving average class prototypes
        self.register_buffer('prototypes', torch.zeros(num_classes, feat_dim))
        self.register_buffer('initialized', torch.zeros(num_classes, dtype=torch.bool))

    def forward(self, latent_feats, targets):
        # Normalize latent vectors for stable metric geometry
        normalized_latents = F.normalize(latent_feats, p=2, dim=1)
        batch_size = latent_feats.size(0)
        
        loss = 0.0
        count = 0
        
        # Calculate loss and update prototypes with moving average
        for c in range(self.num_classes):
            mask = (targets == c)
            if mask.sum() > 0:
                cls_feats = normalized_latents[mask]
                batch_center = cls_feats.mean(dim=0)
                batch_center = F.normalize(batch_center, p=2, dim=0)
                
                # Update running prototype
                if not self.initialized[c]:
                    self.prototypes[c] = batch_center.detach()
                    self.initialized[c] = True
                else:
                    updated = self.momentum * self.prototypes[c] + (1 - self.momentum) * batch_center.detach()
                    self.prototypes[c] = F.normalize(updated, p=2, dim=0)
                
                # Pull perturbed representations toward their own class centroid
                center = self.prototypes[c].unsqueeze(0)
                # Euclidean distance squared: ||z - c_y||^2 = 2 - 2 * <z, c_y> for unit vectors
                cosine_sim = torch.sum(cls_feats * center, dim=1)
                loss += torch.mean(1.0 - cosine_sim)
                count += 1
                
        if count > 0:
            return loss / count
        return torch.tensor(0.0, device=latent_feats.device, requires_grad=True)
