import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Dict, Optional, Tuple


class TEPG(nn.Module):
    """
    Transformer Encoder Positional Guidance module
    Positioned between backbone and transformer encoder, operates in parallel with feature embedding
    Responsible for generating domain prompts and computing encoder-level disentanglement losses
    """

    def __init__(self,
                 feature_dim: int = 256,
                 domain_prompt_dim: int = 256,
                 num_domains: int = 2):
        super().__init__()

        self.feature_dim = feature_dim
        self.domain_prompt_dim = domain_prompt_dim
        self.num_domains = num_domains

        # Domain prompt generation network
        self.domain_prompt_net = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(feature_dim, domain_prompt_dim * 4),
            nn.ReLU(),
            nn.Linear(domain_prompt_dim * 4, domain_prompt_dim)
        )

        # Domain classifier for disentanglement loss
        self.domain_classifier = nn.Sequential(
            nn.Linear(feature_dim, feature_dim // 2),
            nn.ReLU(),
            nn.Linear(feature_dim // 2, num_domains)
        )

    def forward(self,
                features: List[torch.Tensor],
                domain_labels: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Args:
            features: Multi-scale features from backbone [C3, C4, C5]
            domain_labels: Domain labels (0 or 1) with shape [batch_size]

        Returns:
            domain_prompts: Generated domain prompts
            disentanglement_losses: Encoder-level disentanglement losses
        """
        batch_size = features[0].shape[0]

        # Generate domain prompts from features (using the last feature level C5)
        domain_prompts = []
        for i in range(batch_size):
            feature = features[-1][i:i + 1]  # Take C5 feature for prompt generation
            domain_prompt = self.domain_prompt_net(feature)
            domain_prompts.append(domain_prompt)

        domain_prompts = torch.cat(domain_prompts, dim=0)  # [batch_size, domain_prompt_dim]

        # Calculate encoder-level disentanglement losses
        disentanglement_losses = self._compute_encoder_disentanglement_loss(
            features, domain_labels, domain_prompts
        )

        return domain_prompts, disentanglement_losses

    def _compute_encoder_disentanglement_loss(self,
                                              features: List[torch.Tensor],
                                              domain_labels: torch.Tensor,
                                              domain_prompts: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Compute encoder-level alignment and discriminant losses"""
        batch_size = features[0].shape[0]

        # Use C5 features for disentanglement computation
        feature_embeddings = F.adaptive_avg_pool2d(features[-1], 1).flatten(1)  # [batch_size, feature_dim]

        # Normalize features and prompts
        feats_norm = F.normalize(feature_embeddings, p=2, dim=1)
        prompts_norm = F.normalize(domain_prompts, p=2, dim=1)

        # Compute similarity matrix between feature embeddings and domain prompts
        similarity_matrix = torch.matmul(feats_norm, prompts_norm.t())  # [batch_size, batch_size]

        # Create domain mask: 1 for same domain, 0 for different domains
        domain_mask = (domain_labels.unsqueeze(1) == domain_labels.unsqueeze(0)).float()

        # Encoder alignment loss (L_En_a): Maximize similarity for same domain
        same_domain_sim = (similarity_matrix * domain_mask).sum() / (domain_mask.sum() + 1e-8)
        alignment_loss = -same_domain_sim  # Negative because we want to maximize

        # Encoder discriminant loss (L_En_d): Minimize similarity for different domains
        diff_domain_mask = 1 - domain_mask
        diff_domain_sim = (similarity_matrix * diff_domain_mask).sum() / (diff_domain_mask.sum() + 1e-8)
        discriminant_loss = diff_domain_sim  # Positive because we want to minimize

        return {
            "loss_encoder_alignment": alignment_loss,
            "loss_encoder_discriminant": discriminant_loss
        }