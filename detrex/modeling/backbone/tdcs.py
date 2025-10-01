import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Tuple


class TDCS(nn.Module):
    """
    Transformer Decoder Cross-attention Steering module
    Separates queries into domain-invariant (Q_o) and domain-specific (Q_p) channels
    Computes decoder-level disentanglement losses
    """

    def __init__(self,
                 hidden_dim: int = 256,
                 num_layers: int = 6):
        super().__init__()

        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        # Query separation layers for each decoder layer
        self.query_separators = nn.ModuleList([
            nn.Linear(hidden_dim, hidden_dim * 2) for _ in range(num_layers)
        ])

        # Domain classifiers for discriminant loss
        self.domain_classifiers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim // 2),
                nn.ReLU(),
                nn.Linear(hidden_dim // 2, 2)  # 2 domains
            ) for _ in range(num_layers)
        ])

    def forward(self,
                queries: torch.Tensor,
                layer_idx: int,
                domain_labels: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Args:
            queries: Object queries [batch_size, num_queries, hidden_dim]
            layer_idx: Current decoder layer index (0-based)
            domain_labels: Domain labels for disentanglement loss calculation

        Returns:
            domain_invariant_queries: Q_o for detection head
            disentanglement_losses: Decoder-level disentanglement losses
        """
        batch_size, num_queries, hidden_dim = queries.shape

        # Separate queries into domain-invariant (Q_o) and domain-specific (Q_p)
        separated = self.query_separators[layer_idx](queries)
        Q_o = separated[:, :, :self.hidden_dim]  # domain-invariant
        Q_p = separated[:, :, self.hidden_dim:]  # domain-specific

        # Calculate decoder-level disentanglement losses
        disentanglement_losses = self._compute_decoder_disentanglement_loss(
            Q_o, Q_p, domain_labels, layer_idx
        )

        return Q_o, disentanglement_losses

    def _compute_decoder_disentanglement_loss(self,
                                              Q_o: torch.Tensor,
                                              Q_p: torch.Tensor,
                                              domain_labels: torch.Tensor,
                                              layer_idx: int) -> Dict[str, torch.Tensor]:
        """Compute decoder-level alignment and discriminant losses"""
        batch_size, num_queries, hidden_dim = Q_o.shape

        # Decoder alignment loss (L_De_a): Minimize inconsistencies in domain-invariant channels
        # between different source object queries
        Q_o_normalized = F.normalize(Q_o, p=2, dim=-1)

        # Compute similarity matrix between all queries
        similarity_matrix = torch.matmul(Q_o_normalized,
                                         Q_o_normalized.transpose(1, 2))  # [batch_size, num_queries, num_queries]

        # Create mask for queries from different domains
        domain_expanded = domain_labels.view(batch_size, 1, 1).expand(-1, num_queries, num_queries)
        domain_mask = (domain_expanded != domain_expanded.transpose(1, 2)).float()

        # Maximize similarity for queries from different domains (alignment)
        inter_domain_similarity = (similarity_matrix * domain_mask).sum() / (domain_mask.sum() + 1e-8)
        alignment_loss = -inter_domain_similarity  # Negative because we want to maximize similarity

        # Decoder discriminant loss (L_De_d): Maximize inconsistencies in domain-specific channels
        Q_p_mean = Q_p.mean(dim=1)  # Average over queries [batch_size, hidden_dim]
        domain_pred = self.domain_classifiers[layer_idx](Q_p_mean)
        discriminant_loss = F.cross_entropy(domain_pred, domain_labels)

        return {
            "loss_decoder_alignment": alignment_loss,
            "loss_decoder_discriminant": discriminant_loss
        }