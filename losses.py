from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class SupervisedContrastiveLoss(nn.Module):
    def __init__(self, temperature: float = 0.1):
        super().__init__()
        self.temperature = temperature

    def forward(self, fused_features, codon_features, labels):
        fused = F.normalize(fused_features, dim=-1, eps=1e-8)
        codon = F.normalize(codon_features, dim=-1, eps=1e-8)

        logits = torch.matmul(fused, codon.T) / self.temperature
        logits = logits - logits.detach().max(dim=1, keepdim=True).values

        labels = labels.view(-1, 1)
        positive_mask = labels.eq(labels.T).float().to(logits.device)

        log_probabilities = logits - torch.log(
            torch.exp(logits).sum(dim=1, keepdim=True) + 1e-10
        )

        positive_count = positive_mask.sum(dim=1).clamp_min(1e-6)
        return -(
            (positive_mask * log_probabilities).sum(dim=1)
            / positive_count
        ).mean()


class SCoVLoss(nn.Module):
    def __init__(
        self,
        label_smoothing: float = 0.1,
        contrastive_weight: float = 0.1,
        temperature: float = 0.1,
    ):
        super().__init__()
        self.label_smoothing = label_smoothing
        self.contrastive_weight = contrastive_weight
        self.contrastive = SupervisedContrastiveLoss(temperature)

    def forward(self, logits, targets, fused_features, codon_features):
        classification_loss = F.cross_entropy(
            logits,
            targets,
            label_smoothing=self.label_smoothing,
        )
        contrastive_loss = self.contrastive(
            fused_features,
            codon_features,
            targets,
        )
        return classification_loss + self.contrastive_weight * contrastive_loss
