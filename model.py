"""Full SCoV model: nucleotide branch, six reading frames and frame fusion.

The parameter names, architecture and forward mathematics match the supplied
Full_SCoV implementation. No experiment registry is needed.
"""
import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

MODEL_VERSION = "SCoV_revision_frame_prior_v1"
PAD_CODON_ID = 64


def _stop_codon_rates(codon_ids, pad_idx=PAD_CODON_ID):
    # A/C/G/T = 0/1/2/3: TAA=48, TAG=50, TGA=56.
    is_stop = (codon_ids == 48) | (codon_ids == 50) | (codon_ids == 56)
    valid = codon_ids != pad_idx
    return (is_stop & valid).float().sum(-1) / valid.float().sum(-1).clamp_min(1.0)


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        self.d_model = d_model
        self.register_buffer("pe", self._build(max_len, d_model).unsqueeze(0), persistent=False)

    @staticmethod
    def _build(max_len, d_model):
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(max_len).unsqueeze(1).float()
        div = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        return pe

    def forward(self, x):
        length = x.size(1)
        pe = self._build(length, self.d_model).unsqueeze(0) if length > self.pe.size(1) else self.pe[:, :length]
        return x + pe.to(device=x.device, dtype=x.dtype)


class MultiScaleBaseBranch(nn.Module):
    def __init__(self, d_model=128):
        super().__init__()
        self.conv_small = nn.Conv1d(4, d_model // 2, kernel_size=5, padding=2)
        self.conv_large = nn.Conv1d(4, d_model // 2, kernel_size=15, padding=7)
        self.bn_stem = nn.BatchNorm1d(d_model)
        self.dilated_blocks = nn.ModuleList([
            nn.Sequential(nn.Conv1d(d_model, d_model, kernel_size=3, padding=d, dilation=d),
                          nn.BatchNorm1d(d_model), nn.GELU())
            for d in (1, 2, 4)
        ])
        self.bigru = nn.GRU(d_model, d_model // 2, num_layers=1, batch_first=True, bidirectional=True)

    def forward(self, x):
        h = F.gelu(self.bn_stem(torch.cat([self.conv_small(x), self.conv_large(x)], dim=1)))
        for block in self.dilated_blocks:
            h = h + block(h)
        h, _ = self.bigru(h.transpose(1, 2))
        return h


class SixFrameCodonBranch(nn.Module):
    def __init__(self, d_model=128, pad_idx=PAD_CODON_ID):
        super().__init__()
        self.pad_idx = pad_idx
        self.codon_emb = nn.Embedding(65, d_model, padding_idx=pad_idx)
        self.shared_bigru = nn.GRU(d_model, d_model // 2, num_layers=1,
                                  batch_first=True, bidirectional=True)
        self.gamma_raw = nn.Parameter(torch.tensor(10.0))

    def _frame_weights(self, codon_ids):
        rates = _stop_codon_rates(codon_ids, self.pad_idx)
        return F.softmax(-F.softplus(self.gamma_raw) * rates, dim=1)

    def forward(self, codon_ids, codon_mask=None, return_frames=False):
        batch, frames, length = codon_ids.shape
        flat = codon_ids.reshape(batch * frames, length)
        h, _ = self.shared_bigru(self.codon_emb(flat))
        valid = (flat != self.pad_idx).float()
        pooled = (h * valid.unsqueeze(-1)).sum(1) / valid.sum(1, keepdim=True).clamp_min(1.0)
        z_frames = pooled.view(batch, frames, self.codon_emb.embedding_dim)
        weights = self._frame_weights(codon_ids)
        z_codon = (z_frames * weights.unsqueeze(-1)).sum(1)
        return (z_codon, z_frames, weights) if return_frames else z_codon


class FrameFusion(nn.Module):
    def __init__(self, d_model=128, num_heads=4, dropout=0.2):
        super().__init__()
        if d_model % num_heads:
            raise ValueError("d_model must be divisible by num_heads")
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.scale = self.head_dim ** -0.5
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        self.attn_drop = nn.Dropout(dropout)
        self.beta = nn.Parameter(torch.tensor(1.0))
        self.gate_fc = nn.Linear(d_model * 2, d_model)
        self.norm = nn.LayerNorm(d_model)

    @staticmethod
    def _masked_mean(x, mask):
        if mask is None:
            return x.mean(1)
        valid = (~mask).float()
        return (x * valid.unsqueeze(-1)).sum(1) / valid.sum(1, keepdim=True).clamp_min(1.0)

    def forward(self, z_base_seq, z_frames, frame_weights, base_mask=None):
        batch, length, d = z_base_seq.shape
        frames = z_frames.size(1)
        z_base = self._masked_mean(z_base_seq, base_mask)

        def heads(t, n):
            return t.view(batch, n, self.num_heads, self.head_dim).transpose(1, 2)

        q, k, v = heads(self.q_proj(z_base_seq), length), heads(self.k_proj(z_frames), frames), heads(self.v_proj(z_frames), frames)
        scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        prior = self.beta * torch.log(frame_weights.clamp_min(1e-6)).clamp_min(-6.0)
        attn = self.attn_drop(torch.softmax(scores + prior[:, None, None, :], dim=-1))
        cross = torch.matmul(attn, v).transpose(1, 2).contiguous().view(batch, length, d)
        z_cross = self._masked_mean(self.out_proj(cross), base_mask)
        alpha = torch.sigmoid(self.gate_fc(torch.cat([z_base, z_cross], dim=-1)))
        return self.norm(alpha * z_cross + (1.0 - alpha) * z_base)


class SCoV(nn.Module):
    def __init__(self, d_model=128, num_heads=4, dropout=0.2):
        super().__init__()
        if d_model <= 0 or d_model % 2 or num_heads <= 0 or d_model % num_heads:
            raise ValueError("d_model must be positive, even and divisible by num_heads")
        self.base_branch = MultiScaleBaseBranch(d_model)
        self.pos_enc = PositionalEncoding(d_model)
        self.codon_branch = SixFrameCodonBranch(d_model)
        self.fusion = FrameFusion(d_model, num_heads, dropout)
        self.classifier = nn.Sequential(
            nn.Linear(d_model, d_model), nn.LayerNorm(d_model), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(d_model, d_model // 2), nn.GELU(), nn.Dropout(dropout / 2),
            nn.Linear(d_model // 2, 2),
        )

    def forward(self, x_base, codon_ids=None, base_mask=None, codon_mask=None):
        if codon_ids is None:
            raise ValueError("codon_ids are required")
        z_base = self.pos_enc(self.base_branch(x_base))
        z_codon, z_frames, weights = self.codon_branch(codon_ids, codon_mask, return_frames=True)
        z_fused = self.fusion(z_base, z_frames, weights, base_mask)
        return self.classifier(z_fused), z_fused, z_codon


class SupervisedCrossModalContrastiveLoss(nn.Module):
    def __init__(self, temperature=0.1):
        super().__init__()
        if temperature <= 0:
            raise ValueError("temperature must be positive")
        self.temperature = temperature

    def forward(self, z_a, z_b, labels):
        z_a = F.normalize(z_a, dim=-1, eps=1e-8)
        z_b = F.normalize(z_b, dim=-1, eps=1e-8)
        logits = torch.matmul(z_a, z_b.T) / self.temperature
        logits = logits - logits.detach().max(1, keepdim=True)[0]
        labels = labels.view(-1, 1)
        positives = torch.eq(labels, labels.T).float().to(z_a.device)
        log_prob = logits - torch.logsumexp(logits, dim=1, keepdim=True)
        return -((positives * log_prob).sum(1) / positives.sum(1).clamp_min(1.0)).mean()


class HexaCrossLoss(nn.Module):
    def __init__(self, cls_weight=1.0, contra_weight=0.1, smoothing=0.1, temperature=0.1):
        super().__init__()
        self.cls_weight, self.contra_weight, self.smoothing = cls_weight, contra_weight, smoothing
        self.supcon = SupervisedCrossModalContrastiveLoss(temperature)

    def forward(self, logits, targets, z_fused, z_codon: Optional[torch.Tensor] = None, use_supcon=True):
        ce = F.cross_entropy(logits, targets, label_smoothing=self.smoothing)
        if use_supcon and z_codon is not None:
            # Same loss formula; float32 avoids low-precision logsumexp/normalization.
            with torch.autocast(device_type=z_fused.device.type, enabled=False):
                contra = self.supcon(z_fused.float(), z_codon.float(), targets)
            if not torch.isfinite(contra):
                raise FloatingPointError("Non-finite contrastive loss")
            return self.cls_weight * ce + self.contra_weight * contra
        return self.cls_weight * ce
