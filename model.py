from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_length: int = 5000):
        super().__init__()
        self.d_model = d_model
        self.register_buffer(
            "encoding",
            self._build(max_length, d_model).unsqueeze(0),
        )

    @staticmethod
    def _build(length: int, d_model: int):
        encoding = torch.zeros(length, d_model)
        positions = torch.arange(length).unsqueeze(1).float()
        divisor = torch.exp(
            torch.arange(0, d_model, 2).float()
            * (-math.log(10000.0) / d_model)
        )
        encoding[:, 0::2] = torch.sin(positions * divisor)
        encoding[:, 1::2] = torch.cos(positions * divisor)
        return encoding

    def forward(self, x):
        length = x.size(1)
        if length > self.encoding.size(1):
            encoding = self._build(length, self.d_model).to(x.device, x.dtype)
            return x + encoding.unsqueeze(0)
        return x + self.encoding[:, :length].to(dtype=x.dtype)


class MultiScaleBaseBranch(nn.Module):
    def __init__(self, d_model: int = 128):
        super().__init__()

        self.conv_small = nn.Conv1d(4, d_model // 2, kernel_size=5, padding=2)
        self.conv_large = nn.Conv1d(4, d_model // 2, kernel_size=15, padding=7)
        self.batch_norm = nn.BatchNorm1d(d_model)

        self.dilated_blocks = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv1d(
                        d_model,
                        d_model,
                        kernel_size=3,
                        padding=dilation,
                        dilation=dilation,
                    ),
                    nn.BatchNorm1d(d_model),
                    nn.GELU(),
                )
                for dilation in (1, 2, 4)
            ]
        )

        self.bigru = nn.GRU(
            d_model,
            d_model // 2,
            batch_first=True,
            bidirectional=True,
        )

    def forward(self, x):
        hidden = torch.cat([self.conv_small(x), self.conv_large(x)], dim=1)
        hidden = F.gelu(self.batch_norm(hidden))

        for block in self.dilated_blocks:
            hidden = hidden + block(hidden)

        hidden = hidden.transpose(1, 2)
        hidden, _ = self.bigru(hidden)
        return hidden


def _stop_codon_rates(codon_ids, pad_index=64):
    is_stop = codon_ids.eq(48) | codon_ids.eq(50) | codon_ids.eq(56)
    is_valid = codon_ids.ne(pad_index)
    return (
        (is_stop & is_valid).float().sum(dim=-1)
        / is_valid.float().sum(dim=-1).clamp_min(1.0)
    )


class SixFrameCodonBranch(nn.Module):
    def __init__(self, d_model: int = 128, pad_index: int = 64):
        super().__init__()

        self.pad_index = pad_index
        self.embedding = nn.Embedding(65, d_model, padding_idx=pad_index)
        self.stop_scale = nn.Parameter(torch.tensor(10.0))
        self.bigru = nn.GRU(
            d_model,
            d_model // 2,
            batch_first=True,
            bidirectional=True,
        )

    def forward(self, codon_ids):
        batch_size, frame_count, codon_length = codon_ids.shape
        d_model = self.embedding.embedding_dim

        stop_rates = _stop_codon_rates(codon_ids, self.pad_index)
        frame_gates = F.softmax(
            -F.softplus(self.stop_scale) * stop_rates,
            dim=1,
        )

        flattened = codon_ids.reshape(batch_size * frame_count, codon_length)
        embedded = self.embedding(flattened)
        encoded, _ = self.bigru(embedded)

        valid = flattened.ne(self.pad_index).float()
        valid_count = valid.sum(dim=1, keepdim=True).clamp_min(1.0)
        frame_vectors = (
            encoded * valid.unsqueeze(-1)
        ).sum(dim=1) / valid_count

        frame_vectors = frame_vectors.view(batch_size, frame_count, d_model)
        codon_vector = (
            frame_vectors * frame_gates.unsqueeze(-1)
        ).sum(dim=1)

        return codon_vector, frame_vectors, frame_gates


class FrameGatedCIA(nn.Module):
    def __init__(
        self,
        d_model: int = 128,
        num_heads: int = 4,
        dropout: float = 0.2,
    ):
        super().__init__()

        if d_model % num_heads != 0:
            raise ValueError("d_model must be divisible by num_heads.")

        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.scale = self.head_dim ** -0.5

        self.query = nn.Linear(d_model, d_model)
        self.key = nn.Linear(d_model, d_model)
        self.value = nn.Linear(d_model, d_model)
        self.output = nn.Linear(d_model, d_model)
        self.attention_dropout = nn.Dropout(dropout)

        self.frame_prior_strength = nn.Parameter(torch.tensor(1.0))
        self.fusion_gate = nn.Linear(d_model * 2, d_model)
        self.norm = nn.LayerNorm(d_model)

    def _split_heads(self, tensor, length):
        batch_size = tensor.size(0)
        return (
            tensor.view(
                batch_size,
                length,
                self.num_heads,
                self.head_dim,
            )
            .transpose(1, 2)
        )

    @staticmethod
    def _masked_mean(tensor, mask):
        if mask is None:
            return tensor.mean(dim=1)

        valid = (~mask).float()
        return (
            tensor * valid.unsqueeze(-1)
        ).sum(dim=1) / valid.sum(dim=1, keepdim=True).clamp_min(1.0)

    def forward(
        self,
        nucleotide_sequence,
        frame_vectors,
        frame_gates,
        nucleotide_mask=None,
    ):
        batch_size, sequence_length, d_model = nucleotide_sequence.shape

        query = self._split_heads(self.query(nucleotide_sequence), sequence_length)
        key = self._split_heads(self.key(frame_vectors), 6)
        value = self._split_heads(self.value(frame_vectors), 6)

        scores = torch.matmul(query, key.transpose(-2, -1)) * self.scale
        gate_log = torch.log(frame_gates.clamp_min(1e-6)).clamp_min(-6.0)
        scores = scores + (
            self.frame_prior_strength * gate_log
        )[:, None, None, :]

        attention = self.attention_dropout(torch.softmax(scores, dim=-1))
        cross_modal = torch.matmul(attention, value)
        cross_modal = (
            cross_modal.transpose(1, 2)
            .contiguous()
            .view(batch_size, sequence_length, d_model)
        )
        cross_modal = self.output(cross_modal)

        cross_global = self._masked_mean(cross_modal, nucleotide_mask)
        base_global = self._masked_mean(nucleotide_sequence, nucleotide_mask)

        alpha = torch.sigmoid(
            self.fusion_gate(torch.cat([base_global, cross_global], dim=-1))
        )
        fused = alpha * cross_global + (1.0 - alpha) * base_global
        return self.norm(fused)


class SCoV(nn.Module):
    def __init__(
        self,
        d_model: int = 128,
        num_heads: int = 4,
        dropout: float = 0.2,
    ):
        super().__init__()

        self.base_branch = MultiScaleBaseBranch(d_model=d_model)
        self.positional_encoding = PositionalEncoding(d_model)
        self.codon_branch = SixFrameCodonBranch(d_model=d_model)
        self.cia = FrameGatedCIA(
            d_model=d_model,
            num_heads=num_heads,
            dropout=dropout,
        )

        self.classifier = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Dropout(dropout / 2),
            nn.Linear(d_model // 2, 2),
        )

    def forward(
        self,
        nucleotide_one_hot,
        codon_ids,
        nucleotide_mask=None,
        codon_mask=None,
    ):
        nucleotide_sequence = self.base_branch(nucleotide_one_hot)
        nucleotide_sequence = self.positional_encoding(nucleotide_sequence)

        codon_vector, frame_vectors, frame_gates = self.codon_branch(codon_ids)

        fused = self.cia(
            nucleotide_sequence,
            frame_vectors,
            frame_gates,
            nucleotide_mask,
        )

        logits = self.classifier(fused)
        return logits, fused, codon_vector
