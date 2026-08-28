import math
from typing import Optional
import torch
import torch.nn as nn
import torch.nn.functional as F


def _stop_codon_rates(codon_ids: torch.Tensor, pad_idx: int = 64) -> torch.Tensor:
    is_stop = (codon_ids == 48) | (codon_ids == 50) | (codon_ids == 56)
    is_valid = (codon_ids != pad_idx)
    return (is_stop & is_valid).float().sum(-1) / is_valid.float().sum(-1).clamp(1.0)


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()
        self.d_model = d_model
        self.register_buffer("pe", self._build(max_len, d_model).unsqueeze(0))

    @staticmethod
    def _build(max_len, d_model):
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(max_len).unsqueeze(1).float()
        div = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        return pe

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        L = x.size(1)
        if L > self.pe.size(1):
            pe = self._build(L, self.d_model).to(x.device, x.dtype)
            return x + pe.unsqueeze(0)
        return x + self.pe[:, :L].to(dtype=x.dtype)


class MultiScaleBaseBranch(nn.Module):
    def __init__(self, d_model: int = 128, use_bigru: bool = True):
        super().__init__()
        self.use_bigru = use_bigru
        self.conv_small = nn.Conv1d(4, d_model // 2, kernel_size=5, padding=2)
        self.conv_large = nn.Conv1d(4, d_model // 2, kernel_size=15, padding=7)
        self.bn_stem = nn.BatchNorm1d(d_model)
        self.dilated_blocks = nn.ModuleList([
            nn.Sequential(
                nn.Conv1d(d_model, d_model, kernel_size=3, padding=d, dilation=d),
                nn.BatchNorm1d(d_model),
                nn.GELU(),
            ) for d in [1, 2, 4]
        ])
        if use_bigru:
            self.bigru = nn.GRU(
                d_model, d_model // 2, num_layers=1,
                batch_first=True, bidirectional=True
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = F.gelu(self.bn_stem(
            torch.cat([self.conv_small(x), self.conv_large(x)], dim=1)
        ))
        for blk in self.dilated_blocks:
            h = h + blk(h)
        h = h.transpose(1, 2)
        if self.use_bigru:
            h, _ = self.bigru(h)
        return h


class SixFrameCodonBranch(nn.Module):
    def __init__(
        self,
        d_model: int = 128,
        pad_idx: int = 64,
        cub_scores: Optional[torch.Tensor] = None,
        use_stop_gate: bool = True,
        use_cub_pool: bool = True,
    ):
        super().__init__()
        self.pad_idx = pad_idx
        self.use_stop_gate = use_stop_gate
        self.use_cub_pool = use_cub_pool
        self.codon_emb = nn.Embedding(65, d_model, padding_idx=pad_idx)
        _cub = cub_scores if cub_scores is not None else torch.zeros(64)
        assert _cub.shape == (64,)
        self.register_buffer("cub_scores", _cub.float())
        self.gamma = nn.Parameter(torch.tensor(1.0))
        self.lam = nn.Parameter(torch.tensor(10.0))
        self.shared_bigru = nn.GRU(
            d_model, d_model // 2, num_layers=1,
            batch_first=True, bidirectional=True
        )

    def load_cub(self, cub_scores: torch.Tensor):
        assert cub_scores.shape == (64,)
        self.cub_scores.copy_(cub_scores.float())

    def _cub_weights(self, flat_ids: torch.Tensor) -> torch.Tensor:
        safe_ids = flat_ids.clamp(0, 63)
        scores = self.cub_scores[safe_ids].clamp(-6.0, 6.0)
        is_valid = (flat_ids != self.pad_idx).float()
        w = torch.exp(scores) * is_valid
        return w / w.sum(1, keepdim=True).clamp(1e-6)

    def forward(
        self,
        codon_ids_6frame: torch.Tensor,
        codon_mask: Optional[torch.Tensor] = None,
        return_frames: bool = False,
    ):
        B, F_num, L_c = codon_ids_6frame.shape
        d = self.codon_emb.embedding_dim

        if self.use_stop_gate:
            stop_rates = _stop_codon_rates(codon_ids_6frame, self.pad_idx)
            lam_pos = F.softplus(self.lam)
            frame_gates = F.softmax(-lam_pos * stop_rates, dim=1)
        else:
            frame_gates = torch.full(
                (B, F_num), 1.0 / F_num,
                device=codon_ids_6frame.device
            )

        flat_ids = codon_ids_6frame.reshape(B * F_num, L_c)
        emb = self.codon_emb(flat_ids)
        is_valid = (flat_ids != self.pad_idx).float()
        cub_bias = self.cub_scores[flat_ids.clamp(0, 63)]
        emb = emb + self.gamma * (cub_bias * is_valid).unsqueeze(-1)

        h, _ = self.shared_bigru(emb)

        if self.use_cub_pool:
            w_norm = self._cub_weights(flat_ids)
            z_frames_flat = (h * w_norm.unsqueeze(-1)).sum(1)
        else:
            valid_cnt = is_valid.sum(-1, keepdim=True).clamp(1.0)
            z_frames_flat = (h * is_valid.unsqueeze(-1)).sum(1) / valid_cnt

        z_frames = z_frames_flat.view(B, F_num, d)
        Z_codon = (z_frames * frame_gates.unsqueeze(-1)).sum(1)

        if return_frames:
            return Z_codon, z_frames, frame_gates
        return Z_codon


class FrameGatedCIA(nn.Module):
    def __init__(
        self,
        d_model: int = 128,
        num_heads: int = 4,
        dropout: float = 0.2,
        use_frame_prior: bool = True,
    ):
        super().__init__()
        assert d_model % num_heads == 0
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.scale = self.head_dim ** -0.5
        self.use_frame_prior = use_frame_prior

        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        self.attn_drop = nn.Dropout(dropout)

        if use_frame_prior:
            self.gate_strength = nn.Parameter(torch.tensor(1.0))

        self.gate_fc = nn.Linear(d_model * 2, d_model)
        self.norm = nn.LayerNorm(d_model)

    def _masked_mean(self, x, mask=None):
        if mask is None:
            return x.mean(1)
        valid = (~mask).float()
        return (x * valid.unsqueeze(-1)).sum(1) / valid.sum(1, keepdim=True).clamp(1)

    def forward(
        self,
        z_base_seq: torch.Tensor,
        z_frames: torch.Tensor,
        frame_gates: torch.Tensor,
        base_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        B, L, d = z_base_seq.shape
        F_kv = z_frames.size(1)

        def split_heads(t, seq_len):
            return t.view(B, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        q = split_heads(self.q_proj(z_base_seq), L)
        k = split_heads(self.k_proj(z_frames), F_kv)
        v = split_heads(self.v_proj(z_frames), F_kv)

        scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale

        if self.use_frame_prior:
            gate_log = torch.log(frame_gates.clamp(min=1e-6)).clamp(min=-6.0)
            gate_bias = self.gate_strength * gate_log
            scores = scores + gate_bias[:, None, None, :]

        attn = self.attn_drop(torch.softmax(scores, dim=-1))
        z_cross = torch.matmul(attn, v)
        z_cross = z_cross.transpose(1, 2).contiguous().view(B, L, d)
        z_cross = self.out_proj(z_cross)

        z_cia_g = self._masked_mean(z_cross, base_mask)
        z_base_g = self._masked_mean(z_base_seq, base_mask)

        alpha = torch.sigmoid(self.gate_fc(
            torch.cat([z_base_g, z_cia_g], dim=-1)
        ))
        z_fused = alpha * z_cia_g + (1.0 - alpha) * z_base_g
        return self.norm(z_fused)


CIA = FrameGatedCIA


class SCoV(nn.Module):
    def __init__(
        self,
        d_model: int = 128,
        dropout: float = 0.2,
        cub_scores: Optional[torch.Tensor] = None,
    ):
        super().__init__()
        self.base_branch = MultiScaleBaseBranch(d_model, use_bigru=True)
        self.codon_branch = SixFrameCodonBranch(
            d_model, cub_scores=cub_scores,
            use_stop_gate=True, use_cub_pool=True,
        )
        self.pos_enc = PositionalEncoding(d_model)
        self.cia = FrameGatedCIA(
            d_model, num_heads=4, dropout=dropout,
            use_frame_prior=True
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
        x_base: torch.Tensor,
        codon_ids_6frame: torch.Tensor,
        base_mask: Optional[torch.Tensor] = None,
        codon_mask: Optional[torch.Tensor] = None,
    ):
        z_base_seq = self.pos_enc(self.base_branch(x_base))
        z_codon, z_frames, frame_gates = self.codon_branch(
            codon_ids_6frame, codon_mask, return_frames=True
        )
        z_fused = self.cia(z_base_seq, z_frames, frame_gates, base_mask)
        logits = self.classifier(z_fused)
        return logits, z_fused, z_codon


HexaCodonBranchV2 = SixFrameCodonBranch


class AblationSCoV(nn.Module):
    def __init__(
        self,
        d_model: int = 128,
        dropout: float = 0.2,
        cub_scores=None,
        use_codon: bool = True,
        use_cia: bool = True,
        use_stop_gate: bool = True,
        use_cub_pool: bool = True,
        use_bigru_base: bool = True,
        use_frame_prior: bool = True,
        codon_frame_mode: str = "six",
    ):
        super().__init__()
        if codon_frame_mode not in {"six", "plus0"}:
            raise ValueError(
                f"codon_frame_mode must be 'six' or 'plus0', got: {codon_frame_mode}"
            )
        self.use_codon = use_codon
        self.use_cia = use_cia
        self.codon_frame_mode = codon_frame_mode

        self.base_branch = MultiScaleBaseBranch(d_model, use_bigru=use_bigru_base)
        self.pos_enc = PositionalEncoding(d_model)

        if use_codon:
            self.codon_branch = SixFrameCodonBranch(
                d_model, cub_scores=cub_scores,
                use_stop_gate=use_stop_gate, use_cub_pool=use_cub_pool,
            )

        if use_codon and use_cia:
            self.cia = FrameGatedCIA(
                d_model, num_heads=4, dropout=dropout,
                use_frame_prior=use_frame_prior,
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

    def _masked_pool(self, h, mask=None):
        if mask is None:
            return h.mean(1)
        valid = (~mask).float()
        return (h * valid.unsqueeze(-1)).sum(1) / valid.sum(1, keepdim=True).clamp(1)

    def _select_codon_frames(self, codon_ids, codon_mask=None):
        if self.codon_frame_mode == "plus0":
            codon_ids_in = codon_ids[:, 0:1, :]
            if codon_mask is not None and codon_mask.dim() == 3:
                codon_mask_in = codon_mask[:, 0:1, :]
            else:
                codon_mask_in = codon_mask
            return codon_ids_in, codon_mask_in
        return codon_ids, codon_mask

    def forward(self, x_base, codon_ids, base_mask=None, codon_mask=None):
        z_base_seq = self.pos_enc(self.base_branch(x_base))
        z_codon = None

        if self.use_codon:
            codon_ids_in, codon_mask_in = self._select_codon_frames(codon_ids, codon_mask)
            if self.use_cia:
                z_codon, z_frames, frame_gates = self.codon_branch(
                    codon_ids_in, codon_mask_in, return_frames=True,
                )
                z_fused = self.cia(z_base_seq, z_frames, frame_gates, base_mask)
            else:
                z_codon = self.codon_branch(codon_ids_in, codon_mask_in, return_frames=False)
                z_base_g = self._masked_pool(z_base_seq, base_mask)
                z_fused = (z_base_g + z_codon) * 0.5
        else:
            z_fused = self._masked_pool(z_base_seq, base_mask)

        logits = self.classifier(z_fused)
        return logits, z_fused, z_codon


class SupervisedContrastiveLoss(nn.Module):
    def __init__(self, temperature: float = 0.1):
        super().__init__()
        self.temperature = temperature

    def forward(self, z_a, z_b, labels):
        z_a = F.normalize(z_a, dim=-1, eps=1e-8)
        z_b = F.normalize(z_b, dim=-1, eps=1e-8)
        logits = torch.matmul(z_a, z_b.T) / self.temperature
        logits = logits - logits.detach().max(1, keepdim=True)[0]
        labels = labels.view(-1, 1)
        mask = torch.eq(labels, labels.T).float().to(z_a.device)
        log_p = logits - torch.log(torch.exp(logits).sum(1, keepdim=True) + 1e-10)
        return -(mask * log_p).sum(1).div(mask.sum(1).clamp(1e-6)).mean()


class HexaCrossLoss(nn.Module):
    def __init__(self, cls_weight=1.0, contra_weight=0.1, smoothing=0.1, temperature=0.1):
        super().__init__()
        self.cls_weight = cls_weight
        self.contra_weight = contra_weight
        self.smoothing = smoothing
        self.sup_con = SupervisedContrastiveLoss(temperature)

    def forward(self, logits, targets, z_fused, z_codon=None, use_supcon=True):
        ce = F.cross_entropy(logits, targets, label_smoothing=self.smoothing)
        if use_supcon and z_codon is not None:
            contra = self.sup_con(z_fused, z_codon, targets)
            if not torch.isnan(contra):
                return self.cls_weight * ce + self.contra_weight * contra
        return ce


def compute_cub_scores(viral_seqs, host_seqs, pseudocount=1.0):
    BASE = {"A": 0, "C": 1, "G": 2, "T": 3}

    def _count(seqs):
        counts = torch.zeros(64)
        for seq in seqs:
            for i in range(0, len(seq) - 2, 3):
                c = seq[i:i+3].upper()
                if all(b in BASE for b in c):
                    counts[BASE[c[0]] * 16 + BASE[c[1]] * 4 + BASE[c[2]]] += 1
        return counts

    vc = _count(viral_seqs) + pseudocount
    hc = _count(host_seqs) + pseudocount
    return torch.log(vc / vc.sum()) - torch.log(hc / hc.sum())


if __name__ == "__main__":
    print("Parameter count test")
    m = SCoV(d_model=128)
    total = sum(p.numel() for p in m.parameters())
    print(f"SCoV total parameters: {total / 1e6:.3f} M")

    print("\nForward test")
    B, L, Lc = 4, 500, 168
    x = torch.randn(B, 4, L)
    cid = torch.randint(0, 65, (B, 6, Lc))
    bm = torch.zeros(B, L, dtype=torch.bool)
    cm = (cid == 64)
    logits, zf, zc = m(x, cid, bm, cm)
    print(f"logits shape: {logits.shape}")
    print(f"z_fused shape: {zf.shape}")
    print(f"z_codon shape: {zc.shape}")

    print("\nAblation: w/o CIA")
    ab = AblationSCoV(d_model=128, use_cia=False)
    logits2, zf2, zc2 = ab(x, cid, bm, cm)
    print(f"logits shape: {logits2.shape}")

    print("\nAblation: w/o Frame Prior")
    ab2 = AblationSCoV(d_model=128, use_cia=True, use_frame_prior=False)
    logits3, zf3, zc3 = ab2(x, cid, bm, cm)
    print(f"logits shape: {logits3.shape}")

    print("\nAblation: SingleFrame +0, no CIA")
    ab3 = AblationSCoV(d_model=128, use_codon=True, use_cia=False, codon_frame_mode="plus0")
    logits4, zf4, zc4 = ab3(x, cid, bm, cm)
    print(f"logits shape: {logits4.shape}; z_codon: {zc4.shape if zc4 is not None else None}")

    print("\nAblation: BaseOnly / no codon")
    ab4 = AblationSCoV(d_model=128, use_codon=False, use_cia=False)
    logits5, zf5, zc5 = ab4(x, cid, bm, cm)
    print(f"logits shape: {logits5.shape}; z_codon: {zc5}")

    print("\nmodel.py validation passed")
