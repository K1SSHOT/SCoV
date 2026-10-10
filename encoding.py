"""Original nucleotide/reading-frame representation and augmentation."""
import torch
import torch.nn.functional as F

def reverse_complement_batch(x_nt: torch.Tensor) -> torch.Tensor:
    rev = torch.flip(x_nt, dims=[1])
    return torch.where(rev < 4, 3 - rev, torch.full_like(rev, 4))


def maybe_reverse_complement(x_nt: torch.Tensor, prob: float) -> torch.Tensor:
    if prob <= 0:
        return x_nt
    choose = torch.rand(x_nt.size(0), device=x_nt.device) < prob
    if not choose.any():
        return x_nt
    rc = reverse_complement_batch(x_nt)
    return torch.where(choose[:, None], rc, x_nt)


def build_six_frame_codons(x_nt: torch.Tensor) -> torch.Tensor:
    """
    GPU-vectorized six-frame codon conversion.
    x_nt: [B, L], A/C/G/T/N = 0/1/2/3/4
    output: [B, 6, L//3+1], pad/ambiguous = 64
    """
    B, L = x_nt.shape
    max_codons = L // 3 + 1
    pad = 64

    strands = [x_nt, reverse_complement_batch(x_nt)]
    frames = []

    for strand in strands:
        for offset in (0, 1, 2):
            n = (L - offset) // 3
            out = torch.full(
                (B, max_codons), pad,
                device=x_nt.device, dtype=torch.long
            )
            if n > 0:
                sl = strand[:, offset:offset + n * 3].reshape(B, n, 3)
                valid = (sl < 4).all(dim=-1)
                ids = (
                    sl[:, :, 0].long() * 16
                    + sl[:, :, 1].long() * 4
                    + sl[:, :, 2].long()
                )
                ids = torch.where(valid, ids, torch.full_like(ids, pad))
                out[:, :n] = ids
            frames.append(out)

    return torch.stack(frames, dim=1)


def tokens_to_model_inputs(
    x_nt: torch.Tensor,
    needs_codon: bool,
    training: bool,
    cfg,
):
    if x_nt.ndim != 2 or x_nt.shape[1] < 3 or x_nt.dtype not in (torch.uint8, torch.int8, torch.int16, torch.int32, torch.int64) or bool(((x_nt < 0) | (x_nt > 4)).any()):
        raise ValueError("Expected [B, L>=3] DNA tokens in [0, 4]")
    x_nt = x_nt.long()

    # RC augmentation precedes codon construction, preserving the old training logic.
    if training:
        x_nt = maybe_reverse_complement(x_nt, cfg.rc_prob)

    base_mask = x_nt.eq(4)

    codon_ids = None
    codon_mask = None
    if needs_codon:
        codon_ids = build_six_frame_codons(x_nt)
        codon_mask = codon_ids.eq(64)

    x_base = F.one_hot(x_nt, num_classes=5)[:, :, :4].transpose(1, 2).float()

    # As in the old code, random nucleotide masking affects only the base branch.
    if training and cfg.mask_prob > 0:
        random_mask = (
            torch.rand(x_base.size(0), 1, x_base.size(2), device=x_base.device)
            < cfg.mask_prob
        )
        x_base = x_base.masked_fill(random_mask, 0.0)

    return x_base, codon_ids, base_mask, codon_mask


