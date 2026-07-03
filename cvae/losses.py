import torch
import torch.nn.functional as F

from .config import CHORD_DIM, LATENT_DIM, BETA, FREE_BITS, LAMBDA_COH, LAMBDA_TENS, LAMBDA_MOV

_ROOT_SLOT = 0
_ROOT_END = 12
_QUALITY_SLOT = 12
_SEVENTH_SLOT = 17
_EXT_SLOT = 22
_ALT_SLOT = 25
_ADDED_SLOT = 29
_BASS_SLOT = 32

KRUMHANSL_MAJOR = torch.tensor([
    6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88
])
KRUMHANSL_MINOR = torch.tensor([
    6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17
])

MAJOR_SCALE = {0, 2, 4, 5, 7, 9, 11}
MINOR_SCALE = {0, 2, 3, 5, 7, 8, 10}

ALL_KEYS = []
for root in range(12):
    ALL_KEYS.append(('major', root, {(root + d) % 12 for d in MAJOR_SCALE}))
    ALL_KEYS.append(('minor', root, {(root + d) % 12 for d in MINOR_SCALE}))


def infer_tonal_scale(chord_roots):
    profile = torch.zeros(12)
    for r in chord_roots:
        profile[r] += 1.0
    if profile.sum() == 0:
        return set(range(12))

    best_corr = -float('inf')
    best_scale = set(range(12))

    pad = len(KRUMHANSL_MAJOR) // 2
    corr_major = F.conv1d(
        profile.view(1, 1, -1),
        KRUMHANSL_MAJOR.view(1, 1, -1),
        padding=pad
    ).squeeze()
    corr_minor = F.conv1d(
        profile.view(1, 1, -1),
        KRUMHANSL_MINOR.view(1, 1, -1),
        padding=pad
    ).squeeze()

    stacked = torch.stack([corr_major, corr_minor])
    best_idx = stacked.argmax().item()
    best_mode_idx = best_idx // len(profile)
    best_root = best_idx % len(profile)
    best_mode = 'minor' if best_mode_idx == 1 else 'major'

    for mode, root, scale in ALL_KEYS:
        if mode == best_mode and root == best_root:
            best_scale = scale
            break

    return best_scale


def coherence_loss(logits, lengths, inputs):
    loss = 0.0
    n = 0
    for i in range(logits.size(0)):
        seq_len = lengths[i]
        logits_i = logits[i, :seq_len]

        root_logits = logits_i[:, :12]
        root_probs = torch.softmax(root_logits, dim=-1)
        pred_roots = root_probs.argmax(dim=-1)
        scale = infer_tonal_scale(pred_roots)

        scale_tensor = torch.tensor(sorted(scale), device=logits.device, dtype=torch.long)
        scale_mask = torch.zeros(12, device=logits.device)
        scale_mask[scale_tensor] = 1.0
        out_of_scale = 1.0 - scale_mask[pred_roots]

        chord_conf = root_probs.max(dim=-1).values
        loss += (chord_conf * out_of_scale).sum()
        n += seq_len

    return loss / n if n > 0 else 0.0


def tension_loss(logits, lengths):
    loss = 0.0
    n = 0
    for i in range(logits.size(0)):
        seq_len = lengths[i]
        probs = torch.sigmoid(logits[i, :seq_len])

        has_seventh = (probs[:, 17:22].sum(dim=-1) > 0.5).float()
        has_extension = (probs[:, 22:25].sum(dim=-1) > 0.5).float()
        has_added = (probs[:, 29:32].sum(dim=-1) > 0.5).float()
        is_less_than_4 = ((has_seventh + has_extension + has_added) < 0.5).float()

        tension_penalty = (probs.mean(dim=-1) * is_less_than_4).sum()
        loss += tension_penalty
        n += seq_len

    return loss / n if n > 0 else 0.0


def movement_loss(logits, lengths):
    loss = 0.0
    n = 0
    for i in range(logits.size(0)):
        seq_len = lengths[i]
        if seq_len < 2:
            continue

        root_logits = logits[i, :seq_len, :12]
        root_probs = F.softmax(root_logits, dim=-1)
        pred_roots = root_probs.argmax(dim=-1).float()

        root_diff = torch.abs(pred_roots[1:] - pred_roots[:-1])
        root_diff = torch.min(root_diff, 12 - root_diff)
        penalty = torch.clamp(root_diff - 7, min=0)
        loss += penalty.sum()
        n += (seq_len - 1)

    return loss / n if n > 0 else 0.0


def per_dimension_kl(mu, logvar):
    kl_per_dim = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp())
    return kl_per_dim


def cvae_loss(logits, targets, mu, logvar, lengths,
              beta=BETA, free_bits=FREE_BITS,
              lambda_coh=LAMBDA_COH, lambda_tens=LAMBDA_TENS, lambda_mov=LAMBDA_MOV,
              inputs=None):
    batch_size, max_len = targets.shape[:2]
    mask = torch.arange(max_len, device=lengths.device).unsqueeze(0) < lengths.unsqueeze(1)
    loss = F.binary_cross_entropy_with_logits(logits, targets, reduction='none')
    loss = (loss * mask.unsqueeze(-1)).sum() / mask.sum()
    recon_loss = loss

    kl = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=-1)
    kl = torch.max(kl - free_bits, torch.zeros_like(kl))
    kl_loss = kl.mean()

    total = recon_loss + beta * kl_loss

    if lambda_coh > 0 and inputs is not None:
        l_coh = coherence_loss(logits, lengths, inputs)
        total += lambda_coh * l_coh
    else:
        l_coh = torch.tensor(0.0)

    if lambda_tens > 0:
        l_tens = tension_loss(logits, lengths)
        total += lambda_tens * l_tens
    else:
        l_tens = torch.tensor(0.0)

    if lambda_mov > 0:
        l_mov = movement_loss(logits, lengths)
        total += lambda_mov * l_mov
    else:
        l_mov = torch.tensor(0.0)

    return total, recon_loss, kl_loss, l_coh, l_tens, l_mov
