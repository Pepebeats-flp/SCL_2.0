import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from chords.chord_encoder import decode_chord
from cvae.dataset import ChordProgressionDataset
from cvae.model import CVAE


def reharmonize(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Device: {device}')

    model = CVAE(latent_dim=args.latent_dim, condition_dim=args.condition_dim).to(device)
    ckpt = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(ckpt['model_state_dict'])
    print(f'Loaded checkpoint from epoch {ckpt.get("epoch", "?")}')

    dataset = ChordProgressionDataset(max_len=args.max_len, use_conditioning=True)

    pcs_targets = args.pcs_values if args.pcs_values else [0.0, 0.2, 0.4, 0.6]

    for sample_idx in range(args.num_samples):
        seq, n, cond_original = dataset[sample_idx]
        seq = seq.unsqueeze(0).to(device)
        lengths = torch.tensor([n], device=device)
        cond_original = cond_original.unsqueeze(0).to(device)

        model.eval()
        with torch.no_grad():
            mu, logvar = model.encoder(seq, lengths)
            z = model.reparameterize(mu, logvar)

        # Original chords
        orig_chords = [decode_chord(seq[0, t].cpu().numpy()) for t in range(n)]
        print(f'\n=== Sample {sample_idx} (n_chords={n}) ===')
        print(f'Original PCS: {cond_original[0, 0].item():.4f}')
        print(f'Original:      {" | ".join(orig_chords[:12])}')

        # Reconstruct (with original conditioning)
        with torch.no_grad():
            z_with_orig_cond = torch.cat([z, cond_original], dim=-1)
            logits = model.decoder(z_with_orig_cond, seq, lengths)
            probs = torch.sigmoid(logits[0, :n])
            recon_chords = [decode_chord(probs[t].cpu().numpy()) for t in range(n)]
        print(f'Reconstructed: {" | ".join(recon_chords[:12])}')

        # Reharmonize with different PCS levels
        for target_pcs in pcs_targets:
            cond_mod = cond_original.clone()
            cond_mod[0, 0] = target_pcs  # Override PCS

            with torch.no_grad():
                z_cond = torch.cat([z, cond_mod], dim=-1)
                logits = model.decoder(z_cond, seq, lengths)
                probs = torch.sigmoid(logits[0, :n])
                harm_chords = [decode_chord(probs[t].cpu().numpy()) for t in range(n)]

            pcs_label = f'PCS={target_pcs:.2f}'
            print(f'{pcs_label:12s}: {" | ".join(harm_chords[:12])}')

    print()


def generate_reharm(args):
    """Reharmonize by encoding, sampling latent, and decoding with varying PCS."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Device: {device}')

    model = CVAE(latent_dim=args.latent_dim, condition_dim=args.condition_dim).to(device)
    ckpt = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(ckpt['model_state_dict'])
    print(f'Loaded checkpoint from epoch {ckpt.get("epoch", "?")}')

    dataset = ChordProgressionDataset(max_len=args.max_len, use_conditioning=True)

    pcs_targets = args.pcs_values if args.pcs_values else [0.0, 0.2, 0.4, 0.6]

    for sample_idx in range(args.num_samples):
        seq, n, cond_original = dataset[sample_idx]
        seq = seq.unsqueeze(0).to(device)
        lengths = torch.tensor([n], device=device)
        cond_original = cond_original.unsqueeze(0).to(device)

        model.eval()
        with torch.no_grad():
            mu, logvar = model.encoder(seq, lengths)

        orig_chords = [decode_chord(seq[0, t].cpu().numpy()) for t in range(n)]
        print(f'\n=== Sample {sample_idx} (n_chords={n}) ===')
        print(f'Original PCS: {cond_original[0, 0].item():.4f}')
        print(f'Original:      {" | ".join(orig_chords[:12])}')

        for target_pcs in pcs_targets:
            cond_mod = cond_original.clone()
            cond_mod[0, 0] = target_pcs

            with torch.no_grad():
                z = model.reparameterize(mu, logvar)
                z_cond = torch.cat([z, cond_mod], dim=-1)
                logits = model.decoder(z_cond, seq, lengths)
                probs = torch.sigmoid(logits[0, :n])
                harm_chords = [decode_chord(probs[t].cpu().numpy()) for t in range(n)]

            pcs_label = f'PCS={target_pcs:.2f}'
            print(f'{pcs_label:12s}: {" | ".join(harm_chords[:12])}')

    print()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('checkpoint', type=str)
    parser.add_argument('--latent-dim', type=int, default=64)
    parser.add_argument('--condition-dim', type=int, default=5)
    parser.add_argument('--num-samples', type=int, default=3)
    parser.add_argument('--max-len', type=int, default=128)
    parser.add_argument('--pcs-values', type=float, nargs='+',
                        default=[0.0, 0.2, 0.4, 0.6])
    parser.add_argument('--sample-latent', action='store_true',
                        help='Sample z from posterior instead of using mean')
    args = parser.parse_args()

    if args.sample_latent:
        generate_reharm(args)
    else:
        reharmonize(args)
