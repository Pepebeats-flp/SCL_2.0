import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from chords.chord_encoder import decode_chord
from cvae.dataset import CONDITION_COLS, CONDITION_DIM, ChordProgressionDataset
from cvae.model import CVAE


def generate(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Device: {device}')

    use_conditioning = args.condition_dim is not None and args.condition_dim > 0

    model = CVAE(latent_dim=args.latent_dim, condition_dim=args.condition_dim).to(device)

    ckpt = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(ckpt['model_state_dict'])
    print(f'Loaded checkpoint from epoch {ckpt.get("epoch", "?")}')

    # Load a few progressions from the dataset
    dataset = ChordProgressionDataset(max_len=args.max_len, use_conditioning=use_conditioning)
    indices = list(range(min(args.num_samples, len(dataset))))

    print(f'\nGenerating {len(indices)} samples with '
          f'{"conditioning" if use_conditioning else "no conditioning"}:\n')

    for idx in indices:
        if use_conditioning:
            seq, n, cond_original = dataset[idx]
            cond = cond_original.unsqueeze(0).to(device)
        else:
            seq, n = dataset[idx]
            cond = None
        seq = seq.unsqueeze(0).to(device)
        lengths = torch.tensor([n], device=device)

        model.eval()
        with torch.no_grad():
            logits, mu, logvar, z = model(seq, lengths, z_cond=cond)

        # Decode original input
        original_chords = []
        for t in range(n):
            chord = decode_chord(seq[0, t].cpu().numpy())
            original_chords.append(chord)

        # Decode reconstruction
        probs = torch.sigmoid(logits[0, :n])
        recon_chords = []
        for t in range(n):
            chord = decode_chord(probs[t].cpu().numpy())
            recon_chords.append(chord)

        # Generate new progression from latent + conditioning
        with torch.no_grad():
            z_gen = torch.randn(1, args.latent_dim, device=device)
            if use_conditioning:
                # Use original conditioning
                z_gen = torch.cat([z_gen, cond], dim=-1)
            gen_seq = model.generate(z_gen, max_len=n, device=device)

        gen_chords = []
        for t in range(n):
            chord = decode_chord(gen_seq[0, t].cpu().numpy())
            gen_chords.append(chord)

        print(f'--- Sample {idx} (n_chords={n}) ---')
        print(f'Original:     {" | ".join(original_chords[:16])}')
        print(f'Reconstructed: {" | ".join(recon_chords[:16])}')
        print(f'Generated:    {" | ".join(gen_chords[:16])}')

        if use_conditioning:
            cond_np = cond_original.cpu().numpy()
            print(f'Conditioning: PCS={cond_np[0]:.4f}, '
                  f'7C={cond_np[1]:.4f}, VNSPC={cond_np[2]:.4f}, '
                  f'DTMCVI={cond_np[3]:.4f}, VDR={cond_np[4]:.4f}')
        print()


def generate_conditioned(args):
    """Generate with explicit PCS conditioning values."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Device: {device}')

    model = CVAE(latent_dim=args.latent_dim, condition_dim=args.condition_dim).to(device)
    ckpt = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(ckpt['model_state_dict'])
    print(f'Loaded checkpoint from epoch {ckpt.get("epoch", "?")}')

    n = args.gen_length
    pcs_values = args.pcs_values if args.pcs_values else [0.0, 0.3, 0.6]

    print(f'\nGenerating progressions of length {n} with different PCS levels:\n')

    for pcs in pcs_values:
        cond = torch.zeros(1, args.condition_dim, device=device)
        cond[0, 0] = pcs

        model.eval()
        with torch.no_grad():
            z = torch.randn(1, args.latent_dim, device=device)
            z_cond = torch.cat([z, cond], dim=-1)
            gen_seq = model.generate(z_cond, max_len=n, device=device)

        chords = [decode_chord(gen_seq[0, t].cpu().numpy()) for t in range(n)]
        print(f'PCS={pcs:.1f}: {" | ".join(chords)}')

    print()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('checkpoint', type=str, help='Path to checkpoint .pt file')
    parser.add_argument('--latent-dim', type=int, default=64)
    parser.add_argument('--condition-dim', type=int, default=None,
                        help='5 for PCS conditioning')
    parser.add_argument('--num-samples', type=int, default=3)
    parser.add_argument('--max-len', type=int, default=128)
    parser.add_argument('--gen-length', type=int, default=8,
                        help='Length for unconditioned generation')
    parser.add_argument('--pcs-values', type=float, nargs='+', default=None,
                        help='PCS values for conditioned generation test')
    parser.add_argument('--conditioned-gen', action='store_true',
                        help='Use conditioned generation mode with PCS sweep')
    args = parser.parse_args()

    if args.conditioned_gen:
        generate_conditioned(args)
    else:
        generate(args)
