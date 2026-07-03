import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from chords.chord_encoder import decode_chord, encode_chord, parse_chord
from chords.substitution_rules import SubstitutionRuleEngine
from cvae.dataset import ChordProgressionDataset, PERCEPTUAL_COLS
from cvae.models.rvae import RVAE

KEY_NAMES = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']

def detect_key(chord_seq):
    roots = chord_seq[:, :12].argmax(dim=-1)
    counts = torch.zeros(12)
    for r in roots:
        counts[r.long()] += 1
    key = counts.argmax().item()
    key_onehot = torch.zeros(12)
    key_onehot[key] = 1.0
    return key_onehot, key


def get_device(args):
    if args.device:
        return torch.device(args.device)
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def generate_enrichment(args):
    device = get_device(args)
    print(f'Device: {device}')

    model = RVAE(latent_dim=args.latent_dim, condition_dim=args.condition_dim).to(device)
    ckpt = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(ckpt['model_state_dict'])
    print(f'Loaded checkpoint from epoch {ckpt.get("epoch", "?")}')

    cond_dim = args.condition_dim

    if args.sample_prior:
        c_perceptual = torch.tensor(args.c_values[:4], dtype=torch.float, device=device).unsqueeze(0)
        if cond_dim > 4:
            key_onehot = torch.zeros(1, 12, device=device)
            if args.key >= 0:
                key_onehot[0, args.key] = 1.0
            c = torch.cat([c_perceptual, key_onehot], dim=-1)
        else:
            c = c_perceptual

        key_str = f'key = {KEY_NAMES[args.key]}' if args.key >= 0 and cond_dim > 4 else 'no key'
        print(f'\nSampling from prior with C = {args.c_values[:4]}, {key_str}\n')
        with torch.no_grad():
            mu_prior, logvar_prior = model.prior(c)
            print(f'Prior: μ={mu_prior[0,:4].tolist()}..., σ={torch.exp(0.5*logvar_prior)[0,:4].tolist()}...')
            z, _, _ = model.sample_prior(c, n=args.num_samples, device=device)
            for i in range(args.num_samples):
                gen_seq = model.generate(z[i:i+1], c, max_len=args.gen_length, device=device)
                chords = [decode_chord(gen_seq[0, t].cpu().numpy()) for t in range(args.gen_length)]
                print(f'Sample {i}: {" | ".join(chords)}')
        return

    dataset = ChordProgressionDataset(
        max_len=args.max_len, use_conditioning=True,
        cond_cols=PERCEPTUAL_COLS if args.condition_dim >= 4 else None,
    )

    indices = list(range(min(args.num_examples, len(dataset))))

    heuristic_vec = None
    if args.heuristic_vector:
        heuristic_vec = torch.load(args.heuristic_vector, map_location=device)
        if isinstance(heuristic_vec, torch.Tensor):
            heuristic_vec = heuristic_vec.to(device).flatten()
        print(f'Loaded heuristic vector from {args.heuristic_vector} (norm={heuristic_vec.norm():.4f})')

    print(f'\nGenerating enriched progressions from {len(indices)} examples:\n')

    for idx in indices:
        seq, n, c_orig = dataset[idx]
        seq = seq.unsqueeze(0).to(device)
        c_orig = c_orig.unsqueeze(0).to(device)
        lengths = torch.tensor([n], device=device)

        model.eval()
        with torch.no_grad():
            mu, logvar = model.encode(seq, lengths)
            z_enc = model.reparameterize(mu, logvar)

        original_chords = [decode_chord(seq[0, t].cpu().numpy()) for t in range(n)]

        # === Axis 2: Chord substitution (from pairs data) ===
        if args.substitute:
            engine = SubstitutionRuleEngine()
            key_idx = -1
            if cond_dim > 4:
                _, key_idx = detect_key(seq[0])
            subbed = engine.apply_substitutions(original_chords, rate=args.substitution_rate, key=key_idx)
            sub_changed = sum(1 for a, b in zip(original_chords, subbed) if a != b)
            if sub_changed > 0:
                sub_vecs = []
                for cname in subbed:
                    parsed = parse_chord(cname)
                    sub_vecs.append(encode_chord(parsed) if parsed else encode_chord(None))
                sub_tensor = torch.tensor(np.array(sub_vecs), dtype=torch.float, device=device).unsqueeze(0)
                seq = sub_tensor
                lengths = torch.tensor([n], device=device)
                model.eval()
                with torch.no_grad():
                    mu, logvar = model.encode(seq, lengths)
                z_enc = model.reparameterize(mu, logvar)
            print(f'Substitutions: {sub_changed}/{n} chords changed')
            for i in range(min(n, 16)):
                if original_chords[i] != subbed[i]:
                    print(f'  [{i}] {original_chords[i]} → {subbed[i]}')
            original_chords = subbed
            seq = sub_tensor if sub_changed > 0 else seq

        c_perceptual = c_orig[0, :4].clone()

        # Detect key from original progression (only if model uses key)
        if cond_dim > 4:
            key_onehot, key_idx = detect_key(seq[0])
            key_onehot = key_onehot.to(device)
        else:
            key_onehot = None
            key_idx = -1

        c_perceptual_mod = c_perceptual.clone()
        for dim_idx, (dim_name, multiplier) in enumerate(zip(
            ['7C', 'VNSPC', 'DTMCVI', 'VDR'], args.multipliers
        )):
            c_perceptual_mod[dim_idx] = torch.clamp(c_perceptual[dim_idx] * multiplier, 0, 1)

        if cond_dim > 4:
            c_mod = torch.cat([c_perceptual_mod.unsqueeze(0), key_onehot.unsqueeze(0)], dim=-1)
        else:
            c_mod = c_perceptual_mod.unsqueeze(0)

        print(f'--- Example {idx} (n={n}) ---')
        key_str = KEY_NAMES[key_idx] if key_idx >= 0 else 'N/A'
        print(f'Original C: 7C={c_orig[0,0]:.3f} VNSPC={c_orig[0,1]:.3f} '
              f'DTMCVI={c_orig[0,2]:.3f} VDR={c_orig[0,3]:.3f} '
              f'key={key_str}')
        print(f'Original:    {" | ".join(original_chords[:12])}')

        c_mod_np = c_perceptual_mod.cpu().numpy()
        key_str2 = KEY_NAMES[key_idx] if key_idx >= 0 else 'N/A'
        print(f'Modified C:  7C={c_mod_np[0]:.3f} VNSPC={c_mod_np[1]:.3f} '
              f'DTMCVI={c_mod_np[2]:.3f} VDR={c_mod_np[3]:.3f} '
              f'key={key_str2}')
        print(f'Δ:           7C={c_mod_np[0]-c_orig[0,0].item():+.3f} '
              f'VNSPC={c_mod_np[1]-c_orig[0,1].item():+.3f} '
              f'DTMCVI={c_mod_np[2]-c_orig[0,2].item():+.3f} '
              f'VDR={c_mod_np[3]-c_orig[0,3].item():+.3f}')

        if heuristic_vec is not None:
            # Axis 1: Heuristic enrichment (from centroid diff of 877k)
            z = z_enc + args.alpha * heuristic_vec
            c_gen = c_mod
            print(f'Heuristic enrichment: α={args.alpha:.2f}')
        else:
            # Interpolate z between encoded (structure) and prior (new C)
            alpha = args.alpha
            if alpha < 1.0:
                with torch.no_grad():
                    mu_prior, _ = model.prior(c_mod)
                    z_prior = model.reparameterize(mu_prior, torch.zeros_like(mu_prior))
                z = alpha * z_enc + (1 - alpha) * z_prior
                print(f'z interpolation: α={alpha:.2f} (1=original structure, 0=pure prior)')
            else:
                z = z_enc
            c_gen = c_mod

        with torch.no_grad():
            gen_seq = model.generate(z, c_gen, max_len=n, device=device, deterministic=False)

        gen_chords = [decode_chord(gen_seq[0, t].cpu().numpy()) for t in range(n)]
        print(f'Enriched:    {" | ".join(gen_chords[:12])}')

        if n > 12:
            print(f'  ... ({n - 12} more chords)')

        print()


def interpolate_complexity(args):
    device = get_device(args)
    print(f'Device: {device}')

    model = RVAE(latent_dim=args.latent_dim, condition_dim=args.condition_dim).to(device)
    ckpt = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(ckpt['model_state_dict'])
    print(f'Loaded checkpoint from epoch {ckpt.get("epoch", "?")}')

    dataset = ChordProgressionDataset(
        max_len=args.max_len, use_conditioning=True,
        cond_cols=PERCEPTUAL_COLS if args.condition_dim >= 4 else None,
    )

    seq, n, c_orig = dataset[args.example_idx]
    seq = seq.unsqueeze(0).to(device)
    c_orig = c_orig.unsqueeze(0).to(device)
    lengths = torch.tensor([n], device=device)

    model.eval()
    with torch.no_grad():
        mu, logvar = model.encode(seq, lengths)
        z = model.reparameterize(mu, logvar)

    cond_dim = args.condition_dim
    if cond_dim > 4:
        key_onehot, key_idx = detect_key(seq[0])
        key_onehot = key_onehot.to(device)
    else:
        key_onehot = None
        key_idx = -1

    original_chords = [decode_chord(seq[0, t].cpu().numpy()) for t in range(n)]
    key_str = KEY_NAMES[key_idx] if key_idx >= 0 else 'N/A'
    print(f'Original: {" | ".join(original_chords[:12])}')
    print(f'Original C: {c_orig[0, :4].cpu().numpy().tolist()} | key = {key_str}\n')

    c_low = torch.zeros(1, 4, device=device)
    c_high = torch.ones(1, 4, device=device)

    for i, frac in enumerate(args.interpolation_steps):
        c_perceptual = c_low + (c_high - c_low) * frac
        c_perceptual[:, 0] = c_orig[0, 0] + (c_high[0, 0] - c_low[0, 0]) * frac
        c_perceptual = torch.clamp(c_perceptual, 0, 1)
        if cond_dim > 4:
            c = torch.cat([c_perceptual, key_onehot.unsqueeze(0)], dim=-1)
        else:
            c = c_perceptual

        with torch.no_grad():
            gen_seq = model.generate(z, c, max_len=n, device=device, deterministic=False)

        gen_chords = [decode_chord(gen_seq[0, t].cpu().numpy()) for t in range(min(n, 12))]
        c_np = c[0, :4].cpu().numpy()
        print(f'C={c_np.tolist()} (step {i}): {" | ".join(gen_chords)}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('checkpoint', type=str, help='Path to checkpoint .pt file')
    parser.add_argument('--latent-dim', type=int, default=64)
    parser.add_argument('--condition-dim', type=int, default=16)
    parser.add_argument('--num-examples', type=int, default=3)
    parser.add_argument('--num-samples', type=int, default=3)
    parser.add_argument('--max-len', type=int, default=128)
    parser.add_argument('--gen-length', type=int, default=8)
    parser.add_argument('--alpha', type=float, default=1.0,
                        help='Interpolation (1=z_enc, 0=z_prior) or enrichment strength')
    parser.add_argument('--multipliers', type=float, nargs=4, default=[1.0, 1.0, 1.0, 1.0],
                        help='Multipliers for [7C, VNSPC, DTMCVI, VDR] enrichment')
    parser.add_argument('--sample-prior', action='store_true',
                        help='Sample from prior instead of enriching existing')
    parser.add_argument('--c-values', type=float, nargs='+', default=[0.5, 0.5, 0.5, 0.5],
                        help='Perceptual C values for prior sampling')
    parser.add_argument('--key', type=int, default=-1,
                        help='Key index (0-11) for prior sampling, -1 = no key')
    parser.add_argument('--interpolate', action='store_true',
                        help='Interpolate between low and high complexity')
    parser.add_argument('--example-idx', type=int, default=0,
                        help='Example index for interpolation')
    parser.add_argument('--interpolation-steps', type=float, nargs='+',
                        default=[0.0, 0.25, 0.5, 0.75, 1.0])
    parser.add_argument('--device', type=str, default='',
                        help='Device: cpu, cuda, etc. Auto-detected if empty')
    parser.add_argument('--enrichment-model', type=str, default='',
                        help='Path to heurisic vector .pt for centroid-diff enrichment')
    parser.add_argument('--heuristic-vector', type=str, default='',
                        help='Path to heurisic vector .pt for centroid-diff enrichment '
                             '(e.g. checkpoints/rvae_key_v2/enrichment_vector.pt)')
    parser.add_argument('--substitute', action='store_true',
                        help='Apply chord substitutions (Axis 2) before enrichment')
    parser.add_argument('--substitution-rate', type=float, default=0.25,
                        help='Fraction of chords to substitute (0-1)')
    args = parser.parse_args()

    if args.interpolate:
        interpolate_complexity(args)
    elif args.sample_prior:
        generate_enrichment(args)
    else:
        generate_enrichment(args)