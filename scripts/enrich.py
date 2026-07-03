#!/usr/bin/env python3
"""Enrich chord progressions via gradient ascent on z.

Usage:
  python scripts/enrich.py --chords "C | Am | F | G | C"
  python scripts/enrich.py --chords "C | F | G | C" --7c 0.5 --dtmcvi 0.3
  python scripts/enrich.py --chords "C | F | G7 | Am" --show-chords
  python scripts/enrich.py --chords "C | F | G | C" --vnspc 0.4 --vdr 1.0
  python scripts/enrich.py --file progresiones.txt
"""
import sys, argparse, warnings
from pathlib import Path

import numpy as np
import torch

warnings.filterwarnings('ignore')
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from cvae.models.rvae import RVAE
from chords.chord_encoder import (
    decode_chord, progression_to_encoding,
    SEVENTH_SLOT, EXT_SLOT, BASS_SLOT,
)

KEY_NAMES = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
C_NAMES = ['7C', 'VNSPC', 'DTMCVI', 'VDR']
ARG_NAMES = ['7c', 'vnspc', 'dtmcvi', 'vdr']

LR = 0.5
STEPS = 90
REG = 0.005


def analyze(seq_tensor):
    n7 = 0
    n_ext = 0
    for t in range(seq_tensor.size(0)):
        vec = seq_tensor[t]
        s_idx = int(vec[SEVENTH_SLOT:EXT_SLOT].argmax())
        if s_idx != 0:
            n7 += 1
        n_ext += vec[EXT_SLOT:BASS_SLOT].sum().item()
    return n7, int(n_ext)


def detect_key(seq_tensor):
    roots = seq_tensor[:, :12].argmax(dim=-1)
    counts = torch.bincount(roots.long(), minlength=12)
    key = counts.argmax().item()
    key_onehot = torch.zeros(12)
    key_onehot[key] = 1.0
    return key_onehot


def make_cond(key_onehot):
    perceptual = torch.zeros(4)
    return torch.cat([perceptual, key_onehot]).unsqueeze(0).to(DEVICE)


@torch.no_grad()
def encode_progression(model, seq):
    seq = seq.unsqueeze(0).to(DEVICE)
    lengths = torch.tensor([seq.size(1)], device=DEVICE)
    mu, logvar = model.encoder(seq, lengths)
    z = model.reparameterize(mu, logvar)
    gen = model.generate(z, make_cond(detect_key(seq[0])), max_len=seq.size(1), device=DEVICE)
    return z, gen[0]


@torch.no_grad()
def decode_progression(model, z, n, cond):
    gen = model.generate(z, cond, max_len=n, device=DEVICE)
    return [decode_chord(gen[0, t].cpu().numpy()) for t in range(n)], gen[0]


def enrich_gradient(model, z_orig, cond, n, strengths):
    z_opt = z_orig.clone().detach().requires_grad_(True)

    with torch.no_grad():
        c_init = model.c_predictor(z_opt)[0]

    if all(s == 0 for s in strengths):
        chords, seq = decode_progression(model, z_opt, n, cond)
        return chords, seq, [c_init.tolist()]

    targets = [c_init[d].item() + strengths[d] * (1.0 - c_init[d].item()) for d in range(4)]
    active = [d for d in range(4) if strengths[d] > 0]

    opt = torch.optim.Adam([z_opt], lr=LR)
    for _ in range(STEPS):
        opt.zero_grad()
        c_pred = model.c_predictor(z_opt)[0]
        loss = REG * torch.nn.functional.mse_loss(z_opt, z_orig.detach())
        for d in active:
            loss += torch.nn.functional.mse_loss(
                c_pred[d], torch.tensor(targets[d], device=DEVICE))
        loss.backward()
        torch.nn.utils.clip_grad_norm_([z_opt], max_norm=1.0)
        opt.step()

    with torch.no_grad():
        c_final = model.c_predictor(z_opt)[0]
        chords, seq = decode_progression(model, z_opt.detach(), n, cond)

    return chords, seq, [c_final.tolist()]


def load_model(checkpoint_path):
    model = RVAE(latent_dim=32, condition_dim=16, z_only_decoder=True).to(DEVICE)
    ckpt = torch.load(checkpoint_path, map_location=DEVICE, weights_only=False)
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()
    return model


def main():
    parser = argparse.ArgumentParser(description='Enrich chord progressions')
    parser.add_argument('--chords', type=str, default=None,
                        help='Chord progression, e.g. "C | Am | F | G | C"')
    parser.add_argument('--file', type=str, default=None,
                        help='File with one progression per line')
    parser.add_argument('--checkpoint', type=str,
                        default='/home/pepebeats/SCL_2.0/checkpoints/rvae_key_v13/best.pt')
    parser.add_argument('--show-chords', action='store_true',
                        help='Show model reconstruction too')
    parser.add_argument('--7c', type=float, default=1.0,
                        help='Strength for 7C (0=none, 1=full)')
    parser.add_argument('--vnspc', type=float, default=0.0,
                        help='Strength for VNSPC (0=none, 1=full)')
    parser.add_argument('--dtmcvi', type=float, default=0.0,
                        help='Strength for DTMCVI (0=none, 1=full)')
    parser.add_argument('--vdr', type=float, default=0.0,
                        help='Strength for VDR (0=none, 1=full)')
    args = parser.parse_args()

    if not args.chords and not args.file:
        parser.error('Provide --chords or --file')

    strengths = [args.__dict__[name] for name in ARG_NAMES]
    has_targets = any(s > 0 for s in strengths)
    label_targets = ', '.join(f'{n}={s}' for n, s in zip(C_NAMES, strengths) if s > 0)

    print(f'Device: {DEVICE}')
    model = load_model(args.checkpoint)
    print('Model loaded')

    lines = []
    if args.chords:
        lines.append(args.chords)
    if args.file:
        with open(args.file) as f:
            lines.extend(line.strip() for line in f if line.strip())

    for line in lines:
        chord_names = [c.strip() for c in line.replace(',', '|').split('|') if c.strip()]
        if not chord_names:
            continue

        seq_np = progression_to_encoding(chord_names)
        n = seq_np.shape[0]
        seq = torch.from_numpy(seq_np).float()

        cond = make_cond(detect_key(seq))
        z_orig, gen_orig = encode_progression(model, seq)

        with torch.no_grad():
            c_orig = model.c_predictor(z_orig)[0].tolist()

        chords_g, seq_g, c_list = enrich_gradient(
            model, z_orig, cond, n, strengths)
        c_final = c_list[0] if c_list else [0] * 4
        n7_orig, ext_orig = analyze(gen_orig)
        n7_g, ext_g = analyze(seq_g)

        print(f'\n{"─"*60}')
        print(f'  {n} chords')
        if label_targets:
            print(f'  targets: {label_targets}')
        print(f'  In:  {" | ".join(chord_names)}')
        print(f'  Enr: {" | ".join(chords_g)}')
        if args.show_chords:
            orig_chords, _ = decode_progression(model, z_orig, n, cond)
            print(f'  Rec: {" | ".join(orig_chords)}')
        print(f'  7ths:  {n7_orig:>2d}/{n:<2d} → {n7_g:>2d}/{n:<2d}')
        print(f'  Ext:   {ext_orig:>2d} → {ext_g:>2d}')
        if has_targets:
            print(f'  ── Complexity ──')
            for d in range(4):
                arrow = f'↑{strengths[d]:.1f}' if strengths[d] > 0 else ' –'
                print(f'  {C_NAMES[d]:>6s}: {c_orig[d]:.3f} → {c_final[d]:.3f}{arrow}')


if __name__ == '__main__':
    main()
