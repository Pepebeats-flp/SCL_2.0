#!/usr/bin/env python3
"""Enrichment via gradient ascent on z using c_predictor.

Strategy:
  encode real progression → z
  freeze model, set z.requires_grad = True
  gradient ascent on MSE(c_pred(z)[dim], 1.0) → pushes C to 1.0
  decode optimized z → enriched chords

Usage:
  python scripts/enrich_gradient.py --example 0
  python scripts/enrich_gradient.py --example 0 --show-chords
  python scripts/enrich_gradient.py --examples 0 5 10 50 100
  python scripts/enrich_gradient.py --example 0 --all-dims
"""
import sys, argparse, warnings
from pathlib import Path

import numpy as np
import torch

warnings.filterwarnings('ignore')
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from cvae.dataset import ChordProgressionDataset, PERCEPTUAL_COLS
from cvae.models.rvae import RVAE
from chords.chord_encoder import decode_chord, SEVENTH_SLOT, EXT_SLOT, BASS_SLOT

KEY_NAMES = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
C_NAMES = ['7C', 'VNSPC', 'DTMCVI', 'VDR']


def analyze(seq_tensor):
    """Return (n7, n_ext) from (T,48) tensor."""
    n7 = 0
    n_ext = 0
    for t in range(seq_tensor.size(0)):
        vec = seq_tensor[t]
        s_idx = int(vec[SEVENTH_SLOT:EXT_SLOT].argmax())
        if s_idx != 0:
            n7 += 1
        n_ext += vec[EXT_SLOT:BASS_SLOT].sum().item()
    return n7, int(n_ext)


def enrich_gradient(model, z_orig, cond, n, target_dim=0, lr=0.5, steps=60, reg=0.005):
    z_opt = z_orig.clone().detach().requires_grad_(True)

    with torch.no_grad():
        init_c = model.c_predictor(z_opt)[0, target_dim].item()
        if init_c < 0.05:
            z_opt = (z_orig + torch.randn_like(z_orig) * 0.5).detach().requires_grad_(True)

    opt = torch.optim.Adam([z_opt], lr=lr)
    for _ in range(steps):
        opt.zero_grad()
        c_pred = model.c_predictor(z_opt)
        loss = (torch.nn.functional.mse_loss(c_pred[0, target_dim], torch.tensor(1.0, device=DEVICE))
                + reg * torch.nn.functional.mse_loss(z_opt, z_orig.detach()))
        loss.backward()
        torch.nn.utils.clip_grad_norm_([z_opt], max_norm=1.0)
        opt.step()

    with torch.no_grad():
        c_final = model.c_predictor(z_opt)[0, target_dim].item()
        gen = model.generate(z_opt.detach(), cond, max_len=n, device=DEVICE)

    chords = [decode_chord(gen[0, t].cpu().numpy()) for t in range(n)]
    return chords, gen[0], c_final


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=str,
                        default='/home/pepebeats/SCL_2.0/checkpoints/rvae_key_v13/best.pt')
    parser.add_argument('--example', type=int, default=None,
                        help='Single example index')
    parser.add_argument('--examples', type=int, nargs='+', default=None,
                        help='Multiple example indices')
    parser.add_argument('--lr', type=float, default=0.5)
    parser.add_argument('--steps', type=int, default=60)
    parser.add_argument('--reg', type=float, default=0.005)
    parser.add_argument('--dim', type=int, default=0,
                        help='C dim to maximize: 0=7C, 1=VNSPC, 2=DTMCVI, 3=VDR')
    parser.add_argument('--all-dims', action='store_true',
                        help='Try all 4 C dims')
    parser.add_argument('--show-chords', action='store_true')
    parser.add_argument('--max-len', type=int, default=128)
    args = parser.parse_args()

    print(f'Device: {DEVICE}')

    model = RVAE(latent_dim=32, condition_dim=16, z_only_decoder=True).to(DEVICE)
    ckpt = torch.load(args.checkpoint, map_location=DEVICE, weights_only=False)
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()
    print(f'Loaded epoch {ckpt.get("epoch", "?")}')

    dataset = ChordProgressionDataset(
        max_len=args.max_len, use_conditioning=True,
        cond_cols=PERCEPTUAL_COLS, use_key=True,
    )

    indices = args.examples if args.examples else ([args.example] if args.example is not None else [0])

    for idx in indices:
        seq, n, cond = dataset[idx]
        seq = seq[:n].unsqueeze(0).to(DEVICE)
        cond = cond.unsqueeze(0).to(DEVICE)
        lengths = torch.tensor([n], device=DEVICE)

        with torch.no_grad():
            mu, logvar = model.encoder(seq, lengths)
            z_orig = model.reparameterize(mu, logvar)
            c_pred_orig = model.c_predictor(z_orig)[0]
            gen_orig = model.generate(z_orig, cond, max_len=n, device=DEVICE)

        orig_chords = [decode_chord(gen_orig[0, t].cpu().numpy()) for t in range(n)]
        n7_orig, ext_orig = analyze(gen_orig[0])

        if args.all_dims:
            print(f'\n{"="*70}')
            print(f'Example {idx} ({n} chords)')
            print(f'Original: {" | ".join(orig_chords[:12])}')
            print(f'Original: {n7_orig}/{n} 7ths, {ext_orig} ext')
            print(f'C_pred:   [{", ".join(f"{c:.3f}" for c in c_pred_orig.tolist())}]')
            print(f'{"="*70}')
            print(f'{"Dim":>10s}  {"C_orig":>8s}  {"C_grad":>8s}  {"7ths":>6s}  {"Ext":>4s}')
            print(f'{"-"*42}')

            for d in range(4):
                chords_g, seq_g, c_g = enrich_gradient(
                    model, z_orig, cond, n, target_dim=d,
                    lr=args.lr, steps=args.steps, reg=args.reg)
                n7_g, ext_g = analyze(seq_g)
                label = f'max {C_NAMES[d]}'
                print(f'{label:>10s}  {c_pred_orig[d].item():>8.3f}  {c_g:>8.3f}  '
                      f'{n7_g:>3d}/{n:<2d}  {ext_g:>3d}')

            print(f'{"-"*42}')
        else:
            chords_g, seq_g, c_g = enrich_gradient(
                model, z_orig, cond, n, target_dim=args.dim,
                lr=args.lr, steps=args.steps, reg=args.reg)
            n7_g, ext_g = analyze(seq_g)

            print(f'\nExample {idx} ({n} chords) — optimizing {C_NAMES[args.dim]}')
            if args.show_chords:
                print(f'  Orig: {" | ".join(orig_chords[:16])}')
                print(f'  Grad: {" | ".join(chords_g[:16])}')
            print(f'  7ths:  {n7_orig:>2d}/{n:<2d} → {n7_g:>2d}/{n:<2d}')
            print(f'  Ext:   {ext_orig:>2d} → {ext_g:>2d}')
            print(f'  C_pred({C_NAMES[args.dim]}): {c_pred_orig[args.dim].item():.3f} → {c_g:.3f}')


if __name__ == '__main__':
    main()
