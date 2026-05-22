import argparse
import json
import os
import sys
from pathlib import Path

import torch
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from cvae.dataset import create_dataloader, CONDITIONED_PARQUET
from cvae.model import CVAE
from cvae.losses import cvae_loss


def evaluate(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Device: {device}')

    use_conditioning = args.condition_dim is not None and args.condition_dim > 0

    test_loader = create_dataloader(
        args.parquet, batch_size=args.batch_size,
        shuffle=False, max_len=args.max_len,
        use_conditioning=use_conditioning, split='test',
        test_size=args.test_size,
    )

    model = CVAE(
        latent_dim=args.latent_dim,
        condition_dim=args.condition_dim,
    ).to(device)

    ckpt = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()

    epoch_info = ckpt.get('epoch', '?')
    print(f'Loaded checkpoint from epoch {epoch_info}')
    print(f'Test samples: {len(test_loader.dataset)}')

    total_loss = 0.0
    total_recon = 0.0
    total_kl = 0.0
    total_coh = 0.0
    total_tens = 0.0
    total_mov = 0.0
    n_total = 0

    with torch.no_grad():
        for batch in tqdm(test_loader, desc='Evaluating on test set'):
            if use_conditioning:
                x, lengths, cond = batch
                cond = cond.to(device)
            else:
                x, lengths = batch
                cond = None
            x = x.to(device)
            lengths = lengths.to(device)

            logits, mu, logvar, z = model(x, lengths, z_cond=cond)

            total, recon, kl, lcoh, ltens, lmov = cvae_loss(
                logits, x, mu, logvar, lengths,
                beta=args.beta, free_bits=args.free_bits,
                lambda_coh=args.lambda_coh,
                lambda_tens=args.lambda_tens,
                lambda_mov=args.lambda_mov,
                inputs=x,
            )

            b = x.size(0)
            total_loss += total.item() * b
            total_recon += recon.item() * b
            total_kl += kl.item() * b
            total_coh += lcoh.item() * b
            total_tens += ltens.item() * b
            total_mov += lmov.item() * b
            n_total += b

    total_loss /= n_total
    total_recon /= n_total
    total_kl /= n_total
    total_coh /= n_total
    total_tens /= n_total
    total_mov /= n_total

    print(f'\n=== Test Results ({n_total} samples) ===')
    print(f'  Total   Loss: {total_loss:.6f}')
    print(f'  Recon   Loss: {total_recon:.6f}')
    print(f'  KL      Loss: {total_kl:.6f}')
    print(f'  Coh     Loss: {total_coh:.6f}')
    print(f'  Tens    Loss: {total_tens:.6f}')
    print(f'  Mov     Loss: {total_mov:.6f}')
    print(f'========================================')

    if args.output:
        results = {
            'n_samples': n_total,
            'epoch': epoch_info,
            'checkpoint': args.checkpoint,
            'loss': {
                'total': round(total_loss, 6),
                'recon': round(total_recon, 6),
                'kl': round(total_kl, 6),
                'coherence': round(total_coh, 6),
                'tension': round(total_tens, 6),
                'movement': round(total_mov, 6),
            },
        }
        with open(args.output, 'w') as f:
            json.dump(results, f, indent=2)
        print(f'Results saved to {args.output}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Evaluate a trained CVAE checkpoint on the held-out test set')
    parser.add_argument('checkpoint', type=str, help='Path to checkpoint (.pt)')
    parser.add_argument('--parquet', type=str, default=None)
    parser.add_argument('--condition-dim', type=int, default=5,
                        help='Conditioning dimension (5 for PCS)')
    parser.add_argument('--batch-size', type=int, default=128)
    parser.add_argument('--max-len', type=int, default=256)
    parser.add_argument('--latent-dim', type=int, default=64)
    parser.add_argument('--test-size', type=int, default=1000,
                        help='Number of test samples held out')
    parser.add_argument('--beta', type=float, default=1.0)
    parser.add_argument('--free-bits', type=float, default=1.5)
    parser.add_argument('--lambda-coh', type=float, default=0.1)
    parser.add_argument('--lambda-tens', type=float, default=0.05)
    parser.add_argument('--lambda-mov', type=float, default=0.1)
    parser.add_argument('--output', type=str, default=None,
                        help='Path to save results JSON')
    args = parser.parse_args()

    if not os.path.isfile(args.checkpoint):
        print(f'Error: checkpoint not found: {args.checkpoint}')
        sys.exit(1)

    if args.condition_dim and args.parquet is None:
        args.parquet = str(CONDITIONED_PARQUET)
        print(f'Auto-set parquet to {args.parquet}')

    evaluate(args)
