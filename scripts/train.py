import argparse
import os
import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from cvae.config import (
    CHORD_DIM, BATCH_SIZE, LEARNING_RATE,
    MAX_EPOCHS, PATIENCE, GRAD_CLIP, LOG_DIR, CHECKPOINT_DIR,
    BETA, FREE_BITS, LAMBDA_COH, LAMBDA_TENS, LAMBDA_MOV,
    WORD_DROPOUT, KL_WARMUP_EPOCHS,
    KL_COLLAPSE_THRESHOLD, KL_COLLAPSE_PATIENCE,
    RECON_CHEAT_THRESHOLD, COH_WORSEN_PATIENCE,
)
from cvae.dataset import create_dataloader
from cvae.model import CVAE
from cvae.losses import cvae_loss


def check_quality_guards(epoch, args, kl_history, recon_history, coh_history):
    """Returns (stop, reason, hint) tuple."""
    if epoch <= args.kl_warmup:
        return False, '', ''

    # 1. KL Collapse
    if len(kl_history) >= args.kl_collapse_patience:
        recent = kl_history[-args.kl_collapse_patience:]
        if all(k < args.kl_collapse_threshold for k in recent):
            return True, (
                f'KL collapse: KL < {args.kl_collapse_threshold} '
                f'for {args.kl_collapse_patience} epochs after warmup'
            ), 'Increase word_dropout, reduce lr, or increase free_bits'

    # 2. Recon cheating (model memorizing without latent)
    if len(kl_history) >= 3:
        last_kl = kl_history[-1]
        last_recon = recon_history[-1]
        if last_recon < args.recon_cheat_threshold and last_kl < 0.01:
            if all(k < 0.01 for k in kl_history[-3:]):
                return True, (
                    f'Model cheating: recon={last_recon:.4f} with KL~0'
                ), 'Reduce lr, increase beta_target, or increase word_dropout'

    # 3. Coherence worsening
    if len(coh_history) >= args.coh_worsen_patience + 1:
        recent = coh_history[-(args.coh_worsen_patience + 1):]
        if all(recent[i + 1] > recent[i] for i in range(len(recent) - 1)):
            return True, (
                f'Coherence loss worsening for {args.coh_worsen_patience} '
                f'consecutive epochs'
            ), 'Increase lambda_coh or check tonal consistency'

    return False, '', ''


def train(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Device: {device}')

    use_conditioning = args.condition_dim is not None and args.condition_dim > 0

    train_loader = create_dataloader(
        args.parquet, batch_size=args.batch_size,
        shuffle=True, max_len=args.max_len,
        use_conditioning=use_conditioning, split='train',
    )
    val_loader = create_dataloader(
        args.parquet, batch_size=args.batch_size,
        shuffle=False, max_len=args.max_len,
        use_conditioning=use_conditioning, split='val',
    )

    condition_dim = args.condition_dim

    model = CVAE(
        latent_dim=args.latent_dim, condition_dim=condition_dim,
        word_dropout=args.word_dropout,
    ).to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    start_epoch = 1
    best_val_loss = float('inf')

    if args.resume:
        if os.path.isfile(args.resume):
            print(f'Resuming from checkpoint: {args.resume}')
            ckpt = torch.load(args.resume, map_location=device)
            model.load_state_dict(ckpt['model_state_dict'])
            optimizer.load_state_dict(ckpt['optimizer_state_dict'])
            start_epoch = ckpt.get('epoch', 0) + 1
            best_val_loss = ckpt.get('val_loss', float('inf'))
            print(f'  Resumed at epoch {start_epoch - 1}, best val loss: {best_val_loss:.4f}')
        else:
            print(f'Warning: checkpoint {args.resume} not found, starting from scratch')

    writer = SummaryWriter(args.log_dir)
    os.makedirs(args.checkpoint_dir, exist_ok=True)

    patience_counter = 0
    kl_history = []
    recon_history = []
    coh_history = []

    for epoch in range(start_epoch, args.epochs + 1):
        beta_effective = min(1.0, (epoch - 1) / max(args.kl_warmup, 1)) * args.beta

        model.train()
        train_loss = 0.0
        train_recon = 0.0
        train_kl = 0.0
        train_coh = 0.0
        train_tens = 0.0
        train_mov = 0.0
        n_train = 0

        pbar = tqdm(train_loader, desc=f'Epoch {epoch}/{args.epochs} [Train]')
        for batch in pbar:
            if use_conditioning:
                x, lengths, cond = batch
                cond = cond.to(device)
            else:
                x, lengths = batch
                cond = None
            x = x.to(device)
            lengths = lengths.to(device)

            optimizer.zero_grad()

            logits, mu, logvar, z = model(x, lengths, z_cond=cond)

            total, recon, kl, lcoh, ltens, lmov = cvae_loss(
                logits, x, mu, logvar, lengths,
                beta=beta_effective, free_bits=args.free_bits,
                lambda_coh=args.lambda_coh,
                lambda_tens=args.lambda_tens,
                lambda_mov=args.lambda_mov,
                inputs=x,
            )

            total.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optimizer.step()

            b = x.size(0)
            train_loss += total.item() * b
            train_recon += recon.item() * b
            train_kl += kl.item() * b
            train_coh += lcoh.item() * b
            train_tens += ltens.item() * b
            train_mov += lmov.item() * b
            n_train += b

            pbar.set_postfix({
                'loss': f'{total.item():.4f}',
                'recon': f'{recon.item():.4f}',
                'kl': f'{kl.item():.4f}',
            })

        train_loss /= n_train
        train_recon /= n_train
        train_kl /= n_train
        train_coh /= n_train
        train_tens /= n_train
        train_mov /= n_train

        model.eval()
        val_loss = 0.0
        val_recon = 0.0
        val_kl = 0.0
        n_val = 0

        with torch.no_grad():
            for batch in tqdm(val_loader, desc=f'Epoch {epoch}/{args.epochs} [Val]'):
                if use_conditioning:
                    x, lengths, cond = batch
                    cond = cond.to(device)
                else:
                    x, lengths = batch
                    cond = None
                x = x.to(device)
                lengths = lengths.to(device)

                logits, mu, logvar, z = model(x, lengths, z_cond=cond)

                total, recon, kl, _, _, _ = cvae_loss(
                    logits, x, mu, logvar, lengths,
                    beta=beta_effective, free_bits=args.free_bits,
                    lambda_coh=0, lambda_tens=0, lambda_mov=0,
                )

                b = x.size(0)
                val_loss += total.item() * b
                val_recon += recon.item() * b
                val_kl += kl.item() * b
                n_val += b

        val_loss /= n_val
        val_recon /= n_val
        val_kl /= n_val

        kl_history.append(train_kl)
        recon_history.append(train_recon)
        coh_history.append(train_coh)

        print(f'Epoch {epoch:3d} | '
              f'Train: {train_loss:.4f} (recon={train_recon:.4f}, kl={train_kl:.4f}, '
              f'coh={train_coh:.4f}, tens={train_tens:.4f}, mov={train_mov:.4f}) | '
              f'Val: {val_loss:.4f} (recon={val_recon:.4f}, kl={val_kl:.4f}) | '
              f'beta={beta_effective:.3f}')

        writer.add_scalars('loss/total', {'train': train_loss, 'val': val_loss}, epoch)
        writer.add_scalars('loss/recon', {'train': train_recon, 'val': val_recon}, epoch)
        writer.add_scalars('loss/kl', {'train': train_kl, 'val': val_kl}, epoch)
        writer.add_scalar('loss/coh', train_coh, epoch)
        writer.add_scalar('loss/tens', train_tens, epoch)
        writer.add_scalar('loss/mov', train_mov, epoch)

        should_stop, stop_reason, hint = check_quality_guards(
            epoch, args, kl_history, recon_history, coh_history,
        )
        if should_stop:
            crash_path = os.path.join(args.checkpoint_dir, f'crash_epoch{epoch}.pt')
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
                'stop_reason': stop_reason,
                'args': vars(args),
            }, crash_path)
            print(f'\n{"=" * 60}')
            print(f'[EARLY STOP] {stop_reason}')
            print(f'[HINT] {hint}')
            print(f'[SAVED] {crash_path}')
            print(f'{"=" * 60}')
            break

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            ckpt = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
                'args': vars(args),
            }
            path = os.path.join(args.checkpoint_dir, 'best.pt')
            torch.save(ckpt, path)
            print(f'  Saved checkpoint: {path}')

            last_path = os.path.join(args.checkpoint_dir, 'last.pt')
            torch.save({**ckpt, 'epoch': epoch}, last_path)
        else:
            patience_counter += 1
            last_path = os.path.join(args.checkpoint_dir, 'last.pt')
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
                'args': vars(args),
            }, last_path)
            if patience_counter >= args.patience:
                print(f'Early stopping at epoch {epoch}')
                break

    writer.close()
    print(f'\nTraining complete. Best val loss: {best_val_loss:.4f}')
    print(f'Final checkpoint: {os.path.join(args.checkpoint_dir, "last.pt")}')
    print(f'Best checkpoint: {os.path.join(args.checkpoint_dir, "best.pt")}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--parquet', type=str, default=None,
                        help='Path to dataset parquet')
    parser.add_argument('--batch-size', type=int, default=BATCH_SIZE)
    parser.add_argument('--max-len', type=int, default=256)
    parser.add_argument('--latent-dim', type=int, default=64)
    parser.add_argument('--condition-dim', type=int, default=None,
                        help='Conditioning dimension (5 for PCS)')
    parser.add_argument('--lr', type=float, default=LEARNING_RATE)
    parser.add_argument('--epochs', type=int, default=MAX_EPOCHS)
    parser.add_argument('--patience', type=int, default=PATIENCE)
    parser.add_argument('--grad-clip', type=float, default=GRAD_CLIP)
    parser.add_argument('--beta', type=float, default=BETA)
    parser.add_argument('--free-bits', type=float, default=FREE_BITS)
    parser.add_argument('--word-dropout', type=float, default=WORD_DROPOUT)
    parser.add_argument('--kl-warmup', type=int, default=KL_WARMUP_EPOCHS)
    parser.add_argument('--kl-collapse-threshold', type=float, default=KL_COLLAPSE_THRESHOLD)
    parser.add_argument('--kl-collapse-patience', type=int, default=KL_COLLAPSE_PATIENCE)
    parser.add_argument('--recon-cheat-threshold', type=float, default=RECON_CHEAT_THRESHOLD)
    parser.add_argument('--coh-worsen-patience', type=int, default=COH_WORSEN_PATIENCE)
    parser.add_argument('--lambda-coh', type=float, default=LAMBDA_COH)
    parser.add_argument('--lambda-tens', type=float, default=LAMBDA_TENS)
    parser.add_argument('--lambda-mov', type=float, default=LAMBDA_MOV)
    parser.add_argument('--log-dir', type=str, default=LOG_DIR)
    parser.add_argument('--checkpoint-dir', type=str, default=CHECKPOINT_DIR)
    parser.add_argument('--resume', type=str, default=None,
                        help='Path to checkpoint to resume from')
    args = parser.parse_args()

    if args.condition_dim and args.parquet is None:
        from cvae.dataset import CONDITIONED_PARQUET
        args.parquet = str(CONDITIONED_PARQUET)
        print(f'Auto-set parquet to {args.parquet} for conditioning')

    train(args)
