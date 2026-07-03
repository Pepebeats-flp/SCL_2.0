import argparse
import os
import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.amp import autocast, GradScaler
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from cvae.config import (
    CHORD_DIM, BATCH_SIZE, LEARNING_RATE,
    MAX_EPOCHS, PATIENCE, GRAD_CLIP, LOG_DIR, CHECKPOINT_DIR,
    BETA, FREE_BITS, LAMBDA_COH, LAMBDA_TENS, LAMBDA_MOV,
    WORD_DROPOUT, KL_WARMUP_EPOCHS,
    SCHEDULED_SAMPLING_START, SCHEDULED_SAMPLING_END, SCHEDULED_SAMPLING_EPOCHS,
    KL_COLLAPSE_THRESHOLD, KL_COLLAPSE_PATIENCE,
    RECON_CHEAT_THRESHOLD, COH_WORSEN_PATIENCE,
    ACTIVE_UNITS_THRESHOLD, KL_REAL_THRESHOLD,
    LATENT_DIM,
)
from cvae.dataset import ChordProgressionDataset, PERCEPTUAL_COLS, split_indices, create_dataloader
from cvae.models.rvae import RVAE, rvae_kl_loss, per_dim_kl_rvae
from cvae.losses import cvae_loss


def get_beta(epoch, beta_target, kl_warmup, kl_cycle):
    if kl_cycle and kl_cycle > 0:
        pos = (epoch - 1) % kl_cycle
        beta_effective = min(1.0, pos / max(kl_cycle - 1, 1)) * beta_target
    else:
        beta_effective = min(1.0, (epoch - 1) / max(kl_warmup, 1)) * beta_target
    return beta_effective


def get_teacher_forcing_prob(epoch, start, end, total_epochs):
    if epoch > total_epochs:
        return end
    frac = (epoch - 1) / max(total_epochs - 1, 1)
    return start + (end - start) * frac


def check_quality_guards(epoch, args, kl_history, recon_history, coh_history, latent_stats_history):
    """Returns (stop, reason, hint) tuple."""
    if epoch <= 1:
        return False, '', ''

    if not (args.kl_cycle and args.kl_cycle > 0) and len(kl_history) >= args.kl_collapse_patience and args.per_dim_free_bits <= 0:
        recent = kl_history[-args.kl_collapse_patience:]
        if all(k < args.kl_collapse_threshold for k in recent):
            return True, (
                f'KL collapse: KL < {args.kl_collapse_threshold} '
                f'for {args.kl_collapse_patience} epochs after warmup'
            ), 'Increase word_dropout, reduce lr, or increase free_bits'

    if len(kl_history) >= 2:
        last_kl = kl_history[-1]
        last_recon = recon_history[-1]
        if last_kl < 0.01 and last_recon < 0.5:
            if last_recon < 0.1:
                return True, (
                    f'Model cheating: recon={last_recon:.4f} with KL~0'
                ), 'Reduce lr, increase beta_target, or increase word_dropout'

    if len(latent_stats_history) >= 1:
        last = latent_stats_history[-1]
        if last.get('active_units', 1) < args.active_units_threshold:
            return True, (
                f'Active units collapse: {last["active_units"]:.2%} '
                f'< {args.active_units_threshold:.0%}'
            ), 'Reduce decoder capacity, increase beta, or adjust free_bits'

    if len(latent_stats_history) >= 1:
        last = latent_stats_history[-1]
        if last.get('kl_real', 1) < args.kl_real_threshold:
            return True, (
                f'KL real collapse: {last["kl_real"]:.4f} '
                f'< {args.kl_real_threshold}'
            ), 'Scheduled sampling rate too high, reduce TF prob faster'

    if len(coh_history) >= args.coh_worsen_patience + 1:
        recent = coh_history[-(args.coh_worsen_patience + 1):]
        if all(recent[i + 1] > recent[i] for i in range(len(recent) - 1)):
            return True, (
                f'Coherence loss worsening for {args.coh_worsen_patience} '
                f'consecutive epochs'
            ), 'Increase lambda_coh or check tonal consistency'

    return False, '', ''


def compute_latent_stats_rvae(mu, logvar, mu_prior, logvar_prior):
    with torch.no_grad():
        mu_var = mu.var(dim=0)
        active_units = (mu_var > 0.01).float().mean().item()
        kl_per_dim = per_dim_kl_rvae(mu, logvar, mu_prior, logvar_prior)
        kl_real = kl_per_dim.sum(dim=-1).mean().item()
        kl_pd = kl_per_dim.mean(dim=0)
        n_active_dims = (kl_pd > 0.01).float().mean().item()

        prior_std = (0.5 * logvar_prior).exp().mean().item()
        posterior_std = (0.5 * logvar).exp().mean().item()

        return {
            'active_units': active_units,
            'mu_mean': mu.mean().item(),
            'mu_std': mu.std().item(),
            'logvar_mean': logvar.mean().item(),
            'logvar_std': logvar.std().item(),
            'prior_logvar_mean': logvar_prior.mean().item(),
            'prior_std': prior_std,
            'posterior_std': posterior_std,
            'kl_real': kl_real,
            'mu_var_mean': mu_var.mean().item(),
            'n_active_dims': n_active_dims,
            'kl_per_dim_mean': kl_pd.mean().item(),
            'kl_per_dim_max': kl_pd.max().item(),
            'kl_per_dim_min': kl_pd.min().item(),
        }


def train(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Device: {device}')
    print(f'Complexity dimensions: {args.cond_cols} ({args.condition_dim})')

    use_key = args.condition_dim > 4
    dataset = ChordProgressionDataset(
        args.parquet, max_len=args.max_len,
        use_conditioning=True, cond_cols=args.cond_cols,
        use_key=use_key,
    )
    indices = split_indices(len(dataset))

    train_loader = create_dataloader(
        dataset=dataset, indices=indices,
        batch_size=args.batch_size, shuffle=True,
        split='train', num_workers=args.num_workers,
        use_key=use_key,
    )
    val_loader = create_dataloader(
        dataset=dataset, indices=indices,
        batch_size=args.batch_size, shuffle=False,
        split='val', num_workers=args.num_workers,
        use_key=use_key,
    )

    model = RVAE(
        latent_dim=args.latent_dim, condition_dim=args.condition_dim,
        word_dropout=args.word_dropout,
        z_only_decoder=args.z_only_decoder,
        decoder_hidden_dim=args.decoder_hidden_dim,
        decoder_num_layers=args.decoder_num_layers,
    ).to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scaler = GradScaler('cuda', enabled=args.amp)

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

    active_music_losses = []
    if args.lambda_coh > 0:
        active_music_losses.append(f'coherence (λ={args.lambda_coh})')
    if args.lambda_tens > 0:
        active_music_losses.append(f'tension (λ={args.lambda_tens})')
    if args.lambda_mov > 0:
        active_music_losses.append(f'movement (λ={args.lambda_mov})')
    if active_music_losses:
        print(f'Active music losses: {", ".join(active_music_losses)}')
    else:
        print('Music losses: none (only recon + KL)')

    writer = SummaryWriter(args.log_dir)
    os.makedirs(args.checkpoint_dir, exist_ok=True)

    patience_counter = 0
    kl_history = []
    recon_history = []
    coh_history = []
    latent_stats_history = []

    for epoch in range(start_epoch, args.epochs + 1):
        beta_effective = get_beta(epoch, args.beta, args.kl_warmup, args.kl_cycle)
        tf_prob = get_teacher_forcing_prob(
            epoch, args.tf_start, args.tf_end, args.tf_epochs
        )

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
            x, lengths, c = batch
            x = x.to(device, non_blocking=True)
            lengths = lengths.to(device, non_blocking=True)
            c = c.to(device, non_blocking=True)

            optimizer.zero_grad()

            with autocast(device_type=device.type, enabled=args.amp):
                logits, mu, logvar, mu_prior, logvar_prior, z, c_pred, key_logits = model(
                    x, lengths, c=c, teacher_forcing_prob=tf_prob,
                )

                recon_loss = F.binary_cross_entropy_with_logits(
                    logits, x, reduction='none'
                )
                mask = torch.arange(x.size(1), device=lengths.device).unsqueeze(0) < lengths.unsqueeze(1)
                recon_loss = (recon_loss * mask.unsqueeze(-1)).sum() / mask.sum()

                kl_loss, prior_loss = rvae_kl_loss(mu, logvar, mu_prior, logvar_prior, free_bits=args.free_bits, per_dim_free_bits=args.per_dim_free_bits, prior_mse_weight=args.prior_mse_weight)

                c_pred_loss = F.mse_loss(c_pred, c[:, :4]) if args.lambda_c_pred > 0 else torch.tensor(0.0)

                key_loss = F.cross_entropy(key_logits, c[:, 4:].argmax(dim=-1)) if (args.lambda_key_pred > 0 and key_logits is not None) else torch.tensor(0.0)

                total = recon_loss + beta_effective * kl_loss + args.lambda_c_pred * c_pred_loss + args.lambda_key_pred * key_loss

            if args.lambda_coh > 0 or args.lambda_tens > 0 or args.lambda_mov > 0:
                from cvae.losses import coherence_loss, tension_loss, movement_loss
                l_coh = coherence_loss(logits, lengths, x) if args.lambda_coh > 0 else torch.tensor(0.0)
                l_tens = tension_loss(logits, lengths) if args.lambda_tens > 0 else torch.tensor(0.0)
                l_mov = movement_loss(logits, lengths) if args.lambda_mov > 0 else torch.tensor(0.0)
                total = total + args.lambda_coh * l_coh + args.lambda_tens * l_tens + args.lambda_mov * l_mov
            else:
                l_coh = torch.tensor(0.0)
                l_tens = torch.tensor(0.0)
                l_mov = torch.tensor(0.0)

            scaler.scale(total).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            scaler.step(optimizer)
            scaler.update()

            b = x.size(0)
            train_loss += total.item() * b
            train_recon += recon_loss.item() * b
            train_kl += kl_loss.item() * b
            train_coh += l_coh.item() * b
            train_tens += l_tens.item() * b
            train_mov += l_mov.item() * b
            n_train += b

            pbar.set_postfix({
                'loss': f'{total.item():.4f}',
                'recon': f'{recon_loss.item():.4f}',
                'kl': f'{kl_loss.item():.4f}',
                'tf': f'{tf_prob:.2f}',
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
        all_mu = []
        all_logvar = []
        all_mu_prior = []
        all_logvar_prior = []

        with torch.no_grad():
            for batch in tqdm(val_loader, desc=f'Epoch {epoch}/{args.epochs} [Val]'):
                x, lengths, c = batch
                x = x.to(device)
                lengths = lengths.to(device)
                c = c.to(device)

                logits, mu, logvar, mu_prior, logvar_prior, z, c_pred, key_logits = model(
                    x, lengths, c=c, teacher_forcing_prob=1.0,
                )
                all_mu.append(mu.detach().cpu())
                all_logvar.append(logvar.detach().cpu())
                all_mu_prior.append(mu_prior.detach().cpu())
                all_logvar_prior.append(logvar_prior.detach().cpu())

                recon_loss = F.binary_cross_entropy_with_logits(logits, x, reduction='none')
                mask = torch.arange(x.size(1), device=lengths.device).unsqueeze(0) < lengths.unsqueeze(1)
                recon_loss = (recon_loss * mask.unsqueeze(-1)).sum() / mask.sum()

                kl_loss, _ = rvae_kl_loss(mu, logvar, mu_prior, logvar_prior, free_bits=args.free_bits, per_dim_free_bits=args.per_dim_free_bits, prior_mse_weight=args.prior_mse_weight)
                total = recon_loss + beta_effective * kl_loss

                b = x.size(0)
                val_loss += total.item() * b
                val_recon += recon_loss.item() * b
                val_kl += kl_loss.item() * b
                n_val += b

        val_loss /= n_val
        val_recon /= n_val
        val_kl /= n_val

        if all_mu:
            all_mu = torch.cat(all_mu, dim=0)
            all_logvar = torch.cat(all_logvar, dim=0)
            all_mu_prior = torch.cat(all_mu_prior, dim=0)
            all_logvar_prior = torch.cat(all_logvar_prior, dim=0)
            latent_stats = compute_latent_stats_rvae(all_mu, all_logvar, all_mu_prior, all_logvar_prior)
        else:
            latent_stats = {}

        kl_history.append(train_kl)
        recon_history.append(train_recon)
        coh_history.append(train_coh)
        latent_stats_history.append(latent_stats)

        latent_str = ''
        if latent_stats:
            latent_str = (f' | active={latent_stats["active_units"]:.2f} '
                          f'kl_real={latent_stats["kl_real"]:.3f} '
                          f'prior_std={latent_stats["prior_std"]:.3f} '
                          f'post_std={latent_stats["posterior_std"]:.3f} '
                          f'logvar={latent_stats["logvar_mean"]:.1f}')
        music_str = ''
        if args.lambda_coh > 0:
            music_str += f' coh={train_coh:.4f}'
        if args.lambda_tens > 0:
            music_str += f' tens={train_tens:.4f}'
        if args.lambda_mov > 0:
            music_str += f' mov={train_mov:.4f}'
        print(f'Epoch {epoch:3d} | '
              f'Train: {train_loss:.4f} (recon={train_recon:.4f}, kl={train_kl:.4f}{music_str}) | '
              f'Val: {val_loss:.4f} (recon={val_recon:.4f}, kl={val_kl:.4f}) | '
              f'β={beta_effective:.3f} tf={tf_prob:.2f}{latent_str}')

        writer.add_scalars('loss/total', {'train': train_loss, 'val': val_loss}, epoch)
        writer.add_scalars('loss/recon', {'train': train_recon, 'val': val_recon}, epoch)
        writer.add_scalars('loss/kl', {'train': train_kl, 'val': val_kl}, epoch)
        writer.add_scalar('loss/coh', train_coh, epoch)
        writer.add_scalar('loss/tens', train_tens, epoch)
        writer.add_scalar('loss/mov', train_mov, epoch)
        writer.add_scalar('train/tf_prob', tf_prob, epoch)
        writer.add_scalar('train/beta', beta_effective, epoch)
        if latent_stats:
            writer.add_scalar('latent/active_units', latent_stats['active_units'], epoch)
            writer.add_scalar('latent/kl_real', latent_stats['kl_real'], epoch)
            writer.add_scalar('latent/mu_mean', latent_stats['mu_mean'], epoch)
            writer.add_scalar('latent/mu_std', latent_stats['mu_std'], epoch)
            writer.add_scalar('latent/logvar_mean', latent_stats['logvar_mean'], epoch)
            writer.add_scalar('latent/prior_std', latent_stats['prior_std'], epoch)
            writer.add_scalar('latent/posterior_std', latent_stats['posterior_std'], epoch)

        should_stop, stop_reason, hint = check_quality_guards(
            epoch, args, kl_history, recon_history, coh_history, latent_stats_history,
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
    parser.add_argument('--parquet', type=str, default=None)
    parser.add_argument('--batch-size', type=int, default=BATCH_SIZE)
    parser.add_argument('--max-len', type=int, default=256)
    parser.add_argument('--latent-dim', type=int, default=LATENT_DIM)
    parser.add_argument('--condition-dim', type=int, default=4)
    parser.add_argument('--cond-cols', type=str, nargs='+', default=PERCEPTUAL_COLS,
                        help='Conditioning column names')
    parser.add_argument('--num-workers', type=int, default=0)
    parser.add_argument('--lr', type=float, default=LEARNING_RATE)
    parser.add_argument('--epochs', type=int, default=MAX_EPOCHS)
    parser.add_argument('--patience', type=int, default=PATIENCE)
    parser.add_argument('--grad-clip', type=float, default=GRAD_CLIP)
    parser.add_argument('--beta', type=float, default=BETA)
    parser.add_argument('--free-bits', type=float, default=FREE_BITS)
    parser.add_argument('--per-dim-free-bits', type=float, default=0.25)
    parser.add_argument('--prior-mse-weight', type=float, default=0.0)
    parser.add_argument('--lambda-c-pred', type=float, default=0.0)
    parser.add_argument('--lambda-key-pred', type=float, default=0.0)
    parser.add_argument('--word-dropout', type=float, default=WORD_DROPOUT)
    parser.add_argument('--z-only-decoder', action='store_true', default=False,
                        help='Decoder receives only z (condition info must flow through z)')
    parser.add_argument('--kl-warmup', type=int, default=KL_WARMUP_EPOCHS)
    parser.add_argument('--kl-cycle', type=int, default=10,
                        help='Cyclical KL annealing cycle length (0 = disabled)')
    parser.add_argument('--tf-start', type=float, default=1.0)
    parser.add_argument('--tf-end', type=float, default=0.3)
    parser.add_argument('--tf-epochs', type=int, default=15)
    parser.add_argument('--active-units-threshold', type=float, default=ACTIVE_UNITS_THRESHOLD)
    parser.add_argument('--kl-real-threshold', type=float, default=0.01)
    parser.add_argument('--kl-collapse-threshold', type=float, default=KL_COLLAPSE_THRESHOLD)
    parser.add_argument('--kl-collapse-patience', type=int, default=KL_COLLAPSE_PATIENCE)
    parser.add_argument('--recon-cheat-threshold', type=float, default=RECON_CHEAT_THRESHOLD)
    parser.add_argument('--coh-worsen-patience', type=int, default=COH_WORSEN_PATIENCE)
    parser.add_argument('--decoder-hidden-dim', type=int, default=None,
                        help='Decoder LSTM hidden dim (default: config.DECODER_HIDDEN_DIM)')
    parser.add_argument('--decoder-num-layers', type=int, default=None,
                        help='Decoder LSTM num layers (default: config.NUM_LAYERS)')
    parser.add_argument('--lambda-coh', type=float, default=0.0)
    parser.add_argument('--lambda-tens', type=float, default=0.0)
    parser.add_argument('--lambda-mov', type=float, default=0.0)
    parser.add_argument('--log-dir', type=str, default='runs/rvae')
    parser.add_argument('--checkpoint-dir', type=str, default='checkpoints/rvae')
    parser.add_argument('--resume', type=str, default=None)
    parser.add_argument('--amp', action='store_true', help='Enable AMP (mixed precision)')
    args = parser.parse_args()

    if args.parquet is None:
        from cvae.dataset import CONDITIONED_PARQUET
        args.parquet = str(CONDITIONED_PARQUET)
        print(f'Auto-set parquet to {args.parquet}')

    train(args)
