#!/usr/bin/env python3
"""
RVAE hyperparameter sweep.
Tests configs on 100k subset for N quick epochs each.
Reports active_units, kl_real, recon per config.
Self-adjusts: starts with defaults, tries harder reg if collapse.
"""
import subprocess
import re
import sys
import os
import json
import itertools
import shutil
import time
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
CHECKPOINT_BASE = BASE_DIR / 'checkpoints'
NUM_EPOCHS = 3
KL_WARMUP = max(1, NUM_EPOCHS - 1)

SEARCH_SPACE = {
    'word_dropout': [0.8, 0.9, 0.95],
    'per_dim_free_bits': [0.0, 0.1, 0.25],
    'latent_dim': [4, 8],
    'lr': [1e-3],
    'beta_target': [1.0],
    'free_bits': [0.0],
}

def make_config(**overrides):
    cfg = {
        'word_dropout': 0.7,
        'per_dim_free_bits': 0.25,
        'latent_dim': 8,
        'lr': 1e-3,
        'beta_target': 1.0,
        'free_bits': 0.0,
    }
    cfg.update(overrides)
    return cfg

def config_name(cfg):
    return (f"wd{cfg['word_dropout']}_fb{cfg['per_dim_free_bits']}"
            f"_ld{cfg['latent_dim']}")

def run_config(cfg, epochs=NUM_EPOCHS):
    name = config_name(cfg)
    chk_dir = CHECKPOINT_BASE / f"sweep_{name}"
    
    if chk_dir.exists():
        shutil.rmtree(chk_dir)
    
    python_bin = str(BASE_DIR / '.venv' / 'bin' / 'python')
    cmd = [
        python_bin, '-u', str(BASE_DIR / 'scripts' / 'train_rvae.py'),
        '--parquet', str(BASE_DIR / 'Dataset' / 'dataset_conditioned_100k.parquet'),
        '--epochs', str(epochs),
        '--latent-dim', str(cfg['latent_dim']),
        '--word-dropout', str(cfg['word_dropout']),
        '--per-dim-free-bits', str(cfg['per_dim_free_bits']),
        '--free-bits', str(cfg['free_bits']),
        '--lr', str(cfg['lr']),
        '--beta', str(cfg['beta_target']),
        '--kl-warmup', str(min(KL_WARMUP, epochs - 1)),
        '--kl-cycle', '0',
        '--checkpoint-dir', str(chk_dir),
    ]
    
    start = time.time()
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=1200)
    elapsed = time.time() - start
    
    if result.returncode != 0:
        stderr = result.stderr[:500] if result.stderr else ''
        stdout_tail = result.stdout[-1000:] if result.stdout else ''
        return {'config': cfg, 'name': name, 'error': stderr, 'stdout': stdout_tail, 'time': elapsed}
    
    text = result.stdout
    
    epoch_lines = [l for l in text.split('\n') if l.startswith('Epoch') and '|' in l]
    if not epoch_lines:
        return {'config': cfg, 'name': name, 'error': 'no epoch lines', 'time': elapsed}
    
    last = epoch_lines[-1]
    metrics = {'config': cfg, 'name': name, 'time': elapsed}
    
    for key in ['active', 'kl_real', 'kl']:
        m = re.search(rf'\b{key}=([\d.]+)', last)
        if m:
            metrics[key] = float(m.group(1))
    
    m_recon = re.search(r'recon=([\d.]+)', last)
    if m_recon:
        metrics['recon'] = float(m_recon.group(1))
    
    m_val = re.search(r'Val:\s*([\d.]+)', last)
    if m_val:
        metrics['val_loss'] = float(m_val.group(1))
    
    m_epoch = re.search(r'Epoch\s+(\d+)', last)
    if m_epoch:
        metrics['epoch'] = int(m_epoch.group(1))
    
    m_prior_std = re.search(r'prior_std=([\d.]+)', last)
    if m_prior_std:
        metrics['prior_std'] = float(m_prior_std.group(1))
    
    m_post_std = re.search(r'post_std=([\d.]+)', last)
    if m_post_std:
        metrics['post_std'] = float(m_post_std.group(1))
    
    return metrics


def is_good(metrics):
    """Check if config is healthy (not collapsed).
    Requires: active_units > 50%, kl_real > per_dim_free_bits * latent_dim / 2
    (so per_dim_kl > free_bits/2 on average, ensuring gradient flows)."""
    if 'error' in metrics:
        return False
    active = metrics.get('active', 0)
    kl_real = metrics.get('kl_real', 0)
    cfg = metrics.get('config', {})
    ld = cfg.get('latent_dim', 8)
    fb = cfg.get('per_dim_free_bits', 0.25)
    min_kl = fb * ld * 0.5  # at least half the dims above free_bits
    return active >= 0.5 and kl_real >= min_kl


def run_grid():
    keys = list(SEARCH_SPACE.keys())
    value_lists = [SEARCH_SPACE[k] for k in keys]
    
    results = []
    for values in itertools.product(*value_lists):
        cfg = dict(zip(keys, values))
        print(f"\n{'='*60}")
        print(f"Testing: {config_name(cfg)}")
        print(f"  word_dropout={cfg['word_dropout']}, per_dim_free_bits={cfg['per_dim_free_bits']}, latent_dim={cfg['latent_dim']}")
        print(f"{'='*60}")
        
        metrics = run_config(cfg)
        results.append(metrics)
        
        if 'error' in metrics:
            print(f"  ERROR: {metrics['error'][:100]}")
        else:
            print(f"  active={metrics.get('active', '?'):>6}  "
                  f"kl_real={metrics.get('kl_real', '?'):>8.4f}  "
                  f"kl={metrics.get('kl', '?'):>8.4f}  "
                  f"recon={metrics.get('recon', '?'):>8.4f}  "
                  f"val={metrics.get('val_loss', '?'):>8.4f}  "
                  f"time={metrics.get('time', 0):.0f}s")
    
    print(f"\n\n{'='*70}")
    print("RESULTS SORTED BY active_units desc, kl_real desc")
    print(f"{'='*70}")
    good = [r for r in results if is_good(r)]
    bad = [r for r in results if not is_good(r)]
    
    good.sort(key=lambda r: (-r.get('active', 0), -r.get('kl_real', 0)))
    
    print(f"\n--- HEALTHY ({len(good)} configs) ---")
    print(f"{'Name':<30} {'Active':<8} {'KL_real':<8} {'KL':<8} {'Recon':<8} {'Val':<8} {'Time':<6}")
    for r in good:
        print(f"{r['name']:<30} {r.get('active', 0):<8.2f} {r.get('kl_real', 0):<8.4f} {r.get('kl', 0):<8.4f} {r.get('recon', 0):<8.4f} {r.get('val_loss', 0):<8.4f} {r.get('time', 0):<6.0f}s")
    
    print(f"\n--- COLLAPSED/ERROR ({len(bad)} configs) ---")
    for r in bad:
        if 'error' in r:
            print(f"  {r['name']:<30} ERROR: {r['error'][:80]}")
        else:
            print(f"  {r['name']:<30} active={r.get('active', 0):.2f} kl_real={r.get('kl_real', 0):.4f}")
    
    out_file = CHECKPOINT_BASE / 'sweep_results.json'
    with open(out_file, 'w') as f:
        json.dump({
            'healthy': [{k: v for k, v in r.items() if k != 'config'} for r in good],
            'collapsed': [{k: v for k, v in r.items() if k != 'config'} for r in bad],
        }, f, indent=2)
    print(f"\nResults saved to {out_file}")
    
    if good:
        best = good[0]
        print(f"\nBEST CONFIG: {best['name']}")
        print(f"  active={best.get('active', 0):.2f} kl_real={best.get('kl_real', 0):.4f} recon={best.get('recon', 0):.4f}")
        print(f"  Params:")
        for k, v in best['config'].items():
            print(f"    {k}: {v}")
    else:
        print("\nNO HEALTHY CONFIG FOUND. Try increasing word_dropout or per_dim_free_bits.")


def run_adaptive(max_attempts=20):
    """Start conservative, tighten until healthy, then validate."""
    attempts = []
    
    stages = [
        {'word_dropout': wd, 'per_dim_free_bits': fb, 'latent_dim': ld}
        for fb in [0.0, 0.05, 0.1, 0.25, 0.5]
        for wd in [0.8, 0.9, 0.95]
        for ld in [8, 4]
    ]
    
    print(f"Adaptive sweep: trying up to {min(max_attempts, len(stages))} configs\n")
    
    healthy_configs = []
    
    for i, params in enumerate(stages):
        if i >= max_attempts:
            break
        
        cfg = make_config(**params)
        print(f"\n[{i+1}/{min(max_attempts, len(stages))}] {config_name(cfg)}")
        
        metrics = run_config(cfg)
        attempts.append(metrics)
        
        if 'error' in metrics:
            print(f"  ERROR: {metrics['error'][:100]}")
            continue
        
        print(f"  active={metrics.get('active', '?'):>6}  kl_real={metrics.get('kl_real', '?'):>8.4f}  recon={metrics.get('recon', '?'):>8.4f}")
        
        if is_good(metrics):
            print(f"  >>> HEALTHY, validating with {NUM_EPOCHS + 3} more epochs...")
            val_metrics = run_config(cfg, epochs=NUM_EPOCHS + 3)
            if is_good(val_metrics):
                print(f"  >>> CONFIRMED: active={val_metrics.get('active', 0):.2f} kl_real={val_metrics.get('kl_real', 0):.4f}")
                healthy_configs.append((cfg, val_metrics))
            else:
                print(f"  Validation failed: active={val_metrics.get('active', 0):.2f} kl_real={val_metrics.get('kl_real', 0):.4f}")
    
    if healthy_configs:
        # Prefer lowest per_dim_free_bits that still works
        healthy_configs.sort(key=lambda x: (x[0]['per_dim_free_bits'], -x[1].get('active', 0)))
        best_cfg, best_metrics = healthy_configs[0]
        print(f"\n{'='*60}")
        print(f"BEST CONFIG: {config_name(best_cfg)}")
        print(f"  per_dim_free_bits={best_cfg['per_dim_free_bits']}  word_dropout={best_cfg['word_dropout']}  latent_dim={best_cfg['latent_dim']}")
        print(f"  active={best_metrics.get('active', 0):.2f}  kl_real={best_metrics.get('kl_real', 0):.4f}  recon={best_metrics.get('recon', 0):.4f}")
        print(f"  {'='*60}")
        print(f"\nAll healthy configs:")
        for cfg, m in healthy_configs:
            print(f"  {config_name(cfg):<25}  active={m.get('active', 0):.2f}  kl_real={m.get('kl_real', 0):.4f}  recon={m.get('recon', 0):.4f}")
        return best_cfg, attempts
    else:
        print(f"\nNo healthy config found in {len(attempts)} attempts.")
        print("Suggestions:")
        print("  - Increase word_dropout > 0.95")
        print("  - Increase per_dim_free_bits > 1.0")
        print("  - Reduce latent_dim to 2")
        print("  - Add global free_bits")
        return None, attempts


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', choices=['grid', 'adaptive'], default='grid')
    parser.add_argument('--max-attempts', type=int, default=20)
    args = parser.parse_args()
    
    if args.mode == 'adaptive':
        run_adaptive(max_attempts=args.max_attempts)
    else:
        run_grid()
