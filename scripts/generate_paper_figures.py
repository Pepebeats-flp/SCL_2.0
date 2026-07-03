#!/usr/bin/env python3
"""Generate latent space figures for the SCL 2.0 paper."""
import sys, warnings
from pathlib import Path
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

warnings.filterwarnings('ignore')

PROJECT_ROOT = Path('/home/pepebeats/SCL_2.0')
sys.path.insert(0, str(PROJECT_ROOT))
OUTDIR = PROJECT_ROOT / 'JCC2026' / 'img'
OUTDIR.mkdir(exist_ok=True)

from cvae.models.rvae import RVAE
from cvae.dataset import ChordProgressionDataset

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
LATENT_DIM = 32
CONDITION_DIM = 16
MAX_SAMPLES = 3000
KEY_NAMES = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
C_NAMES = ['7C', 'VNSPC', 'DTMCVI', 'VDR']

print(f'Device: {DEVICE}')

# ── Load model ──
print('Loading model...')
model = RVAE(latent_dim=LATENT_DIM, condition_dim=CONDITION_DIM, z_only_decoder=True).to(DEVICE)
ckpt = torch.load(PROJECT_ROOT / 'checkpoints' / 'rvae_key_v13' / 'best.pt',
                  map_location=DEVICE, weights_only=False)
model.load_state_dict(ckpt['model_state_dict'])
model.eval()
print(f'Model loaded (epoch {ckpt.get("epoch", "?")})')

# ── Load dataset ──
print('Loading dataset...')
dataset = ChordProgressionDataset(
    PROJECT_ROOT / 'Dataset' / 'dataset_conditioned.parquet',
    use_conditioning=True,
    cond_cols=C_NAMES,
    use_key=True,
)
n_samples = min(len(dataset), MAX_SAMPLES)
indices = np.random.RandomState(42).choice(len(dataset), size=n_samples, replace=False)
print(f'Using {n_samples} samples')

# ── Encode all samples to z ──
print('Encoding samples...')
z_list = []
c_list = []
key_list = []

with torch.no_grad():
    for idx in indices:
        seq, n, cond_full = dataset[idx]  # cond_full = [7C, VNSPC, DTMCVI, VDR, key_0..11]
        c_val = cond_full[:4]  # PCS dimensions
        key_onehot = cond_full[4:]  # key one-hot
        key = key_onehot.argmax().item()
        
        seq = seq.unsqueeze(0).to(DEVICE)
        lengths = torch.tensor([n], device=DEVICE)
        mu, logvar = model.encoder(seq, lengths)
        z = model.reparameterize(mu, logvar)
        z_list.append(z[0].cpu())
        c_list.append(c_val)
        key_list.append(key)

z_all = torch.stack(z_list).numpy()
c_all = torch.stack(c_list).numpy()
keys_all = np.array(key_list)

print(f'z shape: {z_all.shape}')
print(f'c shape: {c_all.shape}')
print(f'keys: {len(keys_all)}')

# ── KL per dimension ──
print('Computing per-dim KL...')
kl_per_dim = np.zeros(LATENT_DIM)
with torch.no_grad():
    for idx in indices[:min(500, n_samples)]:
        seq, n, cond_full = dataset[idx]
        c_val = cond_full[:4]
        seq = seq.unsqueeze(0).to(DEVICE)
        lengths = torch.tensor([n], device=DEVICE)
        mu, logvar = model.encoder(seq, lengths)
        c_cond = torch.cat([c_val.unsqueeze(0).to(DEVICE),
                            torch.zeros(1, 12, device=DEVICE)], dim=-1)
        mu_prior, logvar_prior = model.prior(c_cond)
        kl = 0.5 * (logvar_prior - logvar +
                    (logvar.exp() + (mu - mu_prior).pow(2)) / logvar_prior.exp() - 1)
        kl_per_dim += kl[0].cpu().numpy()
kl_per_dim /= min(500, n_samples)

active = (kl_per_dim > 0.01).sum()
print(f'Active dims: {active}/{LATENT_DIM}')

# ── PCA ──
from sklearn.decomposition import PCA
print('Computing PCA...')
pca = PCA(n_components=2)
z_pca = pca.fit_transform(z_all)
print(f'PCA explained variance: {pca.explained_variance_ratio_}')

# ── Correlations z vs C ──
print('Computing correlations z vs C...')
c_corr = np.zeros((LATENT_DIM, 4))
for d in range(4):
    for l in range(LATENT_DIM):
        c_corr[l, d] = np.corrcoef(z_all[:, l], c_all[:, d])[0, 1]

# ══════════════════════════════════════════════
# FIGURE 1: PCA colored by Complexity and Key
# ══════════════════════════════════════════════
print('Generating Figure 1: PCA colored by C and Key...')
fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

sc = axes[0].scatter(z_pca[:, 0], z_pca[:, 1],
                     c=c_all.mean(axis=1), cmap='viridis', s=2, alpha=0.5)
axes[0].set_title('PCA colored by mean PCS', fontsize=10)
axes[0].set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%})')
axes[0].set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%})')
cbar = plt.colorbar(sc, ax=axes[0])
cbar.set_label('PCS', fontsize=9)

sc2 = axes[1].scatter(z_pca[:, 0], z_pca[:, 1],
                      c=keys_all, cmap='tab10', s=2, alpha=0.5)
axes[1].set_title('PCA colored by Key', fontsize=10)
axes[1].set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%})')
axes[1].set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%})')
cbar2 = plt.colorbar(sc2, ax=axes[1], ticks=range(12))
cbar2.ax.set_yticklabels(KEY_NAMES, fontsize=7)
cbar2.set_label('Key', fontsize=9)

plt.tight_layout()
plt.savefig(OUTDIR / 'pca_colored.pdf', dpi=150, bbox_inches='tight')
plt.close()
print('  → img/pca_colored.pdf')

# ══════════════════════════════════════════════
# FIGURE 2: Per-dimension z correlation with C
# ══════════════════════════════════════════════
print('Generating Figure 2: z-C correlations...')
fig, axes = plt.subplots(1, 4, figsize=(14, 3.2))

for d in range(4):
    ax = axes[d]
    bars = ax.bar(range(LATENT_DIM), c_corr[:, d], width=0.7, alpha=0.85)
    # Color top 3 positively and negatively
    sorted_idx = np.argsort(np.abs(c_corr[:, d]))[::-1]
    for i in sorted_idx[:3]:
        bars[i].set_color('darkred' if c_corr[i, d] < 0 else 'darkgreen')
    ax.set_title(f'Corr(z$_\\ell$, {C_NAMES[d]})', fontsize=9)
    ax.set_xlabel('Latent dimension $\\ell$', fontsize=8)
    ax.axhline(0.3, color='gray', linestyle='--', alpha=0.4, linewidth=0.8)
    ax.axhline(-0.3, color='gray', linestyle='--', alpha=0.4, linewidth=0.8)
    ax.set_ylim(-1.05, 1.05)
    ax.set_xticks([0, 8, 16, 24, 31])
    ax.tick_params(labelsize=7)

plt.suptitle('Latent Dimension Correlation with Perceptual Complexity Dimensions',
             fontsize=11, y=1.02)
plt.tight_layout()
plt.savefig(OUTDIR / 'corr_z_c.pdf', dpi=150, bbox_inches='tight')
plt.close()
print('  → img/corr_z_c.pdf')

# ══════════════════════════════════════════════
# FIGURE 3: Per-dimension KL + z std
# ══════════════════════════════════════════════
print('Generating Figure 3: Per-dim KL and std...')
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 3.5))

ax1.bar(range(LATENT_DIM), kl_per_dim, width=0.7, alpha=0.85)
ax1.axhline(0.01, color='red', linestyle='--', alpha=0.5, linewidth=0.8)
ax1.axhline(0.05, color='orange', linestyle='--', alpha=0.5, linewidth=0.8)
ax1.set_xlabel('Latent dimension $\\ell$')
ax1.set_ylabel('KL (nats)')
ax1.set_title(f'Per-dim KL — {active}/{LATENT_DIM} active (>0.01)', fontsize=10)
ax1.tick_params(labelsize=7)

z_std = z_all.std(axis=0)
ax2.bar(range(LATENT_DIM), z_std, width=0.7, alpha=0.85)
ax2.axhline(1.0, color='gray', linestyle='--', alpha=0.5, linewidth=0.8)
ax2.set_xlabel('Latent dimension $\\ell$')
ax2.set_ylabel('Std')
ax2.set_title('z standard deviation per dimension', fontsize=10)
ax2.tick_params(labelsize=7)

plt.tight_layout()
plt.savefig(OUTDIR / 'per_dim_kl.pdf', dpi=150, bbox_inches='tight')
plt.close()
print('  → img/per_dim_kl.pdf')

# ── Print summary for paper ──
print(f'\n═══ Paper-ready numbers ═══')
print(f'Active dims: {active}/{LATENT_DIM}')
print(f'PCA 2PC var: {pca.explained_variance_ratio_.sum():.1%}')
print(f'PCA PC1: {pca.explained_variance_ratio_[0]:.1%}, PC2: {pca.explained_variance_ratio_[1]:.1%}')
print(f'Top C correlations:')
for d in range(4):
    top3 = sorted([(l, c_corr[l,d]) for l in range(LATENT_DIM)],
                   key=lambda x: abs(x[1]), reverse=True)[:3]
    print(f'  {C_NAMES[d]}: {[(f"z{l}", round(v,3)) for l,v in top3]}')

print(f'\nDone! Figures saved to {OUTDIR}/')
