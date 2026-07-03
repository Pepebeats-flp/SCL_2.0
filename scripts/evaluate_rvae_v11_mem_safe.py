import gc, json, sys, warnings, math
warnings.filterwarnings('ignore')
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from scipy.stats import pearsonr

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from chords.chord_encoder import decode_chord, encode_chord, parse_chord
from chords.vocab import CHORD_DIM
from cvae.dataset import ChordProgressionDataset, PERCEPTUAL_COLS, split_indices
from cvae.models.rvae import RVAE, per_dim_kl_rvae

sns.set_theme(style='whitegrid')

torch.set_num_threads(2)

LATENT_DIM = 32
CONDITION_DIM = 16
CHECKPOINT_PATH = '/home/pepebeats/SCL_2.0/checkpoints/rvae_key_v11/best.pt'
N_SAMPLES = 500
MAX_LEN = 128
KEY_NAMES = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
PARQUET_PATH = '/home/pepebeats/SCL_2.0/Dataset/dataset_conditioned_100k.parquet'
PLOT_DIR = Path('v11_plots_mem_safe')
PLOT_DIR.mkdir(exist_ok=True)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Device: {device}')

print('Loading model...')
model = RVAE(latent_dim=LATENT_DIM, condition_dim=CONDITION_DIM, z_only_decoder=True).to(device)
ckpt = torch.load(CHECKPOINT_PATH, map_location=device, weights_only=False)
model.load_state_dict(ckpt['model_state_dict'])
model.eval()
epoch = ckpt.get('epoch', '?')
val_loss = ckpt.get('val_loss', None)
print(f'Loaded epoch {epoch}, val_loss={val_loss:.4f}' if val_loss else f'Loaded epoch {epoch}')
n_params = sum(p.numel() for p in model.parameters())
print(f'Params: {n_params:,}')

print('\nLoading dataset (100k subset)...')
dataset = ChordProgressionDataset(
    parquet_path=PARQUET_PATH, max_len=MAX_LEN,
    use_conditioning=True, cond_cols=PERCEPTUAL_COLS, use_key=True
)
indices = split_indices(len(dataset), test_size=500)
test_idx = indices['test']
print(f'Test set: {len(test_idx)} progressions')

# ---- Encode test set ----
@torch.no_grad()
def encode_progressions(indices, max_n=500):
    model.eval()
    all_mu, all_z, all_c = [], [], []
    all_keys, all_seqs, all_lens = [], [], []
    for i in indices[:max_n]:
        seq, n, c = dataset[i]
        seq = seq.unsqueeze(0).to(device)
        c = c.unsqueeze(0).to(device)
        lengths = torch.tensor([n], device=device)
        mu, logvar = model.encode(seq, lengths)
        z = model.reparameterize(mu, logvar)
        key_idx = c[0, 4:].argmax().item()
        all_mu.append(mu.cpu())
        all_z.append(z.cpu())
        all_c.append(c.cpu())
        all_keys.append(key_idx)
        all_seqs.append(seq.cpu())
        all_lens.append(n)
    return (
        torch.cat(all_mu, dim=0),
        torch.cat(all_z, dim=0),
        torch.cat(all_c, dim=0),
        all_keys, all_seqs, all_lens
    )

print('Encoding test progressions...')
mu_all, z_all, c_all, keys_all, seqs_all, lens_all = encode_progressions(test_idx, max_n=N_SAMPLES)
c_perc = c_all[:, :4]
print(f'Encoded {len(mu_all)} progressions')
gc.collect()

# ---- 1. DECODER BYPASS TEST ----
print('\n=== 1. DECODER BYPASS TEST ===')
model.eval()
recon_losses_real, recon_losses_zero = [], []
with torch.no_grad():
    for i in range(min(50, len(test_idx))):
        seq, n, c = dataset[i]
        seq = seq.unsqueeze(0).to(device); c = c.unsqueeze(0).to(device)
        lengths = torch.tensor([n], device=device)
        mu, logvar = model.encode(seq, lengths)
        z = model.reparameterize(mu, logvar)
        z_dec = z if model.z_only_decoder else model._z_dec(z, c)
        logits = model.decoder(z_dec, seq, lengths, teacher_forcing_prob=0.0)
        recon_losses_real.append(torch.nn.functional.binary_cross_entropy_with_logits(
            logits[:, :n], seq[:, :n]).item())

        z_zero = torch.zeros_like(z)
        zd = z_zero if model.z_only_decoder else model._z_dec(z_zero, c)
        logits0 = model.decoder(zd, seq, lengths, teacher_forcing_prob=0.0)
        recon_losses_zero.append(torch.nn.functional.binary_cross_entropy_with_logits(
            logits0[:, :n], seq[:, :n]).item())

avg_real = np.mean(recon_losses_real)
avg_zero = np.mean(recon_losses_zero)
delta = avg_zero - avg_real
print(f'recon(z_real) @ tf=0: {avg_real:.4f} BCE')
print(f'recon(z=0) @ tf=0:    {avg_zero:.4f} BCE')
print(f'Δ (zero - real):       {delta:.4f}')
if delta > 0.15:
    print('✅ Decoder SÍ usa z (Δ > 0.15)')
elif delta > 0.05:
    print('⚠️  Decoder usa z débilmente (Δ > 0.05)')
else:
    print('❌ Decoder ignora z (Δ ≈ 0)')
del recon_losses_real, recon_losses_zero; gc.collect()

# ---- 2. LATENT SPACE SUMMARY ----
print('\n=== 2. LATENT SPACE SUMMARY ===')
kl_per_dim = per_dim_kl_rvae(mu_all, torch.zeros_like(mu_all),
                             torch.zeros_like(mu_all), torch.zeros_like(mu_all))
kl_per_dim = kl_per_dim.mean(dim=0)
total_kl = kl_per_dim.sum().item()
active_dims = (kl_per_dim > 0.01).sum().item()
z_std = z_all.std(dim=0).mean().item()
mu_std = mu_all.std(dim=0).mean().item()
logvar_val = (z_all.var(dim=0).log()).mean().item()

print(f'Per-dim KL range: {kl_per_dim.min().item():.4f} - {kl_per_dim.max().item():.4f}')
print(f'Total KL:         {total_kl:.4f} nats (over {LATENT_DIM} dims)')
print(f'Active dims (>0.01): {active_dims}/{LATENT_DIM}')
print(f'z mean std:       {z_std:.4f}')
print(f'mu mean std:      {mu_std:.4f}')
print(f'avg logvar:       {logvar_val:.4f}')

pca = PCA().fit(z_all.numpy())
cumsum = np.cumsum(pca.explained_variance_ratio_)
print(f'PCA 2 PCs: {cumsum[1]*100:.1f}%')
print(f'PCA 5 PCs: {cumsum[4]*100:.1f}%')
print(f'PCA 10 PCs: {cumsum[9]*100:.1f}%')

print('\nEvaluating predictors...')
with torch.no_grad():
    c_pred = model.c_predictor(z_all.to(device))
    c_mse = torch.nn.functional.mse_loss(c_pred.cpu(), c_perc).item()
    key_logits = model.key_predictor(z_all.to(device))
    key_acc = (key_logits.argmax(dim=-1).cpu() == torch.tensor(keys_all)).float().mean().item()
print(f'c_predictor MSE: {c_mse:.4f}')
print(f'key_predictor acc: {key_acc*100:.1f}%')
gc.collect()

# ---- 3. PCA Plots ----
print('\n=== 3. PCA PLOTS ===')
z_np = z_all.numpy()
pca2 = PCA(n_components=2)
z_pca = pca2.fit_transform(z_np)

fig, axes = plt.subplots(1, 2, figsize=(16, 6))
sc = axes[0].scatter(z_pca[:, 0], z_pca[:, 1], c=c_perc.mean(dim=1), cmap='viridis', s=3, alpha=0.6)
axes[0].set_title('PCA colored by mean complexity (C)')
plt.colorbar(sc, ax=axes[0])
sc2 = axes[1].scatter(z_pca[:, 0], z_pca[:, 1], c=keys_all, cmap='tab10', s=3, alpha=0.6)
axes[1].set_title('PCA colored by Key')
plt.colorbar(sc2, ax=axes[1], ticks=range(12), label='Key')
plt.tight_layout()
plt.savefig(PLOT_DIR / '01_pca_colored.png', dpi=150, bbox_inches='tight')
plt.close('all')
print('Saved 01_pca_colored.png')

# ---- 4. PCA Perceptual Dims ----
print('\n=== 4. PCA PERCEPTUAL DIMS ===')
dim_names = ['7C', 'VNSPC', 'DTMCVI', 'VDR']
fig, axes = plt.subplots(1, 4, figsize=(20, 4.5))
for d, ax in enumerate(axes):
    sc = ax.scatter(z_pca[:, 0], z_pca[:, 1], c=c_perc[:, d],
                    cmap='viridis', s=3, alpha=0.6, vmin=0, vmax=1)
    ax.set_title(f'PCA colored by {dim_names[d]}')
    plt.colorbar(sc, ax=ax, label=dim_names[d])
plt.tight_layout()
plt.savefig(PLOT_DIR / '02_pca_perceptual_dims.png', dpi=150, bbox_inches='tight')
plt.close('all')
print('Saved 02_pca_perceptual_dims.png')

# ---- 5. PCA Key Separation ----
print('\n=== 5. PCA KEY SEPARATION ===')
fig, axes = plt.subplots(3, 4, figsize=(16, 12))
for idx, (ax, key_name) in enumerate(zip(axes.flatten(), KEY_NAMES)):
    mask = [k == idx for k in keys_all]
    ax.scatter(z_pca[:, 0], z_pca[:, 1], c='lightgray', alpha=0.15, s=2)
    if sum(mask) > 5:
        ax.scatter(z_pca[:, 0][mask], z_pca[:, 1][mask], c='red', alpha=0.6, s=10, label=key_name)
    ax.set_title(f'Key = {key_name} (n={sum(mask)})')
    ax.legend(fontsize=8)
plt.tight_layout()
plt.savefig(PLOT_DIR / '03_pca_key_separate.png', dpi=150, bbox_inches='tight')
plt.close('all')
print('Saved 03_pca_key_separate.png')

# ---- 6. t-SNE (reduced samples) ----
N_TSNE = 300
print(f'\n=== 6. T-SNE ({N_TSNE} samples) ===')
tsne = TSNE(n_components=2, perplexity=30, max_iter=800, random_state=42)
z_tsne = tsne.fit_transform(z_np[:N_TSNE])
print('t-SNE done')

fig, axes = plt.subplots(1, 2, figsize=(16, 6))
sc = axes[0].scatter(z_tsne[:, 0], z_tsne[:, 1],
                     c=c_perc[:N_TSNE].mean(dim=1), cmap='viridis', s=6, alpha=0.6)
axes[0].set_title('t-SNE colored by mean complexity (C)')
plt.colorbar(sc, ax=axes[0])
sc2 = axes[1].scatter(z_tsne[:, 0], z_tsne[:, 1],
                      c=[keys_all[i] for i in range(N_TSNE)], cmap='tab10', s=6, alpha=0.6)
axes[1].set_title('t-SNE colored by Key')
plt.colorbar(sc2, ax=axes[1], ticks=range(12), label='Key')
plt.tight_layout()
plt.savefig(PLOT_DIR / '04_tsne_colored.png', dpi=150, bbox_inches='tight')
plt.close('all')
print('Saved 04_tsne_colored.png')

del z_tsne; gc.collect()

# ---- 7. Per-dim KL and Std ----
print('\n=== 7. PER-DIM KL AND STD ===')
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
kl_np = kl_per_dim.numpy()
axes[0].bar(range(LATENT_DIM), kl_np)
axes[0].axhline(0.01, color='r', linestyle='--', alpha=0.5, label='active threshold')
axes[0].axhline(0.15, color='orange', linestyle='--', alpha=0.5, label='free bits (0.15)')
axes[0].set_xlabel('Latent dimension')
axes[0].set_ylabel('KL (nats)')
axes[0].set_title(f'Per-dim KL — {active_dims}/{LATENT_DIM} active (>0.01)')
axes[0].legend()
axes[0].set_xticks(range(0, LATENT_DIM, 4))
z_std_per_dim = z_all.std(dim=0).numpy()
axes[1].bar(range(LATENT_DIM), z_std_per_dim)
axes[1].set_xlabel('Latent dimension')
axes[1].set_ylabel('Std')
axes[1].set_title('z std per dimension')
axes[1].set_xticks(range(0, LATENT_DIM, 4))
plt.tight_layout()
plt.savefig(PLOT_DIR / '05_per_dim_kl_std.png', dpi=150, bbox_inches='tight')
plt.close('all')
print('Saved 05_per_dim_kl_std.png')

# ---- 8. Correlation Matrix ----
print('\n=== 8. CORRELATION MATRIX ===')
z_corr = np.corrcoef(z_np.T)
off_diag = np.triu(z_corr, k=1)[np.triu(np.ones_like(z_corr), k=1).astype(bool)]
print(f'Inter-z |r| mean: {np.abs(off_diag).mean():.4f}')
print(f'Inter-z |r| max:  {np.abs(off_diag).max():.4f}')
print(f'|r|>0.3: {(np.abs(off_diag)>0.3).mean()*100:.1f}%')

fig, ax = plt.subplots(figsize=(10, 8))
sns.heatmap(z_corr, cmap='RdBu_r', vmin=-1, vmax=1, center=0, ax=ax)
ax.set_title('z correlation matrix')
plt.tight_layout()
plt.savefig(PLOT_DIR / '06_correlation_matrix.png', dpi=150, bbox_inches='tight')
plt.close('all')
print('Saved 06_correlation_matrix.png')

# ---- 9. Correlation with C ----
print('\n=== 9. C CORRELATION ===')
c_names = ['7C', 'VNSPC', 'DTMCVI', 'VDR']
c_corr = np.zeros((LATENT_DIM, 4))
for d in range(4):
    for l in range(LATENT_DIM):
        c_corr[l, d] = np.corrcoef(z_np[:, l], c_perc[:, d].numpy())[0, 1]

fig, axes = plt.subplots(1, 4, figsize=(16, 4))
for d in range(4):
    axes[d].bar(range(LATENT_DIM), c_corr[:, d])
    axes[d].set_title(f'Corr(z_dim, {c_names[d]})')
    axes[d].set_xlabel('Latent dim')
    axes[d].axhline(0.3, color='r', linestyle='--', alpha=0.4)
    axes[d].axhline(-0.3, color='r', linestyle='--', alpha=0.4)
    axes[d].set_xticks(range(0, LATENT_DIM, 4))
plt.tight_layout()
plt.savefig(PLOT_DIR / '07_c_correlation.png', dpi=150, bbox_inches='tight')
plt.close('all')
print('Saved 07_c_correlation.png')

print('Top correlations per C-dim:')
for d in range(4):
    top = sorted([(l, c_corr[l,d]) for l in range(LATENT_DIM)],
                 key=lambda x: abs(x[1]), reverse=True)[:5]
    print(f'  {c_names[d]}: {[(f"z{l}", round(v,3)) for l,v in top]}')

# ---- 10. z Norm Distribution ----
print('\n=== 10. Z NORM DISTRIBUTION ===')
z_norm = np.linalg.norm(z_np, axis=1)
fig, ax = plt.subplots(figsize=(8, 4))
ax.hist(z_norm, bins=50, alpha=0.7, edgecolor='black')
ax.axvline(z_norm.mean(), color='r', linestyle='--', label=f'mean={z_norm.mean():.2f}')
ax.axvline(z_norm.mean() + z_norm.std(), color='orange', linestyle=':', label=f'+1 std={z_norm.mean()+z_norm.std():.2f}')
ax.set_xlabel('||z||')
ax.set_ylabel('Count')
ax.set_title('z norm distribution')
ax.legend()
plt.tight_layout()
plt.savefig(PLOT_DIR / '08_z_norm_distribution.png', dpi=150, bbox_inches='tight')
plt.close('all')
print('Saved 08_z_norm_distribution.png')

# ---- 11. C Control Test ----
print('\n=== 11. C CONTROL TEST ===')
model.eval()
example_idx = test_idx[0]
seq, n, c = dataset[example_idx]
seq = seq.unsqueeze(0).to(device); c = c.unsqueeze(0).to(device)
lengths = torch.tensor([n], device=device)

with torch.no_grad():
    mu, logvar = model.encode(seq, lengths)
    z_ctrl = model.reparameterize(mu, logvar)

key_onehot = c[0, 4:].clone()
key_name = KEY_NAMES[key_onehot.argmax().item()]

print(f'Original: {n} chords')
orig_chords = [decode_chord(seq[0, t].cpu().numpy()) for t in range(min(n, 16))]
print(f'Original: {" | ".join(orig_chords)}')
print(f'Original C: {c[0, :4].cpu().numpy()}, key={key_name}')

c_vals = [0.0, 0.25, 0.5, 0.75, 1.0]
print()
for frac in c_vals:
    c_perc_new = torch.full((1, 4), frac, device=device)
    c_new = torch.cat([c_perc_new, key_onehot.unsqueeze(0)], dim=-1)
    with torch.no_grad():
        gen_seq = model.generate(z_ctrl, c_new, max_len=n, device=device)
    gen_chords = [decode_chord(gen_seq[0, t].cpu().numpy()) for t in range(min(n, 16))]
    n_notes = sum(1 for ch in gen_chords if ch != 'N.C.')
    n_7ths = sum(1 for ch in gen_chords if '7' in ch)
    n_exts = sum(1 for ch in gen_chords if any(e in ch for e in ['9', '11', '13']))
    print(f'C={frac:.2f}: notes/step={n_notes/min(n,16):.2f} 7ths={n_7ths} exts={n_exts}')
    print(f'  {" | ".join(gen_chords)}')

del seq, c, z_ctrl; gc.collect()

# ---- 12. C Control - Multiple Examples ----
print('\n=== 12. C CONTROL — MULTI EXAMPLE ===')
def c_control_test(idx, c_vals=[0.0, 0.5, 1.0]):
    seq, n, c = dataset[idx]
    seq = seq.unsqueeze(0).to(device); c = c.unsqueeze(0).to(device)
    lengths = torch.tensor([n], device=device)
    with torch.no_grad():
        mu, logvar = model.encode(seq, lengths)
        z = model.reparameterize(mu, logvar)
    key_oh = c[0, 4:].clone()
    ori = [decode_chord(seq[0, t].cpu().numpy()) for t in range(min(n, 12))]
    print(f'Ex {idx} ({n}c, key={KEY_NAMES[key_oh.argmax().item()]})')
    print(f'  Orig: {" | ".join(ori)}')
    for frac in c_vals:
        c_new = torch.cat([torch.full((1,4), frac, device=device), key_oh.unsqueeze(0)], dim=-1)
        with torch.no_grad():
            gen = model.generate(z, c_new, max_len=min(n,12), device=device)
        chords = [decode_chord(gen[0, t].cpu().numpy()) for t in range(min(n, 12))]
        n7 = sum(1 for ch in chords if '7' in ch)
        n9 = sum(1 for ch in chords if any(e in ch for e in ['9', '11', '13']))
        print(f'  C={frac:.1f}: 7ths={n7} exts={n9}  {" | ".join(chords)}')
    print()

for idx in [0, 10, 50, 100, 200, 300]:
    c_control_test(idx)
gc.collect()

# ---- 13. Prior Sampling ----
print('\n=== 13. PRIOR SAMPLING ===')
test_configs = [
    {'c': [0.1, 0.1, 0.1, 0.1], 'key': 0, 'label': 'simple, C maj'},
    {'c': [0.9, 0.9, 0.9, 0.9], 'key': 0, 'label': 'complex, C maj'},
    {'c': [0.2, 0.2, 0.2, 0.2], 'key': 5, 'label': 'simple, F maj'},
    {'c': [0.8, 0.8, 0.8, 0.8], 'key': 7, 'label': 'complex, G maj'},
    {'c': [0.5, 0.5, 0.5, 0.5], 'key': 9, 'label': 'mid, A maj'},
    {'c': [0.5, 0.5, 0.5, 0.5], 'key': 11, 'label': 'mid, B maj'},
]

for cfg in test_configs:
    c_perc_t = torch.tensor(cfg['c'], device=device).unsqueeze(0)
    key_oh = torch.zeros(1, 12, device=device)
    key_oh[0, cfg['key']] = 1.0
    c_full = torch.cat([c_perc_t, key_oh], dim=-1)
    print(f'\n--- Prior sample: {cfg["label"]} ---')
    with torch.no_grad():
        mu_p, logvar_p = model.prior(c_full)
        print(f'  Prior μ[:4]: {mu_p[0,:4].tolist()}')
        print(f'  Prior σ[:4]: {torch.exp(0.5*logvar_p)[0,:4].tolist()}')
        for s in range(3):
            z_prior, _, _ = model.sample_prior(c_full, n=1, device=device)
            gen = model.generate(z_prior, c_full, max_len=8, device=device)
            chords = [decode_chord(gen[0, t].cpu().numpy()) for t in range(8)]
            print(f'  Sample {s}: {" | ".join(chords)}')
gc.collect()

# ---- 14. Enrichment Test ----
print('\n=== 14. ENRICHMENT TEST ===')

def progression_to_tensor(chord_names):
    vecs = []
    for c_name in chord_names:
        p = parse_chord(c_name)
        if p is not None:
            vecs.append(encode_chord(p))
        else:
            vecs.append(np.zeros(CHORD_DIM, dtype=np.float32))
    return torch.tensor(np.array(vecs), dtype=torch.float, device=device)

def detect_key(chord_names):
    root_map = {'C':0,'C#':1,'Db':1,'D':2,'D#':3,'Eb':3,'E':4,'F':5,'F#':6,'Gb':6,'G':7,'G#':8,'Ab':8,'A':9,'A#':10,'Bb':10,'B':11}
    counts = [0]*12
    for ch in chord_names:
        p = parse_chord(ch)
        if p:
            counts[p['root']] += 1
    return int(np.argmax(counts))

def enrich_progression(chord_names, max_len=None):
    model.eval()
    x = progression_to_tensor(chord_names).unsqueeze(0)
    n = x.size(1)
    max_len = max_len or n
    key_idx = detect_key(chord_names)
    key_oh = torch.zeros(1, 12, device=device)
    key_oh[0, key_idx] = 1.0
    lengths = torch.tensor([n], device=device)
    with torch.no_grad():
        mu, logvar = model.encode(x, lengths)
        z = model.reparameterize(mu, logvar)
    results = {}
    for label, c_val in [
        ('original C', None),
        ('C=0.0', 0.0), ('C=0.25', 0.25), ('C=0.5', 0.5), ('C=0.75', 0.75), ('C=1.0', 1.0),
    ]:
        c_t = torch.full((1, 4), 0.3, device=device) if c_val is None else torch.full((1, 4), c_val, device=device)
        c_full = torch.cat([c_t, key_oh], dim=-1)
        with torch.no_grad():
            gen = model.generate(z, c_full, max_len=max_len, device=device)
        chords = [decode_chord(gen[0, t].cpu().numpy()) for t in range(max_len)]
        n7 = sum(1 for ch in chords if '7' in ch)
        n_ext = sum(1 for ch in chords if any(e in ch for e in ['9', '11', '13']))
        n_changes = sum(1 for a, b in zip(chord_names[:max_len], chords) if a != b)
        results[label] = {'chords': chords, '7ths': n7, 'exts': n_ext, 'changes': n_changes}
    return results

def print_enrichment(prog_name, chord_names):
    print(f'\n{"="*70}')
    print(f'Progresión: {prog_name}')
    print(f'Original: {" | ".join(chord_names)}')
    print()
    results = enrich_progression(chord_names, max_len=len(chord_names))
    for label, r in results.items():
        changes_str = f' [Δ={r["changes"]}]' if label != 'original C' else ''
        print(f'  {label:12s}: 7ths={r["7ths"]} exts={r["exts"]}{changes_str}')
        print(f'              {" | ".join(r["chords"])}')

test_progressions = [
    ('C F G C', ['C', 'F', 'G', 'C']),
    ('Am F C G', ['Am', 'F', 'C', 'G']),
    ('I-IV-V-I in G', ['G', 'C', 'D', 'G']),
    ('Dm G C F', ['Dm', 'G', 'C', 'F']),
    ('Em C G D', ['Em', 'C', 'G', 'D']),
    ('12-bar blues in C', ['C7', 'C7', 'C7', 'C7', 'F7', 'F7', 'C7', 'C7', 'G7', 'F7', 'C7', 'C7']),
    ('canon in D', ['D', 'A', 'Bm', 'F#m', 'G', 'D', 'G', 'A']),
]

for name, chords in test_progressions:
    print_enrichment(name, chords)
gc.collect()

# ---- 15. Enrichment with Multipliers ----
print('\n=== 15. ENRICHMENT WITH MULTIPLIERS ===')
for idx_i in [0, 2, 5]:
    idx = test_idx[idx_i]
    seq, n, c = dataset[idx]
    seq = seq.unsqueeze(0).to(device); c = c.unsqueeze(0).to(device)
    lengths = torch.tensor([n], device=device)
    model.eval()
    with torch.no_grad():
        mu, logvar = model.encode(seq, lengths)
        z = model.reparameterize(mu, logvar)
    key_idx_v = c[0, 4:].argmax().item()
    orig_chords = [decode_chord(seq[0, t].cpu().numpy()) for t in range(min(n, 12))]
    c_perc_orig = c[0, :4].cpu()
    print(f'\n--- Example {idx_i} (key={KEY_NAMES[key_idx_v]}) ---')
    print(f'Original C: 7C={c_perc_orig[0]:.3f} VNSPC={c_perc_orig[1]:.3f} '
          f'DTMCVI={c_perc_orig[2]:.3f} VDR={c_perc_orig[3]:.3f}')
    print(f'Original:  {" | ".join(orig_chords)}')

    for mult_name, mults in [
        ('baseline', [1.0, 1.0, 1.0, 1.0]),
        ('more_7C', [2.0, 1.0, 1.0, 1.0]),
        ('less_7C', [0.0, 1.0, 1.0, 1.0]),
        ('more_VNSPC', [1.0, 2.0, 1.0, 1.0]),
        ('more_DTMCVI', [1.0, 1.0, 2.0, 1.0]),
        ('more_VDR', [1.0, 1.0, 1.0, 2.0]),
        ('all_rich', [1.5, 1.5, 1.5, 1.5]),
    ]:
        c_perc_mod = torch.clamp(c[0, :4] * torch.tensor(mults, device=device), 0, 1)
        c_mod = torch.cat([c_perc_mod.unsqueeze(0), c[:, 4:]], dim=-1)
        with torch.no_grad():
            gen = model.generate(z, c_mod, max_len=min(n, 12), device=device)
        chords = [decode_chord(gen[0, t].cpu().numpy()) for t in range(min(n, 12))]
        n_7 = sum(1 for ch in chords if '7' in ch)
        n_ext = sum(1 for ch in chords if any(e in ch for e in ['9', '11', '13']))
        print(f'  {mult_name:>12s} ({mults}): {" | ".join(chords)}')
        print(f'  {"":>12s}  7ths={n_7} exts={n_ext}')
gc.collect()

# ---- 16. Summary ----
print(f'\n{"="*80}')
print(f'{"V11 Summary vs V10":^80}')
print(f'{"="*80}')
print(f'{"Metric":<40s} {"V10":>15s} {"V11":>15s}')
print(f'{"-"*70}')
print(f'{"Decoder BYPASS Δ":<40s} {"0.21":>15s} {delta:>15.4f}')
print(f'{"Total KL (nats)":<40s} {"0.36":>15s} {total_kl:>15.4f}')
print(f'{"Active dims (>0.01)":<40s} {"10/32":>15s} {f"{active_dims}/32":>15s}')
print(f'{"z mean std":<40s} {"0.97":>15s} {z_std:>15.4f}')
print(f'{"c_predictor MSE":<40s} {"0.0064":>15s} {c_mse:>15.4f}')
print(f'{"key_predictor acc":<40s} {"96.9%":>15s} {key_acc*100:>14.1f}%')
print(f'{"PCA 2 PCs":<40s} {"33.5%":>15s} {cumsum[1]*100:>14.1f}%')
print(f'{"PCA 5 PCs":<40s} {"55.5%":>15s} {cumsum[4]*100:>14.1f}%')
print(f'{"="*80}')

print(f'\nAll plots saved to {PLOT_DIR}/')
print('Done!')
