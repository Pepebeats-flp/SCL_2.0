import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader

from chords.chord_encoder import encode_chord, parse_chord, CHORD_DIM

DATASET_DIR = Path(__file__).resolve().parent.parent / 'Dataset'
DEFAULT_PARQUET = DATASET_DIR / 'dataset_symbolic.parquet'
CONDITIONED_PARQUET = DATASET_DIR / 'dataset_conditioned.parquet'

CONDITION_COLS = ['pcs', '7C', 'VNSPC', 'DTMCVI', 'VDR']
CONDITION_DIM = len(CONDITION_COLS)

PERCEPTUAL_COLS = ['7C', 'VNSPC', 'DTMCVI', 'VDR']
PERCEPTUAL_DIM = len(PERCEPTUAL_COLS)


class ChordProgressionDataset(Dataset):
    def __init__(self, parquet_path=None, max_len=256, use_conditioning=False,
                 cond_cols=None, use_key=False):
        if parquet_path is None:
            if use_conditioning:
                parquet_path = CONDITIONED_PARQUET
            else:
                parquet_path = DEFAULT_PARQUET
        self.max_len = max_len
        self.use_conditioning = use_conditioning
        self.cond_cols = cond_cols if cond_cols is not None else CONDITION_COLS
        self.use_key = use_key

        print(f'Loading {parquet_path}...')
        needed_cols = ['symbolic', 'n_chords']
        if self.use_conditioning:
            needed_cols += [c for c in self.cond_cols if c not in needed_cols]
        df = pd.read_parquet(parquet_path, columns=needed_cols)

        # Store raw JSON strings — parse lazily in __getitem__.
        # This avoids loading all 877K sequences as float32 arrays (~13GB).
        self.raw_seqs = df['symbolic'].values
        self.n_chords = df['n_chords'].values

        if self.use_conditioning:
            cond_data = {}
            for col in self.cond_cols:
                cond_data[col] = df[col].values.astype(np.float32)
            self.conditioning = np.stack(
                [cond_data[col] for col in self.cond_cols], axis=1
            )

        print(f'Loaded {len(self.raw_seqs)} progressions'
              f'{" with conditioning" if use_conditioning else ""}'
              f'{" + key" if use_key else ""}')

    def __len__(self):
        return len(self.raw_seqs)

    def _detect_key(self, seq):
        roots = seq[:self.max_len, :12].argmax(dim=-1)
        if roots.numel() == 0:
            key = 0
        else:
            counts = torch.bincount(roots.long(), minlength=12)
            key = counts.argmax().item()
        key_onehot = torch.zeros(12)
        key_onehot[key] = 1.0
        return key_onehot

    def __getitem__(self, idx):
        arr = json.loads(self.raw_seqs[idx])
        if len(arr) > self.max_len:
            arr = arr[:self.max_len]
        seq = torch.from_numpy(np.array(arr, dtype=np.float32))
        n = len(seq)

        if self.use_conditioning:
            cond = torch.from_numpy(self.conditioning[idx])
            if self.use_key:
                key_onehot = self._detect_key(seq)
                cond = torch.cat([cond, key_onehot])
            return seq, n, cond

        return seq, n


def collate_fn(batch):
    has_conditioning = len(batch[0]) == 3
    seqs, lengths = zip(*[(b[0], b[1]) for b in batch])
    max_n = max(lengths)
    dim = seqs[0].size(-1)
    padded = torch.zeros(len(seqs), max_n, dim)
    for i, (seq, n) in enumerate(zip(seqs, lengths)):
        padded[i, :n] = seq
    lengths = torch.tensor(lengths, dtype=torch.long)

    if has_conditioning:
        conds = torch.stack([b[2] for b in batch])
        return padded, lengths, conds

    return padded, lengths


def split_indices(total, val_split=0.1, test_size=1000, seed=42):
    gen = torch.Generator().manual_seed(seed)
    idx = torch.randperm(total, generator=gen).tolist()
    test_idx = idx[:test_size]
    remaining = total - test_size
    val_size = int(remaining * val_split)
    train_idx = idx[test_size:test_size + remaining - val_size]
    val_idx = idx[test_size + remaining - val_size:]
    return {'train': train_idx, 'val': val_idx, 'test': test_idx}


def create_dataloader(parquet_path=None, batch_size=256, shuffle=True,
                      max_len=256, num_workers=0, use_conditioning=False,
                      split='train', val_split=0.1, test_size=1000,
                      dataset=None, indices=None, use_key=False):
    if dataset is None:
        dataset = ChordProgressionDataset(parquet_path, max_len=max_len,
                                          use_conditioning=use_conditioning,
                                          use_key=use_key)
    if indices is None:
        indices = split_indices(len(dataset), val_split=val_split, test_size=test_size)

    subset = torch.utils.data.Subset(dataset, indices[split])

    return DataLoader(
        subset, batch_size=batch_size, shuffle=shuffle and split == 'train',
        collate_fn=collate_fn, num_workers=num_workers,
        pin_memory=True, persistent_workers=num_workers > 0,
        prefetch_factor=2 if num_workers > 0 else None,
    )
