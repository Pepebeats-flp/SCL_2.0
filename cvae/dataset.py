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


class ChordProgressionDataset(Dataset):
    def __init__(self, parquet_path=None, max_len=256, use_conditioning=False):
        if parquet_path is None:
            if use_conditioning:
                parquet_path = CONDITIONED_PARQUET
            else:
                parquet_path = DEFAULT_PARQUET
        self.max_len = max_len
        self.use_conditioning = use_conditioning

        print(f'Loading {parquet_path}...')
        df = pd.read_parquet(parquet_path)
        self.ids = df['id'].values
        self.n_chords = df['n_chords'].values
        self.symbolic_raw = df['symbolic'].values

        if self.use_conditioning:
            cond_data = {}
            for col in CONDITION_COLS:
                cond_data[col] = df[col].values.astype(np.float32)
            self.conditioning = np.stack(
                [cond_data[col] for col in CONDITION_COLS], axis=1
            )

        print(f'Loaded {len(df)} progressions'
              f'{" with conditioning" if use_conditioning else ""}')

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        n = self.n_chords[idx]
        seq = np.array(json.loads(self.symbolic_raw[idx]), dtype=np.float32)
        seq = torch.from_numpy(seq)

        if seq.size(0) > self.max_len:
            seq = seq[:self.max_len]
            n = self.max_len

        if self.use_conditioning:
            cond = torch.from_numpy(self.conditioning[idx])
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


def create_dataloader(parquet_path=None, batch_size=128, shuffle=True,
                      max_len=256, num_workers=0, use_conditioning=False,
                      split='train', val_split=0.1, test_size=1000):
    dataset = ChordProgressionDataset(parquet_path, max_len=max_len,
                                      use_conditioning=use_conditioning)

    total = len(dataset)
    gen = torch.Generator().manual_seed(42)
    indices = torch.randperm(total, generator=gen).tolist()

    if split == 'test':
        subset_indices = indices[:test_size]
    else:
        remaining = total - test_size
        val_size = int(remaining * val_split)
        train_size = remaining - val_size
        if split == 'train':
            subset_indices = indices[test_size:test_size + train_size]
        else:
            subset_indices = indices[test_size + train_size:]

    dataset = torch.utils.data.Subset(dataset, subset_indices)

    return DataLoader(
        dataset, batch_size=batch_size, shuffle=shuffle and split == 'train',
        collate_fn=collate_fn, num_workers=num_workers,
    )
