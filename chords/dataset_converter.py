import json
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

from chords.chord_encoder import parse_chord, progression_to_encoding

DATASET_DIR = Path(__file__).resolve().parent.parent / 'Dataset'
INPUT_PARQUET = DATASET_DIR / 'dataset.parquet'
OUTPUT_PARQUET = DATASET_DIR / 'dataset_symbolic.parquet'


def convert_dataset(input_path=INPUT_PARQUET, output_path=OUTPUT_PARQUET,
                    max_rows=None):
    print(f'Loading {input_path}...')
    df = pd.read_parquet(input_path)
    print(f'Loaded {len(df)} rows, columns: {list(df.columns)}')

    if max_rows:
        df = df.head(max_rows).copy()
        print(f'LIMITED to {len(df)} rows')

    rows = []
    failed = 0
    for _, row in tqdm(df.iterrows(), total=len(df), desc='Converting'):
        raw = row['chords']
        chords = json.loads(raw) if isinstance(raw, (str, bytes)) else raw
        if not isinstance(chords, (list, tuple)):
            failed += 1
            continue

        good = True
        for c in chords:
            if not isinstance(c, str) or parse_chord(c) is None:
                good = False
                break
        if not good:
            failed += 1
            continue

        encoding = progression_to_encoding(chords)

        record = {
            'id': row['id'],
            'n_chords': row['n_chords'],
            'symbolic': json.dumps(encoding.tolist()),
            'chords': json.dumps(chords),
        }
        rows.append(record)

    result = pd.DataFrame(rows)
    print(f'\nProcessed {len(result)} rows ({failed} failed)')
    dims = np.array(json.loads(result['symbolic'].iloc[0]))
    print(f'Symbolic shape: ({result["n_chords"].iloc[0]}, {dims.shape[1] if dims.ndim > 1 else dims.shape[0]})')

    result.to_parquet(output_path, index=False)
    print(f'Saved: {output_path}')

    return result


if __name__ == '__main__':
    import sys
    max_rows = int(sys.argv[1]) if len(sys.argv) > 1 else None
    convert_dataset(max_rows=max_rows)
