import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from PCS.pcs import PerceptualComplexityScore

PCS_WEIGHTS_PATH = Path(__file__).resolve().parent.parent / 'PCS' / 'pcs_weights.json'
JSYMBOLIC_PATH = Path(__file__).resolve().parent.parent / 'Dataset' / 'dataset_with_jsymbolic.parquet'
SYMBOLIC_PATH = Path(__file__).resolve().parent.parent / 'Dataset' / 'dataset_symbolic.parquet'
OUTPUT_PATH = Path(__file__).resolve().parent.parent / 'Dataset' / 'dataset_conditioned.parquet'

PCS_FEATURES = [
    'Seventh_Chords',
    'Variability_of_Number_of_Simultaneous_Pitch_Classes',
    'Distance_Between_Two_Most_Common_Vertical_Intervals',
    'Vertical_Dissonance_Ratio',
]

SHORT_NAMES = ['7C', 'VNSPC', 'DTMCVI', 'VDR']


def main():
    print('Loading PCS weights...')
    with open(PCS_WEIGHTS_PATH) as f:
        weights_data = json.load(f)
    pcs_calculator = PerceptualComplexityScore(weights_data)

    print(f'Loading {JSYMBOLIC_PATH}...')
    df_jsym = pd.read_parquet(JSYMBOLIC_PATH)
    print(f'  Rows: {len(df_jsym)}, columns: {list(df_jsym.columns)}')

    print(f'Loading {SYMBOLIC_PATH}...')
    df_sym = pd.read_parquet(SYMBOLIC_PATH)
    print(f'  Rows: {len(df_sym)}, columns: {list(df_sym.columns)}')

    assert len(df_jsym) == len(df_sym), 'Dataset size mismatch'
    assert (df_jsym['id'].values == df_sym['id'].values).all(), 'ID mismatch'

    print('Computing PCS values...')
    pcs_series = pcs_calculator.compute(df_jsym)
    print(f'  PCS range: [{pcs_series.min():.4f}, {pcs_series.max():.4f}], mean={pcs_series.mean():.4f}')

    print('Normalizing perceptual features...')
    cond_dict = {'id': df_sym['id'].values.tolist()}
    cond_dict['pcs'] = pcs_series.values.tolist()

    for feat, short in zip(PCS_FEATURES, SHORT_NAMES):
        col = df_jsym[feat].values.astype(np.float64)
        mn, mx = col.min(), col.max()
        if mx > mn:
            col_norm = (col - mn) / (mx - mn)
        else:
            col_norm = np.full_like(col, 0.5)
        cond_dict[short] = col_norm.tolist()
        print(f'  {short} ({feat}): range=[{mn:.4f}, {mx:.4f}] -> norm=[{col_norm.min():.4f}, {col_norm.max():.4f}]')

    df_out = pd.DataFrame(cond_dict)
    df_out['n_chords'] = df_sym['n_chords'].values
    df_out['symbolic'] = df_sym['symbolic'].values
    df_out['chords'] = df_sym['chords'].values

    print(f'\nSaving {OUTPUT_PATH} with columns: {list(df_out.columns)}')
    df_out.to_parquet(OUTPUT_PATH, index=False)
    print(f'Done. Shape: {df_out.shape}')
    print(f'Conditioning dimension: 1 (PCS) + 4 (perceptual) = 5')


if __name__ == '__main__':
    main()
