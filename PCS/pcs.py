import pandas as pd
import numpy as np

FEATURE_LABELS = {
    'VMS': 'Vertical_Minor_Seconds', 'VT': 'Vertical_Tritones',
    'VS': 'Vertical_Sevenths', 'VDR': 'Vertical_Dissonance_Ratio',
    'ST': 'Standard_Triads', '7C': 'Seventh_Chords',
    'NSC': 'Non-Standard_Chords', 'CC': 'Complex_Chords',
    'DTMCVI': 'Distance_Between_Two_Most_Common_Vertical_Intervals',
    'PRTMCVI': 'Prevalence_Ratio_of_Two_Most_Common_Vertical_Intervals',
    'VNSPC': 'Variability_of_Number_of_Simultaneous_Pitch_Classes',
}

SHORT_TO_LONG = FEATURE_LABELS
LONG_TO_SHORT = {v: k for k, v in FEATURE_LABELS.items()}


def compute_pcs_weights(csv_path='survey_correlations.csv', 
                         sig_csv_path='survey_pvalues.csv',
                         p_threshold=0.05,
                         exclude_mixed_sign=True):
    """
    Derive PCS weights from survey correlation data.
    
    Parameters:
    -----------
    csv_path : str
        CSV with columns: Feature, Bajo_r, Medio_r, Alto_r (or Bajo, Medio, Alto)
    sig_csv_path : str
        CSV with columns: Feature, Bajo_p, Medio_p, Alto_p (or Bajo, Medio, Alto)
    p_threshold : float
        p-value threshold for significance
    exclude_mixed_sign : bool
        If True, exclude features with mixed correlation direction across MSI levels
    
    Returns:
    --------
    dict : {long_feature_name: {'weight': float, 'invert': bool}}
    """
    corr = pd.read_csv(csv_path)
    pvals = pd.read_csv(sig_csv_path) if sig_csv_path else None
    
    # Detect column naming: with or without _r / _p suffix
    col_map_r = {'Bajo': 'Bajo_r', 'Medio': 'Medio_r', 'Alto': 'Alto_r'}
    col_map_p = {'Bajo': 'Bajo_p', 'Medio': 'Medio_p', 'Alto': 'Alto_p'}
    r_suffix = '_r' if 'Alto_r' in corr.columns else ''
    p_suffix = '_p' if pvals is not None and 'Alto_p' in pvals.columns else ''
    
    msi_levels = ['Bajo', 'Medio', 'Alto']
    
    weights = {}
    for _, row in corr.iterrows():
        short = row['Feature']
        long_name = SHORT_TO_LONG.get(short, short)
        r_vals = [row[f'{m}{r_suffix}'] for m in msi_levels]
        
        if pvals is not None:
            p_row = pvals[pvals['Feature'] == short]
            sig = False
            if len(p_row) > 0:
                p_vals = [p_row[f'Bajo{p_suffix}'].values[0], p_row[f'Medio{p_suffix}'].values[0], p_row[f'Alto{p_suffix}'].values[0]]
                sig = any(p < p_threshold for p in p_vals if not np.isnan(p))
        else:
            sig = True
        
        mean_r = np.mean(r_vals)
        mean_abs_r = np.mean([abs(v) for v in r_vals])
        
        has_mixed_sign = any(v > 0 for v in r_vals) and any(v < 0 for v in r_vals)
        
        if not sig:
            weights[long_name] = {'weight': 0.0, 'invert': False}
        elif exclude_mixed_sign and has_mixed_sign:
            weights[long_name] = {'weight': 0.0, 'invert': False}
        else:
            # For features with mixed signs, use the sign weighted by significance
            if pvals is not None and has_mixed_sign:
                p_row = pvals[pvals['Feature'] == short]
                if len(p_row) > 0:
                    sig_pos = False
                    sig_neg = False
                    for msi, r_name, p_name in [('Bajo',f'Bajo{r_suffix}',f'Bajo{p_suffix}'), ('Medio',f'Medio{r_suffix}',f'Medio{p_suffix}'), ('Alto',f'Alto{r_suffix}',f'Alto{p_suffix}')]:
                        r = row[r_name]; p = p_row[p_name].values[0]
                        if p < p_threshold:
                            if r > 0: sig_pos = True
                            if r < 0: sig_neg = True
                    if sig_pos and not sig_neg:
                        invert = False
                    elif sig_neg and not sig_pos:
                        invert = True
                    else:
                        invert = mean_r < 0
                else:
                    invert = mean_r < 0
            else:
                invert = mean_r < 0
            
            weight_val = max(0, mean_abs_r)
            weights[long_name] = {'weight': weight_val, 'invert': invert}
    
    return weights


class PerceptualComplexityScore:
    """
    Perceptual Complexity Score (PCS)
    
    Weights derived from correlations between jSymbolic features
    and perceived harmonic complexity (survey data).
    """
    
    def __init__(self, weights=None):
        if weights is None:
            self.weights = {}
        else:
            self.weights = weights
    
    @classmethod
    def from_survey(cls, csv_path='survey_correlations.csv', 
                    sig_csv_path='survey_pvalues.csv',
                    p_threshold=0.05, exclude_mixed_sign=True):
        w = compute_pcs_weights(csv_path, sig_csv_path, p_threshold, exclude_mixed_sign)
        return cls(weights=w)
    
    @property
    def total_weight(self):
        return sum(v['weight'] for v in self.weights.values())
    
    @property
    def active_features(self):
        return [f for f, v in self.weights.items() if v['weight'] > 0]
    
    def compute(self, df):
        result = pd.DataFrame(index=df.index)
        
        for feat, info in self.weights.items():
            if info['weight'] == 0 or feat not in df.columns:
                continue
            col = df[feat]
            if info['invert']:
                col = col.max() - col
            min_v, max_v = col.min(), col.max()
            col_norm = (col - min_v) / (max_v - min_v) if max_v > min_v else pd.Series(0.5, index=df.index)
            result[feat] = col_norm * info['weight']
        
        active = [c for c in result.columns]
        total_w = sum(self.weights[f]['weight'] for f in self.weights if self.weights[f]['weight'] > 0 and f in df.columns)
        result['pcs'] = result[active].sum(axis=1) / total_w if total_w > 0 else 0.0
        result['pcs'] = result['pcs'].clip(0, 1)
        
        return result['pcs']
    
    def summary(self):
        print(f"{'Feature':10s} {'Weight':>8s} {'Invert':>7s}")
        print('-' * 30)
        for feat, info in sorted(self.weights.items(), key=lambda x: x[1]['weight'], reverse=True):
            if info['weight'] > 0:
                short = LONG_TO_SHORT.get(feat, feat)
                print(f"{short:10s} {info['weight']:>8.3f} {str(info['invert']):>7s}")
        print(f"\nTotal active features: {len(self.active_features)}")
        print(f"Total weight sum: {self.total_weight:.3f}")
