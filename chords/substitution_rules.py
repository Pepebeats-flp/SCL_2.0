"""Chord substitution rules extracted from the pairs dataset.

The pairs dataset captures real human reharmonizations where arrangers
change chord triads (e.g., F→Am, C→Dm) — different from the heuristic's
extension enrichment (C→C7).

Usage:
    from chords.substitution_rules import SubstitutionRuleEngine
    
    engine = SubstitutionRuleEngine()
    subs = engine.get_substitutions("C", key=0)
    # Returns: [{"chord": "Am", "weight": 0.8, "type": "relative_minor"}, ...]
    
    engine.apply_substitutions(progression, rate=0.3, key=0)
    # Returns: substituted progression
"""
import json
import sys
from pathlib import Path
from collections import defaultdict
import numpy as np

from .chord_encoder import parse_chord, encode_chord, decode_chord
from .vocab import NOTE_NAMES, CHORD_DIM

ENRICHMENT_CACHE = Path('Dataset/chord_enrichment_cache.npz')

# === Substitution rules extracted from pairs dataset ===
# Format: (original_root, original_quality) -> (target_root, target_quality)
# Weight = frequency in pairs dataset / total substitutions
# These are the top 20 substitutions from 98 pairs (24 same-key)

PAIRS_SUBSTITUTIONS = [
    # (orig_root, orig_qual, tgt_root, tgt_qual, weight)
    # F  →  C      (7x): dominant-function resolution
    (5, 'maj', 0, 'maj', 7),
    # C  →  Dm     (6x): IV substitution, common tone
    (0, 'maj', 2, 'min', 6),
    # F  →  Am     (5x): relative minor of IV
    (5, 'maj', 9, 'min', 5),
    # C  →  Am     (5x): relative minor
    (0, 'maj', 9, 'min', 5),
    # Am →  Dm     (5x): i→iv in Am
    (9, 'min', 2, 'min', 5),
    # G  →  C      (4x): resolution
    (7, 'maj', 0, 'maj', 4),
    # Dm →  F      (4x): i→III in Dm
    (2, 'min', 5, 'maj', 4),
    # F  →  G      (4x): IV→V
    (5, 'maj', 7, 'maj', 4),
    # F  →  Em     (4x): IV→iii
    (5, 'maj', 4, 'min', 4),
    # D  →  C      (3x)
    (2, 'maj', 0, 'maj', 3),
    # Am →  G      (3x)
    (9, 'min', 7, 'maj', 3),
    # Am →  E      (3x)
    (9, 'min', 4, 'maj', 3),
    # C  →  G      (3x)
    (0, 'maj', 7, 'maj', 3),
    # G  →  F      (3x)
    (7, 'maj', 5, 'maj', 3),
    # C  →  F      (3x)
    (0, 'maj', 5, 'maj', 3),
    # F  →  Dm     (3x)
    (5, 'maj', 2, 'min', 3),
    # C#dim→ Dm    (2x)
    (1, 'dim', 2, 'min', 2),
    # F  →  A#     (2x)
    (5, 'maj', 10, 'maj', 2),
    # F  →  C#     (2x)
    (5, 'maj', 1, 'maj', 2),
    # Dm →  C      (2x)
    (2, 'min', 0, 'maj', 2),
]

# === Derived general rules (generalized from pairs patterns) ===
# These generalize the specific substitutions to any key
GENERAL_RULES = [
    # Type: relative minor substitution (major → minor 3rd down)
    # C → Am, F → Dm, G → Em
    ('relative_minor', lambda r, q: q == 'maj', lambda r, q: ((r - 3) % 12, 'min')),
    # Type: relative major (minor → major 3rd up)  
    # Am → C, Dm → F, Em → G
    ('relative_major', lambda r, q: q == 'min', lambda r, q: ((r + 3) % 12, 'maj')),
    # Type: mediant (major → major 3rd up)
    # C → E, F → A
    ('mediant_up', lambda r, q: q == 'maj', lambda r, q: ((r + 4) % 12, 'maj')),
    # Type: submediant (major → major 3rd down)
    # E → C, A → F
    ('submediant', lambda r, q: q == 'maj', lambda r, q: ((r - 4) % 12, 'maj')),
    # Type: supertonic (major → minor 2nd up)
    # C → Dm
    ('supertonic', lambda r, q: q == 'maj', lambda r, q: ((r + 2) % 12, 'min')),
    # Type: subdominant parallel (major → minor 3rd up)
    # C → Em
    ('subdominant_parallel', lambda r, q: q == 'maj', lambda r, q: ((r + 4) % 12, 'min')),
    # Type: diminished resolution (dim → minor 2nd up)
    # C#dim → Dm
    ('dim_resolution', lambda r, q: q == 'dim', lambda r, q: ((r + 1) % 12, 'min')),
    # Type: dominant resolution (major 5th up → root)
    # G → C, D → G
    ('dominant_resolution', lambda r, q: True, lambda r, q: ((r + 7) % 12, q)),
    # Type: plagal (major 4th up → chord a 4th away)
    # C → F
    ('plagal', lambda r, q: q == 'maj', lambda r, q: ((r + 5) % 12, 'maj')),
]

# Manual weights for general rules
RULE_WEIGHTS = {
    'relative_minor': 0.20,
    'relative_major': 0.15,
    'mediant_up': 0.08,
    'submediant': 0.08,
    'supertonic': 0.12,
    'subdominant_parallel': 0.10,
    'dim_resolution': 0.05,
    'dominant_resolution': 0.10,
    'plagal': 0.05,
}


def _chord_to_key(root, quality):
    """Map 'maj' → root, 'min' → root, 'dim' → root."""
    return (root, quality)


def _chord_name(root, quality):
    note = NOTE_NAMES[root]
    if quality == 'min': return note + 'm'
    if quality == 'dim': return note + 'dim'
    if quality == 'aug': return note + 'aug'
    if quality == 'sus2': return note + 'sus2'
    if quality == 'sus4': return note + 'sus4'
    if quality == 'no3d': return note + 'no3d'
    return note


class SubstitutionRuleEngine:
    """Apply human-like chord substitutions to a progression.

    Uses rules extracted from the enrichment pairs dataset.
    The rules capture real arranger behavior: changing chord triads
    while maintaining musical coherence.

    Integration with enrichment pipeline:
    1. Apply substitutions to the original progression
    2. Encode substituted progression to z
    3. Apply heuristic enrichment (extension) on z
    4. Decode enriched z with high-C target
    """

    def __init__(self, pairs_weight=1.0, general_weight=0.5, c_guide_dim=None):
        self.pairs_weight = pairs_weight
        self.general_weight = general_weight
        self.c_guide_dim = c_guide_dim
        self._chord_c_table = None
        if c_guide_dim is not None:
            self._load_c_table()
        self._build_index()

    def _load_c_table(self):
        """Load chord→C enrichment table from 877k."""
        try:
            data = np.load(ENRICHMENT_CACHE)
            names = data['names']
            means = data['means']
            self._chord_c_table = {n: means[i] for i, n in enumerate(names)}
            print(f'[SubstitutionRuleEngine] Loaded {len(self._chord_c_table)} chord-C profiles'
                  f' (guide dim={self.c_guide_dim})', file=sys.stderr)
        except FileNotFoundError:
            print(f'[SubstitutionRuleEngine] WARN: {ENRICHMENT_CACHE} not found', file=sys.stderr)

    def _build_index(self):
        # Index pairs substitutions by original chord key
        self.pairs_by_orig = defaultdict(list)
        total = sum(w for _, _, _, _, w in PAIRS_SUBSTITUTIONS)
        for r_o, q_o, r_t, q_t, w in PAIRS_SUBSTITUTIONS:
            orig = _chord_name(r_o, q_o)
            tgt = _chord_name(r_t, q_t)
            self.pairs_by_orig[orig].append({
                'target': tgt,
                'weight': w / total * self.pairs_weight,
                'source': 'pairs',
            })

    def get_substitutions(self, chord_name, key=None, c_orig=None):
        """Get candidate substitutions for a chord.

        Args:
            chord_name: e.g. 'C', 'Am', 'F#dim'
            key: optional key index (0-11) for key-constrained substitutions
            c_orig: original C values [7C, VNSPC, DTMCVI, VDR] to guide selection

        Returns:
            list of {target, weight, source}
        """
        candidates = []

        # Direct from pairs
        if chord_name in self.pairs_by_orig:
            candidates.extend(self.pairs_by_orig[chord_name])

        # General rules
        parsed = parse_chord(chord_name)
        if parsed:
            r = parsed['root']
            q = parsed['quality']
            for name, condition, transform in GENERAL_RULES:
                if condition(r, q):
                    nr, nq = transform(r, q)
                    tgt = _chord_name(nr, nq)
                    if tgt != chord_name:
                        candidates.append({
                            'target': tgt,
                            'weight': RULE_WEIGHTS.get(name, 0.05),
                            'source': name,
                        })

        # Sort by weight, deduplicate
        seen = set()
        unique = []
        for c in sorted(candidates, key=lambda x: -x['weight']):
            if c['target'] not in seen:
                seen.add(c['target'])
                unique.append(c)

        # C-guided reweighting: prefer substitutions that increase target C dim
        if self.c_guide_dim is not None and self._chord_c_table is not None and c_orig is not None:
            orig_c = self._chord_c_table.get(chord_name)
            if orig_c is not None:
                for c in unique:
                    tgt_c = self._chord_c_table.get(c['target'])
                    if tgt_c is not None:
                        dc = tgt_c[self.c_guide_dim] - orig_c[self.c_guide_dim]
                        # Boost weight if DC > 0 (substitution enriches target dim)
                        c['weight'] *= (1.0 + max(0, dc * 20))

        return sorted(unique, key=lambda x: -x['weight'])

    def apply_substitutions(self, chords, rate=0.3, key=None, c_values=None):
        """Apply substitutions to a chord progression.

        Args:
            chords: list of chord name strings
            rate: fraction of chords to substitute (0-1)
            key: optional key index
            c_values: optional list of C vectors per chord or a single C vector

        Returns:
            list of chord name strings (may be same as input)
        """
        result = chords.copy()
        n = len(chords)
        if n < 2:
            return result

        # Select positions to substitute
        n_subs = max(1, int(n * rate)) if rate > 0 else 0
        positions = sorted(np.random.choice(n, n_subs, replace=False))

        for pos in positions:
            c_orig = c_values[pos] if isinstance(c_values, (list, np.ndarray)) and c_values is not None else None
            candidates = self.get_substitutions(chords[pos], key=key, c_orig=c_orig)
            if candidates:
                weights = [c['weight'] for c in candidates]
                total = sum(weights)
                if total > 0:
                    probs = [w / total for w in weights]
                    chosen = candidates[np.random.choice(len(candidates), p=probs)]
                    result[pos] = chosen['target']

        return result


def test():
    engine = SubstitutionRuleEngine()
    print(f'Substitutions for C: {engine.get_substitutions("C")}')
    print(f'Substitutions for F: {engine.get_substitutions("F")}')
    print(f'Substitutions for Am: {engine.get_substitutions("Am")}')
    print(f'Substitutions for Dm: {engine.get_substitutions("Dm")}')

    prog = ['C', 'F', 'G', 'C']
    print(f'\nOriginal: {prog}')
    for rate in [0.0, 0.25, 0.5, 1.0]:
        subs = engine.apply_substitutions(prog, rate=rate)
        print(f'  rate={rate}: {subs}')


if __name__ == '__main__':
    test()
