import re

import numpy as np

from .vocab import (
    NOTE_NAMES, NOTE_TO_INDEX, QUALITIES, QUALITY_TO_INDEX,
    SEVENTH_TYPES, SEVENTH_TO_INDEX,
    NUM_ROOT, NUM_QUALITY, NUM_SEVENTH, NUM_EXTENSIONS, NUM_ALTERATIONS,
    NUM_ADDED, NUM_BASS, CHORD_DIM, EXT_NAMES, ALT_NAMES, ADDED_NAMES,
)

ROOT_SLOT = 0
QUALITY_SLOT = ROOT_SLOT + NUM_ROOT
SEVENTH_SLOT = QUALITY_SLOT + NUM_QUALITY
EXT_SLOT = SEVENTH_SLOT + NUM_SEVENTH
ALT_SLOT = EXT_SLOT + NUM_EXTENSIONS
ADDED_SLOT = ALT_SLOT + NUM_ALTERATIONS
BASS_SLOT = ADDED_SLOT + NUM_ADDED

_TOK_PATTERN = re.compile(
    r'dim7|dimmaj7|dim'
    r'|augmaj7|aug7|aug'
    r'|mmaj7|minmaj7|min7|min|m13|m11|m9|m7'
    r'|maj13|maj11|maj9|maj7|maj'
    r'|no3d'
    r'|sus2|sus4|us2|us4'
    r'|\+'
    r'|add(?:9|11|13)'
    r'|1[13][bs]|11[bs]|9[bs]'
    r'|1[13]|11|9|7'
    r'|[bs]1[13]|[bs]9|[bs]11|[bs]13'
    r'|[a-zA-Z]',
)

_ROOTS_FLAT = {'Db': 1, 'Eb': 3, 'Gb': 6, 'Ab': 8, 'Bb': 10, 'Cb': 11, 'Fb': 4}
_ROOTS_SHARP = {'Cs': 1, 'Ds': 3, 'Es': 5, 'Fs': 6, 'Gs': 8, 'As': 10, 'Bs': 0}

_ALT_MAP_RS = {'b9': 'b9', 's9': '#9', 'b11': 'b11', 's11': '#11', 'b13': 'b13', 's13': '#13',
               '9b': 'b9', '9s': '#9', '11b': 'b11', '11s': '#11', '13b': 'b13', '13s': '#13'}
_ALT_TO_OUT = {'b9': 'b9', '#9': 's9', 'b11': 'b11', '#11': 's11', 'b13': 'b13', '#13': 's13'}


def _parse_note(name):
    if not name:
        return None
    if name in NOTE_TO_INDEX:
        return NOTE_TO_INDEX[name]
    if len(name) >= 2 and name[0] in 'CDEFGAB' and name[1] in 's#':
        two = name[0] + '#'
        if two in NOTE_TO_INDEX:
            return NOTE_TO_INDEX[two]
    return None


def _tokenize(rest):
    tokens = []
    while rest:
        m = _TOK_PATTERN.match(rest)
        if m:
            tokens.append(m.group(0))
            rest = rest[m.end():]
        else:
            break
    return tokens


def parse_chord(chord_str):
    parts = chord_str.split('/')
    chord_part = parts[0]
    bass_part = parts[1] if len(parts) > 1 else None

    if not chord_part:
        return None

    root = None
    quality = 'maj'
    seventh = 'none'
    extensions = set()
    alterations = set()
    added = set()

    rest = chord_part
    if not rest:
        return None

    first = rest[0]
    second = rest[1] if len(rest) > 1 else ''

    if first not in 'CDEFGAB':
        return None

    if second == 'b' and rest[:2] in _ROOTS_FLAT:
        root = _ROOTS_FLAT[rest[:2]]; rest = rest[2:]
    elif second and second in 's#' and first not in ('C', 'F'):
        rem = rest[2:]
        if not rem.startswith('us2') and not rem.startswith('us4'):
            two = first + '#'
            if two in NOTE_TO_INDEX:
                root = NOTE_TO_INDEX[two]; rest = rem
            else:
                root = NOTE_TO_INDEX[first]; rest = rest[1:]
        else:
            root = NOTE_TO_INDEX[first]; rest = rest[1:]
    elif second == 's' and first in ('C', 'F'):
        rem = rest[1:]
        root = NOTE_TO_INDEX[first]; rest = rem
    elif second == '#':
        two = first + '#'
        if two in NOTE_TO_INDEX:
            root = NOTE_TO_INDEX[two]; rest = rest[2:]
        else:
            root = NOTE_TO_INDEX[first]; rest = rest[1:]
    else:
        root = NOTE_TO_INDEX[first]; rest = rest[1:]

    if root is None:
        return None

    tokens = _tokenize(rest)
    i = 0
    while i < len(tokens):
        t = tokens[i]
        if t in ('dim7',):
            quality = 'dim'; seventh = 'dim7'
        elif t == 'dimmaj7':
            quality = 'dim'; seventh = 'maj7'
        elif t == 'dim':
            quality = 'dim'
        elif t == 'augmaj7':
            quality = 'aug'; seventh = 'maj7'
        elif t in ('aug7',):
            quality = 'aug'; seventh = 'aug7'
        elif t == 'aug':
            quality = 'aug'
        elif t in ('mmaj7', 'minmaj7'):
            quality = 'min'; seventh = 'maj7'
        elif t == 'min7':
            quality = 'min'; seventh = 'dom7'
        elif t == 'min':
            quality = 'min'
        elif t in ('m13', 'm11', 'm9'):
            quality = 'min'
            extensions.add(t[1:])
        elif t == 'm7':
            quality = 'min'; seventh = 'dom7'
        elif t.startswith('m') and (not t[1:] or not t[1:].isalnum()):
            quality = 'min'
        elif t in ('maj13', 'maj11', 'maj9'):
            quality = 'maj'
            seventh = 'maj7'
            extensions.add(t[3:])
        elif t == 'maj7':
            quality = 'maj'; seventh = 'maj7'
        elif t == 'maj':
            quality = 'maj'
        elif t == 'no3d':
            quality = 'no3d'
        elif t in ('sus2', 'us2', 'sus4', 'us4'):
            quality = 'sus2' if '2' in t else 'sus4'
        elif t == '+':
            quality = 'aug'
        elif t.startswith('add'):
            added.add(t)
        elif t == '7' and seventh == 'none':
            seventh = 'dom7'
        elif t in ('13', '11', '9'):
            extensions.add(t)
        elif t in _ALT_MAP_RS:
            alterations.add(_ALT_MAP_RS[t])
        i += 1

    bass = None
    if bass_part:
        bass_part = bass_part.strip()
        if bass_part:
            bass = _parse_note(bass_part)
            if bass is None:
                for b_rep in ['C', 'C#', 'Db', 'D', 'D#', 'Eb', 'E', 'F', 'F#', 'Gb', 'G', 'G#', 'Ab', 'A', 'A#', 'Bb', 'B']:
                    if bass_part.startswith(b_rep):
                        bass = _parse_note(b_rep)
                        break

    if quality not in QUALITY_TO_INDEX:
        quality = 'other'

    return {
        'root': root,
        'quality': quality,
        'seventh': seventh,
        'extensions': extensions,
        'alterations': alterations,
        'added': added,
        'bass': bass,
    }


def encode_chord(parsed):
    vec = np.zeros(CHORD_DIM, dtype=np.float32)
    if parsed is None:
        return vec

    vec[ROOT_SLOT + parsed['root']] = 1.0

    q_idx = QUALITY_TO_INDEX.get(parsed['quality'], QUALITY_TO_INDEX['other'])
    vec[QUALITY_SLOT + q_idx] = 1.0

    s_idx = SEVENTH_TO_INDEX.get(parsed['seventh'], 0)
    vec[SEVENTH_SLOT + s_idx] = 1.0

    for i, ext in enumerate(EXT_NAMES):
        if ext in parsed['extensions']:
            vec[EXT_SLOT + i] = 1.0

    for i, alt in enumerate(ALT_NAMES):
        if alt in parsed['alterations']:
            vec[ALT_SLOT + i] = 1.0

    for i, ad in enumerate(ADDED_NAMES):
        if ad in parsed['added']:
            vec[ADDED_SLOT + i] = 1.0

    if parsed['bass'] is not None:
        vec[BASS_SLOT + parsed['bass']] = 1.0
    else:
        vec[BASS_SLOT + 12] = 1.0

    return vec


def decode_chord(vec):
    root_idx = int(np.argmax(vec[ROOT_SLOT:QUALITY_SLOT]))
    q_idx = int(np.argmax(vec[QUALITY_SLOT:SEVENTH_SLOT]))
    s_idx = int(np.argmax(vec[SEVENTH_SLOT:EXT_SLOT]))

    root = NOTE_NAMES[root_idx]
    quality = QUALITIES[q_idx]
    seventh = SEVENTH_TYPES[s_idx]

    ext_flags = [vec[EXT_SLOT + i] > 0.5 for i in range(NUM_EXTENSIONS)]
    alt_flags = [vec[ALT_SLOT + i] > 0.5 for i in range(NUM_ALTERATIONS)]
    added_flags = [vec[ADDED_SLOT + i] > 0.5 for i in range(NUM_ADDED)]

    ext_strs = [EXT_NAMES[i] for i, f in enumerate(ext_flags) if f]
    alt_strs = [_ALT_TO_OUT[ALT_NAMES[i]] for i, f in enumerate(alt_flags) if f]
    added_strs = [ADDED_NAMES[i] for i, f in enumerate(added_flags) if f]

    parts = [root]

    q_lbl = {'min': 'm', 'dim': 'dim', 'aug': 'aug', 'sus2': 'sus2',
             'sus4': 'sus4', 'no3d': 'no3d'}
    is_sus = quality in ('sus2', 'sus4')

    if seventh == 'dim7':
        if quality == 'dim':
            parts.append('dim7')
        else:
            parts.append(q_lbl.get(quality, ''))
            parts.append('dim7')
    elif seventh == 'aug7':
        if quality == 'aug':
            parts.append('aug7')
        else:
            parts.append(q_lbl.get(quality, ''))
            parts.append('aug7')
    else:
        if is_sus:
            if seventh == 'dom7':
                parts.append('7' + q_lbl[quality])
            else:
                parts.append(q_lbl[quality])
        else:
            if quality in q_lbl:
                parts.append(q_lbl[quality])
            if seventh == 'dom7':
                parts.append('7')
            elif seventh == 'maj7':
                if quality == 'min':
                    parts.append('maj7')
                elif ext_strs:
                    parts.append('maj' + ''.join(sorted(ext_strs)))
                    ext_strs = []
                else:
                    parts.append('maj7')
            elif seventh == 'none' and alt_strs and quality not in q_lbl:
                parts.append('maj')

    for a in added_strs:
        parts.append(a)
    for e in ext_strs:
        parts.append(e)
    for a in alt_strs:
        parts.append(a)

    bass_idx = int(np.argmax(vec[BASS_SLOT:]))
    if bass_idx < 12:
        parts.append('/')
        parts.append(NOTE_NAMES[bass_idx])

    return ''.join(parts)


def progression_to_encoding(chord_names):
    encodings = []
    for name in chord_names:
        parsed = parse_chord(name)
        encodings.append(encode_chord(parsed))
    return np.stack(encodings, axis=0)
