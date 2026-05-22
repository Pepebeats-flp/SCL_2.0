NOTE_NAMES = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
NOTE_TO_INDEX = {n: i for i, n in enumerate(NOTE_NAMES)}
NOTE_TO_INDEX.update({'Db': 1, 'Eb': 3, 'Gb': 6, 'Ab': 8, 'Bb': 10})

QUALITIES = ['maj', 'min', 'dim', 'aug', 'sus2', 'sus4', 'no3d', 'other']
QUALITY_TO_INDEX = {q: i for i, q in enumerate(QUALITIES)}

SEVENTH_TYPES = ['none', 'dom7', 'maj7', 'dim7', 'aug7']
SEVENTH_TO_INDEX = {s: i for i, s in enumerate(SEVENTH_TYPES)}

NUM_ROOT = 12
NUM_QUALITY = len(QUALITIES)
NUM_SEVENTH = len(SEVENTH_TYPES)
NUM_EXTENSIONS = 3
NUM_ALTERATIONS = 4
NUM_ADDED = 3
NUM_BASS = 13

CHORD_DIM = NUM_ROOT + NUM_QUALITY + NUM_SEVENTH + NUM_EXTENSIONS + NUM_ALTERATIONS + NUM_ADDED + NUM_BASS

EXT_NAMES = ['9', '11', '13']
ALT_NAMES = ['b9', '#9', '#11', 'b13']
ADDED_NAMES = ['add9', 'add11', 'add13']
