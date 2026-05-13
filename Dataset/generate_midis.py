from mido import MidiFile, MidiTrack, Message, MetaMessage
import pandas as pd
import os

# ==========================================
# CONFIG
# ==========================================

PARQUET_PATH = "dataset.parquet"

OUTPUT_DIR = "midis"

BATCH_SIZE = 10000

TICKS = 480

BASE_NOTE = 60

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ==========================================
# CARGAR DATASET
# ==========================================

print("Loading dataset...")

df = pd.read_parquet(
    PARQUET_PATH,
    engine="fastparquet"
)

print(f"Loaded {len(df)} rows")

# ==========================================
# GENERAR MIDIS
# ==========================================

for idx, row in df.iterrows():

    vectors = row['vectors']

    # ======================================
    # BATCH FOLDER
    # ======================================

    batch_num = idx // BATCH_SIZE

    batch_folder = f"batch_{batch_num:04d}"

    batch_path = os.path.join(
        OUTPUT_DIR,
        batch_folder
    )

    os.makedirs(batch_path, exist_ok=True)

    # ======================================
    # FILENAME
    # ======================================

    filename = f"{idx:08d}.mid"

    midi_path = os.path.join(
        batch_path,
        filename
    )

    # ======================================
    # MIDI
    # ======================================

    mid = MidiFile()

    track = MidiTrack()

    mid.tracks.append(track)

    # metadata
    track.append(MetaMessage(
        'set_tempo',
        tempo=500000,
        time=0
    ))

    track.append(MetaMessage(
        'time_signature',
        numerator=4,
        denominator=4,
        time=0
    ))

    # ======================================
    # CREAR ACORDES
    # ======================================

    for vector in vectors:

        active_notes = []

        # NOTE ON
        for pitch_class, value in enumerate(vector):

            if value == 1:

                note = BASE_NOTE + pitch_class

                active_notes.append(note)

                track.append(Message(
                    'note_on',
                    note=note,
                    velocity=80,
                    time=0
                ))

        # NOTE OFF
        for i, note in enumerate(active_notes):

            track.append(Message(
                'note_off',
                note=note,
                velocity=80,
                time=TICKS if i == 0 else 0
            ))

    # ======================================
    # SAVE MIDI
    # ======================================

    mid.save(midi_path)

    # ======================================
    # LOGGING
    # ======================================

    if idx % 1000 == 0:

        print(
            f"[{idx}/{len(df)}] "
            f"Generated: {batch_folder}/{filename}"
        )

print("\nDONE")