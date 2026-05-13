import argparse
import json
import os
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pandas as pd
from mido import MidiFile, MidiTrack, Message, MetaMessage

SCRIPT_DIR = Path(__file__).parent.resolve()
PARQUET_PATH = SCRIPT_DIR / "dataset.parquet"
OUTPUT_DIR = SCRIPT_DIR / "midis"
BATCH_SIZE = 10000
TICKS = 480
BASE_NOTE = 60
JSYMBOLIC_JAR = SCRIPT_DIR.parent / "jSymbolic_2_2_user" / "jSymbolic2.jar"
JSYMBOLIC_CONFIG = SCRIPT_DIR / "jsymbolic_config.txt"
OUTPUT_PARQUET = SCRIPT_DIR / "dataset_with_jsymbolic.parquet"
OUTPUT_CSV = SCRIPT_DIR / "dataset_with_jsymbolic.csv"

FEATURE_NAMES = [
    "Vertical_Minor_Seconds",
    "Vertical_Tritones",
    "Vertical_Sevenths",
    "Vertical_Dissonance_Ratio",
    "Standard_Triads",
    "Seventh_Chords",
    "Non-Standard_Chords",
    "Complex_Chords",
    "Distance_Between_Two_Most_Common_Vertical_Intervals",
    "Prevalence_Ratio_of_Two_Most_Common_Vertical_Intervals",
    "Variability_of_Number_of_Simultaneous_Pitch_Classes",
]




def generate_midis_for_batch(df_batch, batch_dir):
    os.makedirs(batch_dir, exist_ok=True)
    count = 0
    for _, row in df_batch.iterrows():
        vectors = json.loads(row["vectors"]) if isinstance(row["vectors"], (str, bytes)) else row["vectors"]
        mid = MidiFile()
        track = MidiTrack()
        mid.tracks.append(track)

        track.append(MetaMessage("set_tempo", tempo=500000, time=0))
        track.append(MetaMessage("time_signature", numerator=4, denominator=4, time=0))

        for vector in vectors:
            active_notes = []
            for pitch_class, value in enumerate(vector):
                if value == 1:
                    note = BASE_NOTE + pitch_class
                    active_notes.append(note)
                    track.append(Message("note_on", note=note, velocity=80, time=0))

            for i, note in enumerate(active_notes):
                track.append(Message("note_off", note=note, velocity=80, time=TICKS if i == 0 else 0))

        midi_path = batch_dir / f"{row['id']:08d}.mid"
        mid.save(str(midi_path))
        count += 1

    return count


def run_jsymbolic_on_dir(input_dir, output_dir, timeout=600):
    os.makedirs(output_dir, exist_ok=True)
    values_xml = output_dir / "values.xml"
    defs_xml = output_dir / "definitions.xml"

    cmd = [
        "java", "-Xmx1g", "-jar", str(JSYMBOLIC_JAR),
        "-configrun", str(JSYMBOLIC_CONFIG),
        str(input_dir),
        str(values_xml),
        str(defs_xml),
    ]

    t0 = time.time()
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    elapsed = time.time() - t0

    if result.returncode != 0:
        stderr_short = result.stderr[:800] if result.stderr else "(no stderr)"
        print(f"  jSymbolic error (rc={result.returncode}): {stderr_short}")
        if result.stdout:
            print(f"  stdout: {result.stdout[:500]}")
        return None, elapsed

    csv_path = output_dir / "values.csv"
    if csv_path.exists():
        return csv_path, elapsed
    return None, elapsed


def parse_jsymbolic_csv(csv_path, batch_id_set):
    df = pd.read_csv(csv_path, skipinitialspace=True)
    first_col = df.columns[0]
    df.rename(columns={first_col: "midi_path"}, inplace=True)

    def extract_id(path):
        return int(Path(str(path).strip('"')).stem)

    df["id"] = df["midi_path"].apply(extract_id)
    df.drop(columns=["midi_path"], inplace=True)

    df = df[df["id"].isin(batch_id_set)].copy()

    keep = ["id"] + [c for c in df.columns if c in FEATURE_NAMES]
    missing = set(FEATURE_NAMES) - set(df.columns)
    if missing:
        print(f"  Warning: features not found in CSV: {missing}")

    return df[keep]


def process_batch(df_batch, batch_num, output_root, force=False):
    batch_dir = output_root / f"batch_{batch_num:04d}"
    jtmp_dir = output_root / f"jsym_batch_{batch_num:04d}"
    csv_path = jtmp_dir / "values.csv"

    if csv_path.exists() and not force:
        print(f"  Batch {batch_num}: CSV already exists, reusing")
        batch_ids = set(df_batch["id"])
        df_jsym = parse_jsymbolic_csv(csv_path, batch_ids)
        merged = df_batch.merge(df_jsym, on="id", how="left")
        return merged, "cached"

    t0 = time.time()
    n_midis = generate_midis_for_batch(df_batch, batch_dir)
    t_gen = time.time() - t0
    print(f"  Batch {batch_num}: generated {n_midis} MIDIs in {t_gen:.1f}s")

    print(f"  Batch {batch_num}: starting jSymbolic on {n_midis} MIDIs...")
    csv_result, t_jsym = run_jsymbolic_on_dir(batch_dir, jtmp_dir)
    if csv_result is None:
        print(f"  Batch {batch_num}: jSymbolic FAILED after {t_jsym:.1f}s")
        return None, "failed"

    print(f"  Batch {batch_num}: jSymbolic done in {t_jsym:.1f}s")

    batch_ids = set(df_batch["id"])
    df_jsym = parse_jsymbolic_csv(csv_result, batch_ids)
    merged = df_batch.merge(df_jsym, on="id", how="left")
    merged_rows = merged["id"].isin(df_jsym["id"]).sum()
    print(f"  Batch {batch_num}: merged {merged_rows}/{len(df_batch)} rows with features")
    return merged, "ok"


def main():
    parser = argparse.ArgumentParser(description="Generate dataset with jSymbolic features")
    parser.add_argument("--test", type=int, help="Run on first N rows only")
    parser.add_argument("--workers", type=int, default=1, help="Parallel jSymbolic processes")
    parser.add_argument("--resume", action="store_true", help="Skip already processed batches")
    parser.add_argument("--force", action="store_true", help="Reprocess even if cached")
    args = parser.parse_args()

    if not JSYMBOLIC_JAR.exists():
        print(f"ERROR: jSymbolic JAR not found at {JSYMBOLIC_JAR}")
        sys.exit(1)
    if not JSYMBOLIC_CONFIG.exists():
        print(f"ERROR: config not found at {JSYMBOLIC_CONFIG}")
        sys.exit(1)

    print(f"Loading {PARQUET_PATH}...")
    df = pd.read_parquet(PARQUET_PATH, engine="fastparquet")
    print(f"Loaded {len(df)} rows, columns: {list(df.columns)}")

    if args.test:
        df = df.head(args.test).copy()
        print(f"TEST MODE: using first {len(df)} rows")

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    batches = []
    for start in range(0, len(df), BATCH_SIZE):
        end = min(start + BATCH_SIZE, len(df))
        batch_num = start // BATCH_SIZE
        batches.append((df.iloc[start:end].copy(), batch_num))

    print(f"\nProcessing {len(batches)} batch(es) with {args.workers} worker(s)...\n")

    def run_batch(df_batch, batch_num):
        batch_dir = OUTPUT_DIR / f"batch_{batch_num:04d}"
        csv_path = OUTPUT_DIR / f"jsym_batch_{batch_num:04d}" / "values.csv"

        if args.resume and csv_path.exists():
            print(f"Batch {batch_num}: already processed (resume), skipping")
            df_jsym = parse_jsymbolic_csv(csv_path, set(df_batch["id"]))
            return df_batch.merge(df_jsym, on="id", how="left")

        result = process_batch(df_batch, batch_num, OUTPUT_DIR, force=args.force)
        if result is not None:
            merged, status = result
            return merged
        return None

    results = []
    if args.workers > 1:
        with ThreadPoolExecutor(max_workers=args.workers) as executor:
            fut_map = {executor.submit(run_batch, df_batch, bn): bn for df_batch, bn in batches}
            for future in as_completed(fut_map):
                bn = fut_map[future]
                try:
                    res = future.result()
                    if res is not None:
                        results.append(res)
                        print(f"  >> Batch {bn}: completed")
                except Exception as e:
                    print(f"  >> Batch {bn}: FAILED: {e}")
    else:
        for df_batch, batch_num in batches:
            res = run_batch(df_batch, batch_num)
            if res is not None:
                results.append(res)

    if not results:
        print("No results produced!")
        sys.exit(1)

    print(f"\nConcatenating {len(results)} batch(es)...")
    t0 = time.time()
    df_final = pd.concat(results, ignore_index=True)
    print(f"Concatenated in {time.time() - t0:.1f}s")
    print(f"Final shape: {df_final.shape}")
    print(f"Columns: {list(df_final.columns)}")

    feat_cols = [c for c in df_final.columns if c in FEATURE_NAMES]
    print(f"\nSummary stats for jSymbolic features ({len(feat_cols)}):")
    print(df_final[feat_cols].describe().to_string())

    df_final.to_parquet(OUTPUT_PARQUET, engine="fastparquet", index=False)
    print(f"\nSaved parquet: {OUTPUT_PARQUET}")
    df_final.to_csv(OUTPUT_CSV, index=False)
    print(f"Saved CSV:     {OUTPUT_CSV}")

    print("\nFirst 5 rows:")
    print(df_final.head(5).to_string())
    print(f"\nNull counts:")
    print(df_final[feat_cols].isnull().sum().to_string())


if __name__ == "__main__":
    main()
