#!/usr/bin/env python3
"""
filter_sp_by_results.py

Filters a PRESTO .singlepulse file based on a classification CSV file.

This script reads a CSV file (e.g., from an ML classifier) that contains
candidate filenames and labels (1 for FRB, 0 for RFI). It then identifies
the candidates labeled as FRBs and creates a new .singlepulse file containing
only those candidates.

The original .singlepulse file is backed up with a '.oldsingpulseold' extension.
"""
import argparse
import pandas as pd
import sys
import shutil
import re
from pathlib import Path

def filter_singlepulse_file(directory: Path):
    """
    Finds classification and singlepulse files in a directory,
    filters the singlepulse file, and backs up the original.
    """
    # 1. Define and find the necessary files
    csv_path = directory / 'results_model_z.csv'
    if not csv_path.exists():
        print(f"Error: Classification CSV file not found at '{csv_path}'", file=sys.stderr)
        sys.exit(1)

    sp_files = list(directory.glob('*.singlepulse'))
    if not sp_files:
        print(f"Error: No '.singlepulse' files found in '{directory}'", file=sys.stderr)
        sys.exit(1)
    if len(sp_files) > 1:
        print(f"Warning: Found multiple '.singlepulse' files. Using the first one: '{sp_files[0]}'", file=sys.stderr)
    
    sp_path = sp_files[0]
    print(f"Processing singlepulse file: {sp_path}")
    print(f"Using classification from: {csv_path}")

    # 2. Read the CSV and extract FRB candidate parameters
    try:
        df_results = pd.read_csv(csv_path)
        df_frbs = df_results[df_results['label'] == 1.0].copy()
    except Exception as e:
        print(f"Error reading or processing CSV file '{csv_path}': {e}", file=sys.stderr)
        sys.exit(1)
    
    if df_frbs.empty:
        print("No candidates labeled as FRB (label=1.0) found in the CSV. The new singlepulse file will be empty (only comments).")

    # Regex to extract tcand, dm, and snr from the 'candidate' filename string
    # e.g., ./cand_tstart_..._tcand_1857.7509580_dm_100.00000_snr_12.89000.h5
    pat = re.compile(r"tcand_([0-9.]+)_dm_([0-9.]+)_snr_([0-9.]+)\.h5")
    
    extracted_data = df_frbs['candidate'].str.extract(pat)
    extracted_data.columns = ['time', 'dm', 'snr']
    extracted_data = extracted_data.dropna().astype(float)

    # Create a set of unique identifiers for fast lookup.
    # We round the values to handle potential floating point precision differences
    # between the CSV filename and the .singlepulse file content.
    # DM, SNR, Time are the first three columns in a standard .singlepulse file.
    good_candidate_keys = set()
    for _, row in extracted_data.iterrows():
        # Key: (DM, SNR, Time)
        key = (
            round(row['dm'], 2), 
            round(row['snr'], 2), 
            round(row['time'], 6) # Time often has higher precision
        )
        good_candidate_keys.add(key)

    print(f"Found {len(df_frbs)} candidates labeled as FRB in CSV.")
    print(f"Extracted {len(good_candidate_keys)} unique FRB candidate keys.")

    # 3. Backup the original .singlepulse file
    backup_path = sp_path.with_suffix(sp_path.suffix + '.oldsingpulseold')
    try:
        print(f"Backing up '{sp_path}' to '{backup_path}'")
        shutil.copy(sp_path, backup_path)
    except Exception as e:
        print(f"Error creating backup file: {e}", file=sys.stderr)
        sys.exit(1)

    # 4. Read the original .singlepulse file and write the filtered version
    kept_lines = []
    candidates_matched = 0
    try:
        with open(backup_path, 'r') as f_in:
            for line in f_in:
                line_stripped = line.strip()
                if not line_stripped:
                    continue
                
                if line_stripped.startswith('#'):
                    kept_lines.append(line)
                    continue

                parts = line_stripped.split()
                if len(parts) < 3:
                    continue

                try:
                    dm, snr, time = float(parts[0]), float(parts[1]), float(parts[2])
                    current_key = (round(dm, 2), round(snr, 2), round(time, 6))

                    if current_key in good_candidate_keys:
                        kept_lines.append(line)
                        candidates_matched += 1
                except (ValueError, IndexError):
                    print(f"Warning: Could not parse line: '{line_stripped}'", file=sys.stderr)
                    continue
    except Exception as e:
        print(f"Error reading singlepulse file '{backup_path}': {e}", file=sys.stderr)
        sys.exit(1)

    # 5. Write the new, filtered .singlepulse file
    with open(sp_path, 'w') as f_out:
        f_out.writelines(kept_lines)
        
    print(f"Successfully wrote {len(kept_lines)} lines ({candidates_matched} candidates) to new file: '{sp_path}'")

def main():
    parser = argparse.ArgumentParser(
        description="Filters a PRESTO .singlepulse file based on a 'results_model_z.csv' classification file.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument('directory', type=Path, help="The directory containing the '.singlepulse' file and 'results_model_z.csv'.")
    args = parser.parse_args()

    if not args.directory.is_dir():
        print(f"Error: Provided path '{args.directory}' is not a valid directory.", file=sys.stderr)
        sys.exit(1)

    filter_singlepulse_file(args.directory)

if __name__ == '__main__':
    main()
