#!/usr/bin/env python3
"""
Processes PRESTO '.singlepulse' files from one or more directories,
and saves the results to a 'candidates.csv' file in each directory,
formatted for use with `fetch` and `candmaker`.
"""

import argparse
import pandas as pd
import glob
import warnings
import sys
from pathlib import Path
from typing import Optional

warnings.filterwarnings("ignore")


def find_fil_file(search_dir: Path) -> Optional[Path]:
    """
    Tries to automatically find the .fil file associated with a PRESTO output directory.
    Assumes a directory structure like the one created by Raftaar.py, e.g.:
    - analysis_run/
      - 1_injection_output/injected_file.fil
      - 2_presto_output/  (this is the search_dir)
    """
    injection_dir = search_dir.parent / '1_injection_output'
    if injection_dir.is_dir():
        fil_files = list(injection_dir.glob('*.fil'))
        if fil_files:
            if len(fil_files) > 1:
                print(f"  Warning: Found multiple .fil files in {injection_dir}. Using the first one: {fil_files[0]}")
            return fil_files[0]
    return None


def create_csv(dir_path: Path, fil_file_path: Path, args: argparse.Namespace) -> Optional[Path]:
    """
    Reads .singlepulse files, processes them, and writes candidates.csv.
    Returns the path to the created CSV file, or None if no candidates were found.
    """
    print(f"  Reading .singlepulse files from: {dir_path}")
    file_list = glob.glob(f'{dir_path}/*.singlepulse')
    if not file_list:
        print(f"  Warning: No '.singlepulse' files found in {dir_path}. Skipping.")
        return None

    print(f"  Found {len(file_list)} '.singlepulse' file(s).")

    # Standard PRESTO .singlepulse columns
    columns = ['dm', 'snr', 'stime', 'sample', 'downfact']

    dfs = []
    for file in file_list:
        try:
            data = pd.read_csv(file, comment='#', delim_whitespace=True, header=None)
            if data.shape[1] >= 5:
                dfs.append(data.iloc[:, :5])
            else:
                print(f"  Warning: Skipping file {file} which has fewer than 5 columns.")
        except pd.errors.EmptyDataError:
            print(f"  Warning: Skipping empty file {file}.")

    if not dfs:
        print("  No valid data found in any '.singlepulse' files. Skipping directory.")
        return None

    df = pd.concat(dfs, ignore_index=True)
    df.columns = columns

    # Per user request, no filtering is applied. All candidates are kept.
    print(f"  Found {len(df)} total candidates. No filtering applied.")

    if df.empty:
        print("  No candidates found. Skipping CSV creation.")
        return None

    # Rename 'downfact' to 'width' for fetch/candmaker compatibility.
    # 'width' in this context is the number of samples summed (downfact).
    df.rename(columns={'downfact': 'width'}, inplace=True)

    # Add new required columns
    df['file'] = str(fil_file_path.resolve())
    df['label'] = 1
    df['chan_mask_path'] = ''
    df['num_files'] = 1

    # Select and reorder columns for the final CSV
    # CSV format for candmaker/fetch: file,snr,stime,width,dm,label,chan_mask_path,num_files
    output_columns = ['file', 'snr', 'stime', 'width', 'dm', 'label', 'chan_mask_path', 'num_files']
    df_final = df[output_columns]

    output_path = dir_path / 'candidates.csv'
    df_final.to_csv(output_path, index=False)
    print(f"  Successfully created '{output_path}'")
    return output_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Processes PRESTO outputs within an analysis directory structure to create a candidates.csv file for `fetch`.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        'dirs',
        type=Path,
        nargs='+',
        help='One or more main analysis directories (e.g., analysis_run) to process.'
    )
    parser.add_argument(
        '--fil-file',
        type=Path,
        help='Path to the corresponding .fil file. If provided, this will be used for all analysis directories, overriding automatic detection.'
    )
    parser.add_argument(
        '--master-csv',
        type=Path,
        default=Path('master_candidates.csv'),
        help='Path to save the combined master CSV file.'
    )
    parser.add_argument(
        '--no-master-csv',
        action='store_true',
        help='Do not create a combined master CSV file.'
    )
    args = parser.parse_args()

    all_candidate_files = []

    for analysis_dir in args.dirs:
        if not analysis_dir.is_dir():
            print(f"Error: '{analysis_dir}' is not a valid analysis directory. Skipping.", file=sys.stderr)
            continue

        print(f"\n>>> Processing analysis directory: {analysis_dir}")

        # Define paths based on the expected structure from Raftaar.py
        presto_dir = analysis_dir / '2_presto_output'
        if not presto_dir.is_dir():
            print(f"  Error: PRESTO output directory not found at '{presto_dir}'. Skipping.", file=sys.stderr)
            continue

        # Find the .fil file. If a global one is provided, use it. Otherwise, search.
        fil_file = args.fil_file or find_fil_file(presto_dir)
        if not fil_file or not fil_file.exists():
            print(f"  Error: Could not find a valid .fil file for '{analysis_dir}'. Please provide one using --fil-file.", file=sys.stderr)
            continue
        if args.fil_file is None:
            print(f"  Automatically found .fil file: {fil_file}")

        # The create_csv function expects the directory with .singlepulse files
        csv_path = create_csv(presto_dir, fil_file, args)
        if csv_path:
            all_candidate_files.append(csv_path)

    if not args.no_master_csv:
        if all_candidate_files:
            print(f"\n>>> Combining {len(all_candidate_files)} candidates.csv files into a master file...")
            master_df_list = [pd.read_csv(f) for f in all_candidate_files]
            master_df = pd.concat(master_df_list, ignore_index=True)
            master_df.to_csv(args.master_csv, index=False)
            print(f"Successfully created master file: '{args.master_csv}' with {len(master_df)} candidates.")
        else:
            print("\n>>> No candidate files were generated, so no master CSV file will be created.")