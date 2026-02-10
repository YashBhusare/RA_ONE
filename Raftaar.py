#!/usr/bin/env python3
"""
Raftaar.py

A wrapper script to automate the injection and recovery analysis workflow.
This script performs the following steps:
1. Runs Ra_one.py to inject simulated pulses into a filterbank file.
2. Runs the PRESTO pipeline (prepdata + single_pulse_search.py) to search for pulses.
3. Runs G_one.py to compare the injected pulses with the detected candidates and generate analysis plots.
"""

import argparse
import os
import subprocess
import sys
import json
from pathlib import Path


def run_command(cmd: list, cwd: Path = None):
    """
    Executes a command, prints it, and checks for errors.
    """
    cmd_str = ' '.join(map(str, cmd))
    print(f"\n--> Running command: {cmd_str}")
    if cwd:
        print(f"    in directory: {cwd}")

    try:
        # Using capture_output=True to hide command output unless there's an error
        result = subprocess.run(
            cmd,
            check=True,            capture_output=False,
            text=True,
            cwd=cwd
        )
        print("--> Command successful.")
        # For more verbose output, uncomment the following line
        # print(result.stdout)
    except FileNotFoundError:
        print(f"--> ERROR: Command not found: '{cmd[0]}'. Is it in your PATH?", file=sys.stderr)
        sys.exit(1)
    except subprocess.CalledProcessError as e:
        print(f"--> ERROR: Command failed with exit code {e.returncode}", file=sys.stderr)
        print("\n--- STDOUT ---", file=sys.stderr)
        print(e.stdout, file=sys.stderr)
        print("\n--- STDERR ---", file=sys.stderr)
        print(e.stderr, file=sys.stderr)
        sys.exit(1)

def main():
    parser = argparse.ArgumentParser(
        description="Automate the Ra_one -> PRESTO -> G_one analysis pipeline.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        'input_fil',
        type=Path,
        help='Path to the input filterbank file to inject pulses into.'
    )
    parser.add_argument(
        '--output-dir',
        type=Path,
        default=Path('./analysis_run'),
        help='Main directory to store all outputs.'
    )
    parser.add_argument(
        '--n-pulses',
        type=int,
        default=600,
        help='Number of pulses to inject (passed to Ra_one.py).'
    )
    parser.add_argument(
        '--sps-threshold',
        type=float,
        default=6.0,
        help='SNR threshold for `single_pulse_search.py` (-t flag).'
    )
    parser.add_argument(
        '--max-width-ms',
        type=float,
        default=12,
        help='Maximum pulse width in milliseconds for `single_pulse_search.py` (-m flag).'
    )
    parser.add_argument(
        '--config',
        type=Path,
        default=SCRIPT_DIR / 'injection_config.json',
        help='Path to the injection_config.json file for Ra_one.py.'
    )
    parser.add_argument(
        '--downsamp',
        type=int,
        default=1,
        help='Downsampling factor for `prepdata` (-downsamp flag).'
    )
    parser.add_argument(
        '--no-visualize',
        action='store_true',
        help='Disable the --visualize flag for G_one.py.'
    )
    parser.add_argument(
        '--start-from',
        choices=['injection', 'prepdata', 'sps', 'analysis'],
        default='injection',
        help='The step to start the pipeline from. "sps" will skip injection and prepdata.'
    )

    args = parser.parse_args()

    # --- Setup directories and check files ---
    args.output_dir.mkdir(exist_ok=True)

    if not args.input_fil.exists():
        print(f"ERROR: Input file not found: {args.input_fil}", file=sys.stderr)
        sys.exit(1)

    if not args.config.exists():
        print(f"ERROR: Config file not found: {args.config}", file=sys.stderr)
        sys.exit(1)

    # =========================================================================
    # Step 0: Load and validate config for DM search
    # =========================================================================
    print("="*50)
    print("Step 0: Validating configuration")
    print("="*50)
    print(f"--> Loading DM configuration from: {args.config}")
    with open(args.config, 'r') as f:
        config_data = json.load(f)

    dm_config = config_data.get('dm', {})
    if dm_config.get('mode') != 'fixed':
        print(
            f"\nERROR: This script requires a 'fixed' DM mode in '{args.config}'.\n"
            f"The current mode is '{dm_config.get('mode', 'not specified')}'. "
            "The PRESTO `prepdata` step is configured to search at a single DM.",
            file=sys.stderr
        )
        sys.exit(1)

    dm_search_value = dm_config.get('value')
    if dm_search_value is None:
        print(f"ERROR: 'value' for 'dm' not found in '{args.config}' even though mode is 'fixed'.", file=sys.stderr)
        sys.exit(1)
    print(f"--> Using fixed DM search value from config: {dm_search_value}")

    # --- Define file paths ---
    injection_outdir = args.output_dir / '1_injection_output'
    presto_outdir = args.output_dir / '2_presto_output'
    analysis_outdir = args.output_dir / '3_analysis_results'
    master_log = args.output_dir / 'master_injection.log'

    # =========================================================================
    # Step 1: Run Ra_one.py to inject pulses
    # =========================================================================
    if args.start_from == 'injection':
        print("="*50)
        print("Step 1: Injecting pulses with Ra_one.py")
        print("="*50)
        injection_outdir.mkdir(exist_ok=True)
        ra_one_cmd = [
            'ra-one',
            '--filepath', str(args.input_fil.with_suffix('')), # ra-one expects path without extension
            '--log', str(master_log),
            '--o', str(injection_outdir),
            '--n', str(args.n_pulses),
            '--nf', '1',
            '--spilt', '1',
            '--config', str(args.config)
        ]
        run_command(ra_one_cmd)
    else:
        print("="*50)
        print("Step 1: Skipping pulse injection.")
        print("="*50)

    # --- Locate injection files (needed by all subsequent steps) ---
    print("\n--> Locating injection files...")
    injected_fils = list(injection_outdir.glob('*.fil'))
    if not injected_fils:
        print(f"ERROR: No .fil file found in {injection_outdir}. Cannot proceed.", file=sys.stderr)
        sys.exit(1)
    injected_fil_path = injected_fils[0]
    injected_injinf_path = injected_fil_path.with_suffix('.injinf')
    if not injected_injinf_path.exists():
        print(f"ERROR: Could not find companion .injinf file for {injected_fil_path}", file=sys.stderr)
        sys.exit(1)
    print(f"--> Using injected file: {injected_fil_path}")
    print(f"--> Using injection info: {injected_injinf_path}")

    # =========================================================================
    # Step 2: Run PRESTO search
    # =========================================================================
    presto_basename = presto_outdir / f"presto_search_{injected_fil_path.stem}"
    dat_file = presto_basename.with_suffix('.dat')

    # --- Sub-step 2a: prepdata ---
    if args.start_from in ['injection', 'prepdata']:
        print("="*50)
        print("Step 2a: De-dispersing data with prepdata")
        print("="*50)
        presto_outdir.mkdir(exist_ok=True)
        prepdata_cmd = [
            'prepdata',
            '-nobary',
            '-o', str(presto_basename),
            '-dm', str(dm_search_value),
            '-downsamp', str(args.downsamp),
            str(injected_fil_path)
        ]
        run_command(prepdata_cmd)
    else:
        print("="*50)
        print("Step 2a: Skipping prepdata.")
        print("="*50)

    # --- Sub-step 2b: single_pulse_search.py ---
    if args.start_from in ['injection', 'prepdata', 'sps']:
        print("="*50)
        print("Step 2b: Searching for pulses with single_pulse_search.py")
        print("="*50)
        if not dat_file.exists():
            print(f"ERROR: Cannot run single_pulse_search.py because .dat file is missing: {dat_file}", file=sys.stderr)
            print("       Please run the 'prepdata' step first.", file=sys.stderr)
            sys.exit(1)

        # Convert max width from ms to seconds for single_pulse_search.py
        max_width_s = args.max_width_ms / 1000.0

        sps_cmd = ['single_pulse_search.py', '-t', str(args.sps_threshold), '-m', str(max_width_s), '-b', dat_file.name]
        run_command(sps_cmd, cwd=presto_outdir)
    else:
        print("="*50)
        print("Step 2b: Skipping single_pulse_search.py.")
        print("="*50)

    # --- Locate detection file (needed for analysis) ---
    print("\n--> Locating detection files...")
    detection_file_path = dat_file.with_suffix('.singlepulse')
    if not detection_file_path.exists():
        print(f"ERROR: No .singlepulse file found at {detection_file_path}. Cannot proceed to analysis.", file=sys.stderr)
        sys.exit(1)
    print(f"--> Using detection file: {detection_file_path}")

    # =========================================================================
    # Step 3: Run G_one.py for analysis
    # =========================================================================
    if args.start_from in ['injection', 'prepdata', 'sps', 'analysis']:
        print("="*50)
        print("Step 3: Analyzing recovery with G_one.py")
        print("="*50)
        analysis_outdir.mkdir(exist_ok=True)
        g_one_cmd = [
            'g-one',
            '--inject-file', str(injected_injinf_path),
            '--detection-file', str(detection_file_path),
            '--det-format', 'presto',
            '--outdir', str(analysis_outdir),
            '--fil-file', str(injected_fil_path),
            '--prepdata-downsamp', str(args.downsamp),
        ]
        if not args.no_visualize:
            g_one_cmd.append('--visualize')
        run_command(g_one_cmd)

    print("\n" + "="*50)
    print("Analysis pipeline finished successfully!")
    print(f"Results are in: {analysis_outdir}")
    print("="*50)


if __name__ == '__main__':
    main()
