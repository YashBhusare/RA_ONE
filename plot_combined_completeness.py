#!/usr/bin/env python3
"""
plot_combined_completeness.py

A script to aggregate completeness data from multiple analysis runs
and generate a combined completeness plot.
"""
import argparse
import json
from pathlib import Path
import sys

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.patheffects as path_effects
from scipy.interpolate import interp1d
from scipy.optimize import brentq

def main():
    parser = argparse.ArgumentParser(
        description="Combine completeness data from multiple analysis runs and plot the results.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        'analysis_dirs',
        type=Path,
        nargs='+',
        help='One or more analysis run directories (e.g., analysis_run_1_ms, analysis_run_3_ms).'
    )
    parser.add_argument(
        '--output',
        '-o',
        type=Path,
        default='combined_completeness.png',
        help='Output filename for the combined plot.'
    )
    parser.add_argument(
        '--zoom-snr',
        type=float,
        default=12.0,
        help='Upper SNR limit for a zoomed version of the plot.'
    )
    args = parser.parse_args()

    all_completeness_dfs = []
    all_thresholds = {} # Dict to store lists of thresholds per width, e.g., {'7.5': [snr1, snr2, ...]}

    print("--- Loading data from analysis directories ---")
    for run_dir in args.analysis_dirs:
        completeness_csv = run_dir / '3_analysis_results' / 'completeness_data.csv'
        thresholds_json = run_dir / '3_analysis_results' / '90_thresholds.json'

        if not completeness_csv.exists():
            print(f"Warning: '{completeness_csv}' not found. Skipping directory '{run_dir}'.", file=sys.stderr)
            continue

        print(f"Loading data from: {run_dir}")
        df = pd.read_csv(completeness_csv)
        all_completeness_dfs.append(df)

        if thresholds_json.exists():
            with open(thresholds_json, 'r') as f:
                thresholds = json.load(f)
                for width_ms, snr_thresh in thresholds.items():
                    if snr_thresh is not None:
                        width_key = float(width_ms)
                        if width_key not in all_thresholds:
                            all_thresholds[width_key] = []
                        all_thresholds[width_key].append(snr_thresh)
        else:
            print(f"Warning: '{thresholds_json}' not found in '{run_dir}'.", file=sys.stderr)

    if not all_completeness_dfs:
        print("Error: No completeness data found in any of the specified directories. Exiting.", file=sys.stderr)
        sys.exit(1)

    print("\n--- Aggregating data ---")
    combined_df = pd.concat(all_completeness_dfs, ignore_index=True)
    combined_df['width_ms'] = combined_df['width_ms'].astype(float)

    aggregated_df = combined_df.groupby(['width_ms', 'snr_bin_center']).agg(
        n_detected_total=('n_detected', 'sum'),
        n_total_total=('n_total', 'sum')
    ).reset_index()
    aggregated_df['completeness'] = aggregated_df['n_detected_total'] / aggregated_df['n_total_total']
    
    avg_thresholds = {width: np.mean(snrs) for width, snrs in all_thresholds.items()}
    print("Average 90% SNR thresholds:")
    for width, thresh in sorted(avg_thresholds.items()):
        print(f"  {width} ms: {thresh:.2f}")

    print("\n--- Generating plot ---")
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax = plt.subplots(figsize=(6, 5))

    unique_widths = sorted(aggregated_df['width_ms'].unique())
    # Use a qualitative palette for distinct colors
    colors = sns.color_palette('tab10', n_colors=len(unique_widths))
    width_color_map = dict(zip(unique_widths, colors))

    all_95_thresholds = {}
    all_90_thresholds = {}

    for width in unique_widths:
        width_df = aggregated_df[aggregated_df['width_ms'] == width].copy()
        color = width_color_map[width]

        # Prepare data for fitting
        x_data = width_df['snr_bin_center']
        y_data = width_df['completeness']

        valid_mask = ~np.isnan(x_data) & ~np.isnan(y_data)
        if valid_mask.sum() < 4:
            print(f"Warning: Not enough data points ({valid_mask.sum()}) for cubic interpolation for width {width} ms. Skipping.", file=sys.stderr)
            continue # Not enough points to fit

        try:
            # --- 1. Cubic Spline Interpolation ---
            x_fit = x_data[valid_mask].to_numpy()
            y_fit = y_data[valid_mask].to_numpy()

            # Ensure data is sorted by x-values for interpolation
            sort_idx = np.argsort(x_fit)
            x_fit = x_fit[sort_idx]
            y_fit = y_fit[sort_idx]

            # To prevent oscillations, ensure the y-values are monotonically increasing
            y_fit_monotonic = np.maximum.accumulate(y_fit)

            # Create the interpolation function
            interp_func = interp1d(x_fit, y_fit_monotonic, kind='cubic', bounds_error=False, fill_value=(y_fit_monotonic[0], y_fit_monotonic[-1]))

            # Generate smooth x values for plotting the fitted curve
            x_smooth = np.linspace(x_fit.min(), x_fit.max(), 300)
            y_smooth = interp_func(x_smooth)
            # Clip to [0, 1] to avoid overshoots from the cubic spline
            y_smooth = np.clip(y_smooth, 0, 1)

            label_text = f'{width:.2f} ms' # Default label if threshold not found

            # --- 2. Find 90% Threshold using Root Finding ---
            x_90 = None
            if y_fit_monotonic[-1] >= 0.9:
                try:
                    # Define the function to find the root of: f(x) = interp(x) - 0.9
                    func_to_solve = lambda x: interp_func(x) - 0.9
                    # Find the interval to search for the root
                    first_above_90_idx = np.where(y_fit_monotonic >= 0.9)[0][0]
                    if first_above_90_idx > 0:
                        a, b = x_fit[first_above_90_idx - 1], x_fit[first_above_90_idx]
                        x_90 = brentq(func_to_solve, a, b)
                    else: # The very first point is already >= 0.9
                        x_90 = x_fit[0]
                except (IndexError, ValueError) as e:
                    print(f"Warning: Could not find 90% threshold for width {width} ms via root finding. Reason: {e}", file=sys.stderr)

            # --- 3. Find 95% Threshold for printing ---
            x_95 = None
            if y_fit_monotonic[-1] >= 0.95:
                try:
                    func_to_solve_95 = lambda x: interp_func(x) - 0.95
                    first_above_95_idx = np.where(y_fit_monotonic >= 0.95)[0][0]
                    if first_above_95_idx > 0:
                        a, b = x_fit[first_above_95_idx - 1], x_fit[first_above_95_idx]
                        x_95 = brentq(func_to_solve_95, a, b)
                    else:
                        x_95 = x_fit[0]
                except (IndexError, ValueError):
                    pass # Fail silently

            if x_95 is not None:
                all_95_thresholds[width] = x_95

            if x_90 is not None:
                all_90_thresholds[width] = x_90
                label_text = f"{width:.2f} ms: {x_90:.1f}"
                ax.axvline(x=x_90, color=color, linestyle='--', alpha=0.8, linewidth=1.5)

            # Plot the fitted sigmoid curve
            ax.plot(x_smooth, y_smooth, linestyle='-', label=label_text, color=color, linewidth=2.5)

            strong_points = width_df[width_df['n_total_total'] >= 10]
            weak_points = width_df[width_df['n_total_total'] < 10]

            ax.scatter(
                strong_points['snr_bin_center'], strong_points['completeness'],
                color=color, alpha=0.9, s=50, marker='o', zorder=5, label=None
            )
            ax.scatter(
                weak_points['snr_bin_center'], weak_points['completeness'],
                color=color, alpha=0.4, s=30, marker='o', zorder=5, label=None
            )
        except (RuntimeError, ValueError) as e:
            print(f"Warning: Could not create interpolation for width {width} ms. Plotting points only. Reason: {e}", file=sys.stderr)
            # Plot raw points even if fit fails, but with a different marker
            ax.scatter(width_df['snr_bin_center'], width_df['completeness'], color=color, marker='x', s=40, label=f'{width:.2f} ms (interpolation failed)')
            continue

    # Add horizontal line for the 90% completeness level
    ax.axhline(y=0.9, color='gray', linestyle=':', alpha=0.9, zorder=0, linewidth=2)

    ax.set_xlabel('Injected SNR', fontsize=14)
    ax.set_ylabel('Completeness', fontsize=14)
    ax.set_ylim(-0.05, 1.05)
    #ax.set_title('Combined Pipeline Completeness by Pulse Width')
    ax.legend(title='90% completeness\nthreshold.', fontsize='small')
    ax.grid(True, linestyle=':', alpha=0.7)

    print("\nCalculated 95% SNR thresholds:")
    if all_95_thresholds:
        for width, thresh in sorted(all_95_thresholds.items()):
            print(f"  {width:.2f} ms: {thresh:.2f}")
    else:
        print("  None could be calculated.")

    plt.tight_layout()
    plt.savefig(args.output)
    print(f"Saved combined plot to '{args.output}'")

    # --- Create publication-quality zoomed plot ---
    ax.set_xlim(2, args.zoom_snr)
    #ax.set_title(f'Completeness for differnt widths')
    zoomed_output_path = args.output.with_name(f"{args.output.stem}_zoomed{args.output.suffix}")
    plt.savefig(zoomed_output_path, dpi=300)
    print(f"Saved zoomed plot to '{zoomed_output_path}'")
    plt.close(fig)

if __name__ == '__main__':
    main()
