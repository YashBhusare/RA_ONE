#!/usr/bin/env python3
"""
completeness_analyser.py

A script to analyze burst completeness using the Keane & Petroff (2015) methodology.
It reads matched burst data from multiple analysis runs, calculates fluence, and plots
the results in the fluence-width plane to compare completeness definitions.
"""
import argparse
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import matplotlib.gridspec as gridspec
import seaborn as sns
from scipy.interpolate import interp1d
from pathlib import Path

def calculate_fluence(df: pd.DataFrame, sefd: float, n_p: int, bandwidth_hz: float) -> pd.DataFrame:
    """
    Calculates fluence in Jy-s from S/N and width in seconds.
    The radiometer equation for flux density is S = S/N * SEFD / sqrt(n_p * BW * W).
    Fluence is F = S * W.
    Therefore, F = (S/N * SEFD / sqrt(n_p * BW)) * sqrt(W).
    """
    # This constant C converts (S/N * sqrt(W)) to fluence in Jy-s.
    C = sefd / np.sqrt(n_p * bandwidth_hz)
    df['fluence_jys'] = C * df['det_snr'] * np.sqrt(df['det_width_s'])
    return df

def plot_keane_petroff_style1(df: pd.DataFrame, sn_thresh: float, sefd: float, bandwidth_mhz: float, outdir: Path, avg_thresholds: dict):
    """
    Generates a publication-quality completeness plot (Style 1: Different Markers).
    """
    plot_df = df.copy()
    if plot_df.empty:
        print("No detected bursts to plot.")
        return

    # --- Setup and Calculations ---
    n_p = 2
    bandwidth_hz = bandwidth_mhz * 1e6
    w_max_s = plot_df['det_width_s'].max()
    if pd.isna(w_max_s) or w_max_s <= 0:
        print("Warning: Maximum detected width is invalid. Cannot determine completeness threshold.")
        return

    C = sefd / np.sqrt(n_p * bandwidth_hz)
    fluence_const_thresh_jys = C * sn_thresh * np.sqrt(w_max_s)

    # --- Plotting ---
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax = plt.subplots(figsize=(6, 5))

    green_bursts = plot_df[plot_df['color'] == 'green']
    gray_bursts = plot_df[plot_df['color'] == 'gray']
    red_bursts = plot_df[plot_df['color'] == 'red']

    ax.scatter(
        gray_bursts['det_width_ms'], gray_bursts['fluence_jyms'],
        color='gray', label=f'Incomplete ({len(gray_bursts)})',
        alpha=0.5, s=35, zorder=5, marker='x'
    )
    ax.scatter(
        red_bursts['det_width_ms'], red_bursts['fluence_jyms'],
        color='red', label=f'Inconsistent ({len(red_bursts)})',
        alpha=0.7, s=45, zorder=10, marker='+'
    )
    ax.scatter(
        green_bursts['det_width_ms'], green_bursts['fluence_jyms'],
        color='green', label=f'Complete ({len(green_bursts)})',
        alpha=0.8, edgecolors='k', s=50, zorder=15, marker='o'
    )

    # Plot threshold lines
    w_min_s = plot_df['det_width_s'].min()
    if pd.isna(w_min_s) or w_min_s <= 0:
        w_min_s = 1e-4 # a small default if min is invalid
    w_range_s = np.logspace(np.log10(w_min_s), np.log10(w_max_s), 100)

    fluence_sn_thresh_jys = C * sn_thresh * np.sqrt(w_range_s)
    ax.plot(
        w_range_s * 1000, fluence_sn_thresh_jys * 1000, color='black',
        linestyle='--', linewidth=1.5, label=f'S/N = {sn_thresh} Threshold'
    )

    ax.axhline(
        fluence_const_thresh_jys * 1000,
        color='darkorange', linestyle='-', linewidth=2,
        label=f'K&P15 Fluence Threshold'
    )

    if avg_thresholds:
        g1_widths_ms = sorted(avg_thresholds.keys())
        g1_widths_s = np.array(g1_widths_ms) / 1000.0
        g1_snrs = np.array([avg_thresholds[w] for w in g1_widths_ms])
        g1_fluence_jys = C * g1_snrs * np.sqrt(g1_widths_s)
        ax.plot(
            g1_widths_ms, g1_fluence_jys * 1000, color='purple',
            linestyle=':', marker='o', markersize=4, linewidth=2,
            label='Pipeline 90% Completeness'
        )

    ax.set_xlabel('Detected Pulse Width (ms)', fontsize=14)
    ax.set_ylabel('Fluence (Jy ms)', fontsize=14)
    ax.tick_params(axis='both', which='major', labelsize=12)
    ax.set_title('Burst Completeness Comparison', fontsize=14)
    ax.legend(loc='upper left', fontsize='small', frameon=True)
    ax.grid(True, which="both", ls=":", lw=0.7)

    plot_path = outdir / 'completeness_keane_petroff_style1.png'
    plt.tight_layout()
    plt.savefig(plot_path, dpi=300)
    plt.close(fig)
    print(f"Saved Style 1 completeness plot to: {plot_path}")

def plot_keane_petroff_style2(df: pd.DataFrame, sn_thresh: float, sefd: float, bandwidth_mhz: float, outdir: Path, avg_thresholds: dict):
    """
    Generates a completeness plot with marginal histograms (Style 2).
    """
    plot_df = df.copy()
    if plot_df.empty:
        print("No detected bursts to plot.")
        return

    # --- Setup and Calculations ---
    n_p = 2
    bandwidth_hz = bandwidth_mhz * 1e6
    w_max_s = plot_df['det_width_s'].max()
    if pd.isna(w_max_s) or w_max_s <= 0:
        print("Warning: Maximum detected width is invalid. Cannot determine completeness threshold.")
        return

    C = sefd / np.sqrt(n_p * bandwidth_hz)
    fluence_const_thresh_jys = C * sn_thresh * np.sqrt(w_max_s)

    # --- Plotting Setup ---
    plt.style.use('seaborn-v0_8-whitegrid')
    fig = plt.figure(figsize=(8, 8)) # Larger to accommodate histograms
    gs = gridspec.GridSpec(2, 2, width_ratios=[4, 1], height_ratios=[1, 4], hspace=0.05, wspace=0.05)

    ax_scatter = fig.add_subplot(gs[1, 0])
    ax_hist_x = fig.add_subplot(gs[0, 0], sharex=ax_scatter)
    ax_hist_y = fig.add_subplot(gs[1, 1], sharey=ax_scatter)

    # Remove labels from marginal plots
    ax_hist_x.tick_params(axis="x", labelbottom=False)
    ax_hist_y.tick_params(axis="y", labelleft=False)

    # --- Scatter Plot (main plot) ---
    green_bursts = plot_df[plot_df['color'] == 'green']
    gray_bursts = plot_df[plot_df['color'] == 'gray']
    red_bursts = plot_df[plot_df['color'] == 'red']

    ax_scatter.scatter(gray_bursts['det_width_ms'], gray_bursts['fluence_jyms'], color='gray', label=f'Incomplete ({len(gray_bursts)})', alpha=0.6, s=30, zorder=5)
    ax_scatter.scatter(red_bursts['det_width_ms'], red_bursts['fluence_jyms'], color='red', label=f'Inconsistent ({len(red_bursts)})', alpha=0.7, edgecolors='k', s=40, zorder=10)
    ax_scatter.scatter(green_bursts['det_width_ms'], green_bursts['fluence_jyms'], color='green', label=f'Complete ({len(green_bursts)})', alpha=0.8, edgecolors='k', s=50, zorder=15)

    # Plot threshold lines
    w_min_s = plot_df['det_width_s'].min()
    if pd.isna(w_min_s) or w_min_s <= 0:
        w_min_s = 1e-4
    w_range_s = np.logspace(np.log10(w_min_s), np.log10(w_max_s), 100)

    fluence_sn_thresh_jys = C * sn_thresh * np.sqrt(w_range_s)
    ax_scatter.plot(w_range_s * 1000, fluence_sn_thresh_jys * 1000, color='black', linestyle='--', linewidth=1.5, label=f'S/N = {sn_thresh} Threshold')
    ax_scatter.axhline(fluence_const_thresh_jys * 1000, color='darkorange', linestyle='-', linewidth=2, label=f'K&P15 Fluence Threshold')
    if avg_thresholds:
        g1_widths_ms = sorted(avg_thresholds.keys())
        g1_widths_s = np.array(g1_widths_ms) / 1000.0
        g1_snrs = np.array([avg_thresholds[w] for w in g1_widths_ms])
        g1_fluence_jys = C * g1_snrs * np.sqrt(g1_widths_s)
        ax_scatter.plot(g1_widths_ms, g1_fluence_jys * 1000, color='purple', linestyle=':', marker='o', markersize=4, linewidth=2, label='Pipeline 90% Completeness')

    ax_scatter.set_xlabel('Detected Pulse Width (ms)', fontsize=14)
    ax_scatter.set_ylabel('Fluence (Jy ms)', fontsize=14)
    ax_scatter.tick_params(axis='both', which='major', labelsize=12)
    ax_scatter.legend(loc='upper left', fontsize='small', frameon=True)
    ax_scatter.grid(True, which="both", ls=":", lw=0.7)

    # --- Marginal Histograms ---
    # X-axis histogram (width)
    x_bins = np.linspace(plot_df['det_width_ms'].min(), plot_df['det_width_ms'].max(), 50)
    ax_hist_x.hist([gray_bursts['det_width_ms'], red_bursts['det_width_ms'], green_bursts['det_width_ms']], bins=x_bins, color=['gray', 'red', 'green'], alpha=0.7, histtype='stepfilled', stacked=True)
    ax_hist_x.set_ylabel('Count')
    ax_hist_x.set_title('Burst Completeness Comparison (Style 2)', fontsize=14)

    # Y-axis histogram (fluence)
    min_fluence = plot_df['fluence_jyms'].min()
    if min_fluence <= 0: min_fluence = 1e-3 # Handle non-positive values for log scale
    y_bins = np.logspace(np.log10(min_fluence), np.log10(plot_df['fluence_jyms'].max()), 50)
    ax_hist_y.hist([gray_bursts['fluence_jyms'], red_bursts['fluence_jyms'], green_bursts['fluence_jyms']], bins=y_bins, color=['gray', 'red', 'green'], alpha=0.7, orientation='horizontal', histtype='stepfilled', stacked=True)
    ax_hist_y.set_xlabel('Count')
    ax_hist_y.set_yscale('log')
    ax_scatter.set_yscale('log') # Match scale

    plot_path = outdir / 'completeness_keane_petroff_style2.png'
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved Style 2 completeness plot to: {plot_path}")

def plot_violin_distributions(df: pd.DataFrame, outdir: Path):
    """
    Generates violin plots to show the distribution of key parameters for each
    completeness category.
    """
    if df.empty:
        print("No data for violin plots.")
        return

    plt.style.use('seaborn-v0_8-whitegrid')
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    plot_order = ['gray', 'red', 'green']
    palette = {'gray': 'gray', 'red': 'red', 'green': 'green'}

    # Plot 1: Fluence Distribution
    sns.violinplot(ax=axes[0], data=df, x='color', y='fluence_jyms', order=plot_order, palette=palette, cut=0)
    axes[0].set_title('Fluence Distribution by Category', fontsize=14)
    axes[0].set_xlabel('Completeness Category', fontsize=12)
    axes[0].set_ylabel('Fluence (Jy ms)', fontsize=12)
    axes[0].set_xticklabels(['Incomplete', 'Inconsistent', 'Complete'])
    axes[0].set_yscale('log') # Fluence often spans orders of magnitude

    # Plot 2: Detected SNR Distribution
    sns.violinplot(ax=axes[1], data=df, x='color', y='det_snr', order=plot_order, palette=palette, cut=0)
    axes[1].set_title('Detected SNR Distribution by Category', fontsize=14)
    axes[1].set_xlabel('Completeness Category', fontsize=12)
    axes[1].set_ylabel('Detected S/N', fontsize=12)
    axes[1].set_xticklabels(['Incomplete', 'Inconsistent', 'Complete'])

    # Plot 3: Detected Width Distribution
    sns.violinplot(ax=axes[2], data=df, x='color', y='det_width_ms', order=plot_order, palette=palette, cut=0)
    axes[2].set_title('Detected Width Distribution by Category', fontsize=14)
    axes[2].set_xlabel('Completeness Category', fontsize=12)
    axes[2].set_ylabel('Detected Width (ms)', fontsize=12)
    axes[2].set_xticklabels(['Incomplete', 'Inconsistent', 'Complete'])

    fig.suptitle('Distribution of Burst Properties by Completeness Category', fontsize=16, y=1.02)
    plt.tight_layout(rect=[0, 0, 1, 0.98])
    
    plot_path = outdir / 'completeness_violin_plots.png'
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved violin plots to: {plot_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Analyze and compare burst completeness from multiple analysis runs using the Keane & Petroff (2015) method.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        'analysis_dirs', type=Path, nargs='+',
        help='One or more analysis run directories containing "3_analysis_results".'
    )
    parser.add_argument(
        '--output-dir', type=Path, default=Path('.'),
        help="Output directory for the plot."
    )
    parser.add_argument(
        '--bandwidth-mhz', type=float, required=True,
        help="Observation bandwidth in MHz."
    )
    parser.add_argument(
        '--sn-thresh', type=float, default=6.0,
        help="S/N threshold for the Keane & Petroff analysis."
    )
    parser.add_argument(
        '--sefd', type=float, default=100.0,
        help="System Equivalent Flux Density (SEFD) in Janskys for the telescope used."
    )
    args = parser.parse_args()
    args.output_dir.mkdir(exist_ok=True)

    # --- 1. Aggregate data from all specified directories ---
    all_bursts_dfs, all_thresholds = [], {}
    for run_dir in args.analysis_dirs:
        csv_path = run_dir / '3_analysis_results' / 'injections_matched.csv'
        json_path = run_dir / '3_analysis_results' / '90_thresholds.json'

        if csv_path.exists():
            print(f"Loading data from: {csv_path}")
            all_bursts_dfs.append(pd.read_csv(csv_path))
        else:
            print(f"Warning: '{csv_path}' not found. Skipping.")

        if json_path.exists():
            with open(json_path, 'r') as f:
                thresholds = json.load(f)
                for width_ms, snr_thresh in thresholds.items():
                    if snr_thresh is not None:
                        width_key = float(width_ms)
                        if width_key not in all_thresholds:
                            all_thresholds[width_key] = []
                        all_thresholds[width_key].append(snr_thresh)

    if not all_bursts_dfs:
        parser.error("No 'injections_matched.csv' files found in the provided directories.")

    combined_df = pd.concat(all_bursts_dfs, ignore_index=True)
    avg_thresholds = {width: np.mean(snrs) for width, snrs in all_thresholds.items()}

    # --- 2. Filter for detected bursts and calculate completeness ---
    detected_df = combined_df[combined_df['is_detected']].copy()
    detected_df.dropna(subset=['det_snr', 'det_width_s', 'expected_det_snr', 'width_ms_group'], inplace=True)
    if detected_df.empty:
        print("No valid detected bursts found across all runs. Exiting.")
        return

    # Method 1: Keane & Petroff completeness
    detected_df = calculate_fluence(detected_df, args.sefd, 2, args.bandwidth_mhz * 1e6)
    w_max_s = detected_df['det_width_s'].max()
    C = args.sefd / np.sqrt(2 * args.bandwidth_mhz * 1e6)
    fluence_const_thresh_jys = C * args.sn_thresh * np.sqrt(w_max_s)
    detected_df['is_complete_kp'] = detected_df['fluence_jys'] >= fluence_const_thresh_jys

    # Method 2: G_one 90% threshold completeness, evaluated at the DETECTED width.
    # To do this, we create an interpolation function for the pipeline's threshold curve.
    g1_widths_ms = []
    g1_fluence_thresh_jys = []
    if avg_thresholds:
        # Sort by width to ensure monotonic x-axis for interpolation
        for width_ms in sorted(avg_thresholds.keys()):
            snr_thresh = avg_thresholds[width_ms]
            if snr_thresh is not None:
                width_s = width_ms / 1000.0
                # Fluence_thresh = C * SNR_thresh * sqrt(Width)
                fluence_thresh = C * snr_thresh * np.sqrt(width_s)
                g1_widths_ms.append(width_ms)
                g1_fluence_thresh_jys.append(fluence_thresh)

    # Add detected width in ms for interpolation and plotting
    detected_df['det_width_ms'] = detected_df['det_width_s'] * 1000.0

    if len(g1_widths_ms) >= 2:
        # Create interpolation function from width (ms) to fluence threshold (Jy-s)
        fluence_thresh_interpolator = interp1d(
            g1_widths_ms,
            g1_fluence_thresh_jys,
            kind='linear',  # Linear is safer than cubic for extrapolation
            bounds_error=False,
            fill_value="extrapolate"
        )
        # Get the threshold for each detected burst using its detected width
        fluence_threshold_g1_jys = fluence_thresh_interpolator(detected_df['det_width_ms'])
        # A burst is complete if its measured fluence is >= the interpolated threshold for its detected width.
        detected_df['is_complete_g1'] = detected_df['fluence_jys'] >= fluence_threshold_g1_jys
    else:
        # If not enough data to interpolate, mark all as incomplete for this method
        print("Warning: Not enough data points (<2) to build G_one completeness threshold. Marking all as incomplete for this method.")
        detected_df['is_complete_g1'] = False

    # --- 3. Assign colors for comparative plot ---
    conditions = [
        (detected_df['is_complete_kp'] & detected_df['is_complete_g1']),
        (~detected_df['is_complete_kp'] & ~detected_df['is_complete_g1']),
    ]
    choices = ['green', 'gray']
    detected_df['color'] = np.select(conditions, choices, default='red')

    # --- 3.5. Analyze and report on misclassifications ---
    print("\n--- Misclassification Analysis ---")
    total_detected = len(detected_df)
    
    complete_both = (detected_df['color'] == 'green').sum()
    incomplete_both = (detected_df['color'] == 'gray').sum()
    inconsistent = (detected_df['color'] == 'red').sum()

    kp_complete_g1_incomplete = (detected_df['is_complete_kp'] & ~detected_df['is_complete_g1']).sum()
    kp_incomplete_g1_complete = (~detected_df['is_complete_kp'] & detected_df['is_complete_g1']).sum()

    print(f"Total detected and valid bursts analyzed: {total_detected}")
    print(f"  - Complete by both methods (Green): {complete_both} ({complete_both/total_detected:.1%})")
    print(f"  - Incomplete by both methods (Gray): {incomplete_both} ({incomplete_both/total_detected:.1%})")
    print(f"  - Inconsistent (Red): {inconsistent} ({inconsistent/total_detected:.1%})")
    
    if inconsistent > 0:
        print("\nBreakdown of inconsistent classifications:")
        print(f"  - Classified as COMPLETE by Keane & Petroff, but INCOMPLETE by G_one: {kp_complete_g1_incomplete}")
        print(f"  - Classified as INCOMPLETE by Keane & Petroff, but COMPLETE by G_one: {kp_incomplete_g1_complete}")

    # Add columns for plotting
    detected_df['fluence_jyms'] = detected_df['fluence_jys'] * 1000

    # --- 4. Generate plots ---
    print("\n--- Generating Keane & Petroff completeness plot (Style 1: Markers) ---")
    plot_keane_petroff_style1(detected_df, args.sn_thresh, args.sefd, args.bandwidth_mhz, args.output_dir, avg_thresholds)

    print("\n--- Generating Keane & Petroff completeness plot (Style 2: Histograms) ---")
    plot_keane_petroff_style2(detected_df, args.sn_thresh, args.sefd, args.bandwidth_mhz, args.output_dir, avg_thresholds)

    print("\n--- Generating violin plots for parameter distributions ---")
    plot_violin_distributions(detected_df, args.output_dir)

    print("\nAnalysis complete.")
    print("To generate the combined completeness vs. SNR plot, please run 'plot_combined_completeness.py' on your analysis directories.")
if __name__ == '__main__':
    main()
