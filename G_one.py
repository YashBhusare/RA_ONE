#!/usr/bin/env python3
"""
G_one_v2.py
A script to analyze the performance of single-pulse search pipelines by
comparing detected candidates against a catalog of injected pulses.
It matches detections to injections, calculates recovery statistics,
and generates diagnostic plots and reports.
"""
import argparse
import os
import sys
import json
from typing import Optional, Dict, Any, List

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import interpolate
from scipy.optimize import brentq
from scipy.stats import linregress
from pathlib import Path

try:
    from sigpyproc.readers import FilReader
except ImportError:
    FilReader = None  # Define as None if sigpyproc is not installed


def get_disp_delay(freq_lo_ghz, freq_hi_ghz, DM):
    """Estimate the dispersion time delay (in seconds) between the high and low freq
    channel for source at given DM.
    Args:
        freq_lo_ghz (float): Frequency corresponding to the lowest channel in GHz.
        freq_hi_ghz (float): Frequency corresponding to the highest channel in GHz.
        DM (float): Source DM.
    Returns:
        float: Time delay in sec.
    """
    # 4.15 * 10^3 * DM * (1/f_low_MHz^2 - 1/f_high_MHz^2) -> ms
    # 4.15 * DM * (1/f_low_GHz^2 - 1/f_high_GHz^2) -> s
    return 4.15 * DM * (1/freq_lo_ghz**2 - 1/freq_hi_ghz**2)


def load_injections(filepath: Path) -> pd.DataFrame:
    """
    Read an injection info file (.injinf) created by Ra_one.py.
    The file format is expected to be: DM final_snr time_s width_s target_snr
    """
    if not filepath.exists():
        raise FileNotFoundError(f"Injection file not found: {filepath}")
    df = pd.read_csv(filepath, sep=r'\s+', comment='#', header=None, engine='python')
    if df.shape[1] < 5:
        raise ValueError(f"Unexpected .injinf format in {filepath}. Expected at least 5 columns.")
    df = df.iloc[:, :5]
    df.columns = ['inj_dm', 'inj_snr', 'inj_time', 'inj_width_s', 'inj_target_snr']
    return df.astype(float)


def _parse_presto(filepath: Path, tsamp: Optional[float], prepdata_downsamp: int) -> pd.DataFrame:
    """Parse a Presto-style .singlepulse output file."""
    df = pd.read_csv(filepath, sep=r'\s+', comment='#', header=None, engine='python')
    cols = df.shape[1]

    if cols >= 5:
        # Standard PRESTO format: DM, SNR, Time, Sample, Downfact
        det_dm = df.iloc[:, 0]
        det_snr = df.iloc[:, 1]
        det_time = df.iloc[:, 2]
        downfact = df.iloc[:, 4]
        if tsamp:
            # The effective sampling time is the original tsamp * prepdata downsamp factor
            effective_tsamp = tsamp * prepdata_downsamp
            det_width_s = downfact.astype(float) * effective_tsamp
        else:
            print("Warning: --presto-tsamp not provided. Cannot calculate width for PRESTO detections.", file=sys.stderr)
            det_width_s = np.nan
    elif cols >= 4:
        # Some custom formats might be: SNR, Time, DM, Width
        det_snr = df.iloc[:, 0]
        det_time = df.iloc[:, 1]
        det_dm = df.iloc[:, 2]
        det_width_s = df.iloc[:, 3].astype(float)
    else:
        raise ValueError(f"Unrecognized PRESTO singlepulse format in {filepath}")

    return pd.DataFrame({
        'det_dm': det_dm.astype(float),
        'det_snr': det_snr.astype(float),
        'det_time': det_time.astype(float),
        'det_width_s': pd.to_numeric(det_width_s, errors='coerce'),
    })


def _parse_transientx(filepath: Path) -> pd.DataFrame:
    """Parse a generic transient_X output (time, snr, dm, width)."""
    df = pd.read_csv(filepath, sep=r'\s+', comment='#', header=None, engine='python')
    if df.shape[1] < 4:
        raise ValueError(f"Unexpected transient_X format in {filepath}. Expected at least 4 columns.")
    df = df.iloc[:, :4]
    df.columns = ['det_time', 'det_snr', 'det_dm', 'det_width_s']
    return df.astype(float)


def load_detections(filepath: Path, det_format: str, tsamp: Optional[float], prepdata_downsamp: int) -> pd.DataFrame:
    """Load detection file by dispatching to the correct parser."""
    if not filepath.exists():
        raise FileNotFoundError(f"Detection file not found: {filepath}")

    if det_format == 'presto':
        return _parse_presto(filepath, tsamp, prepdata_downsamp)
    elif det_format == 'transientx':
        return _parse_transientx(filepath)
    else:
        raise ValueError(f"Unknown detection format: {det_format}")


def match_detections(
    inj_df: pd.DataFrame,
    det_df: pd.DataFrame,
    tolerances: Dict[str, float]
) -> pd.DataFrame:
    """Match detections to injections using a multi-stage filtering process."""
    inj = inj_df.copy()
    det = det_df.copy()

    # Prepare columns for matching results
    inj['is_detected'] = False
    for col in ['det_snr', 'det_dm', 'det_time', 'det_width_s']:
        inj[col] = np.nan
    det['matched_to_inj_index'] = -1

    # Iterate through each injection to find a match
    for inj_idx, injection in inj.iterrows():
        unmatched_dets = det[det['matched_to_inj_index'] == -1].copy()
        if unmatched_dets.empty:
            break

        # 1. Time filter
        time_tol = max(injection['inj_width_s'] * tolerances['time_factor'], tolerances['min_time_tol'])
        time_diff = np.abs(unmatched_dets['det_time'] - injection['inj_time'])
        candidates = unmatched_dets[time_diff <= time_tol].copy()
        if candidates.empty:
            continue
        candidates['time_diff'] = np.abs(candidates['det_time'] - injection['inj_time'])

        # 2. DM filter
        dm_tol = max(tolerances['dm_abs'], injection['inj_dm'] * tolerances['dm_rel'])
        dm_diff = np.abs(candidates['det_dm'] - injection['inj_dm'])
        candidates = candidates[dm_diff <= dm_tol].copy()
        if candidates.empty:
            continue
        
        # 3. Width filter is REMOVED as requested, as detected width can be inaccurate.

        # 4. Select best match (highest SNR, then smallest time diff)
        candidates = candidates.sort_values(by=['det_snr', 'time_diff'], ascending=[False, True])
        best_match = candidates.iloc[0]
        
        # 5. Record the match
        inj.loc[inj_idx, 'is_detected'] = True
        for col in ['snr', 'dm', 'time', 'width_s']:
            inj.loc[inj_idx, f'det_{col}'] = best_match.get(f'det_{col}', np.nan)
        
        det.loc[best_match.name, 'matched_to_inj_index'] = inj_idx

    return inj


def compute_snr_scale(matched_df: pd.DataFrame, min_count: int, det_snr_cutoff: float) -> Dict[str, Any]:
    """
    Calculate an overall SNR scaling factor and per-width scaling factors.
    Factors are calculated as the median of det_snr/inj_snr for high-SNR detections.
    """
    used = matched_df[matched_df['is_detected']].dropna(subset=['det_snr', 'inj_snr']).copy()

    if used.empty:
        return {
            'per_width_factors': {},
            'overall_scale_factor': np.nan,
            'fit_data': pd.DataFrame(),
            'warning': "Warning: No matched pulses found. Could not determine scale factor."
        }

    per_width_factors = {}
    overall_scale_factor = np.nan
    warning = None

    # Calculate per-width scale factors
    # Ensure 'width_ms_group' is available for grouping
    if 'width_ms_group' not in used.columns:
        warning = "Warning: 'width_ms_group' not found in matched_df. Cannot calculate per-width scale factors."
    else:
        for width_ms_group, group in used.groupby('width_ms_group'):
            # Filter for detections above the specified detected SNR cutoff
            high_snr_det_subset = group[group['det_snr'] > det_snr_cutoff]
            
            if len(high_snr_det_subset) < min_count:
                per_width_factors[str(width_ms_group)] = np.nan
                continue

            # Calculate ratios, ensuring inj_snr is not zero
            ratios = high_snr_det_subset[high_snr_det_subset['inj_snr'] > 0]['det_snr'] / \
                     high_snr_det_subset[high_snr_det_subset['inj_snr'] > 0]['inj_snr']
            
            if not ratios.empty:
                median_ratio = ratios.median()
                per_width_factors[str(width_ms_group)] = median_ratio
            else:
                per_width_factors[str(width_ms_group)] = np.nan

    # Calculate an overall scale factor from all high-SNR detections (across all widths)
    # This serves as a fallback if a specific width group doesn't have enough data.
    all_high_snr_det = used[used['det_snr'] > det_snr_cutoff]
    if len(all_high_snr_det) >= min_count:
        overall_ratios = all_high_snr_det[all_high_snr_det['inj_snr'] > 0]['det_snr'] / \
                         all_high_snr_det[all_high_snr_det['inj_snr'] > 0]['inj_snr']
        if not overall_ratios.empty:
            overall_scale_factor = overall_ratios.median()
    
    if np.isnan(overall_scale_factor) and warning is None:
        warning = f"Warning: Could not determine an overall SNR scale factor (not enough detections with det_snr > {det_snr_cutoff})."

    return {
        'per_width_factors': per_width_factors,
        'overall_scale_factor': overall_scale_factor,
        'fit_data': used,  # All detected pulses
        'warning': warning
    }


def find_recovery_thresholds(matched_df: pd.DataFrame, level: float = 0.90) -> Dict[str, Optional[float]]:
    """Compute the EXPECTED DETECTED SNR at which recovery reaches a certain level, per width."""
    if 'expected_det_snr' not in matched_df.columns:
        print("Skipping recovery threshold calculation: 'expected_det_snr' not available.")
        return {}

    thresholds = {}
    for width_ms_group, group in matched_df.groupby('width_ms_group'):
        width_ms = str(width_ms_group)
        if len(group) < 10:
            print(f"Warning: Not enough data points ({len(group)}) for width {width_ms}ms to calculate {level*100}% recovery threshold.", file=sys.stderr)
            thresholds[width_ms] = None
            continue

        # Bin data by expected SNR with a bin width of 1
        max_snr = group['expected_det_snr'].max()
        if pd.isna(max_snr) or max_snr < 1:
            thresholds[width_ms] = None
            continue

        bins = np.arange(0.5, np.ceil(max_snr) + 1.5, 1.0)
        bin_centers = bins[:-1] + 0.5

        total_in_bin, _ = np.histogram(group['expected_det_snr'], bins=bins)
        detected_in_bin, _ = np.histogram(group[group['is_detected']]['expected_det_snr'], bins=bins)

        completeness = np.divide(detected_in_bin, total_in_bin, out=np.full_like(total_in_bin, np.nan, dtype=float), where=(total_in_bin >= 5))

        # Filter to valid bins and interpolate to find the threshold
        valid_mask = ~np.isnan(completeness)
        if not np.any(valid_mask) or valid_mask.sum() < 2:
            thresholds[width_ms] = None
            continue

        valid_snrs = bin_centers[valid_mask]
        valid_completeness = completeness[valid_mask]

        # Ensure completeness values are monotonically increasing for interpolation
        # by taking the cumulative maximum. This prevents issues with noisy curves.
        monotonic_completeness = np.maximum.accumulate(valid_completeness)

        if monotonic_completeness.max() < level:
            thresholds[width_ms] = None  # Cannot reach the desired completeness level
            continue

        # Use cubic spline interpolation and root finding to get the threshold
        try:
            # Ensure data is sorted by SNR (bin_centers are already sorted)
            x_fit = valid_snrs
            y_fit_monotonic = monotonic_completeness

            # Create the interpolation function
            interp_func = interpolate.interp1d(x_fit, y_fit_monotonic, kind='cubic', bounds_error=False, fill_value=(y_fit_monotonic[0], y_fit_monotonic[-1]))

            # Define the function to find the root of: f(x) = interp(x) - level
            func_to_solve = lambda x: interp_func(x) - level

            # Find the interval to search for the root
            first_above_level_idx = np.where(y_fit_monotonic >= level)[0][0]

            if first_above_level_idx > 0:
                a, b = x_fit[first_above_level_idx - 1], x_fit[first_above_level_idx]
                # Use brentq for robust root finding
                threshold_snr = brentq(func_to_solve, a, b)
            else: # The very first point is already >= level
                threshold_snr = x_fit[0]

            thresholds[width_ms] = threshold_snr
        except (IndexError, ValueError) as e:
            print(f"Warning: Could not find {level*100}% threshold for width {width_ms}ms via interpolation. Reason: {e}", file=sys.stderr)
            thresholds[width_ms] = None

    return thresholds


def plot_snr_comparison(detected_df: pd.DataFrame, per_width_factors: Dict, overall_factor: float, outdir: Path):
    """
    Plot scaled injected SNR vs. detected SNR.
    The x-axis shows the injected SNR after applying the per-width scale factor.
    Points are colored by their width group.
    """
    if detected_df.empty or 'expected_det_snr' not in detected_df.columns:
        print("Skipping SNR comparison plot: no matched data or expected_det_snr.")
        return

    plt.figure(figsize=(8, 6))

    # Add y=x line for reference, representing a perfect scaling
    max_val = max(detected_df['expected_det_snr'].max(), detected_df['det_snr'].max())
    if pd.notna(max_val):
        plt.plot([0, max_val * 1.1], [0, max_val * 1.1], color='k', linestyle='--', label='Ideal (y=x)')

    # Use seaborn for easier hue-based plotting, coloring by width group
    if 'width_ms_group' in detected_df.columns:
        plot_df = detected_df.copy()
        # Create a prettier label for the legend showing the scale factor for each width
        # Ensure width_ms_group is treated as string for dict lookup
        plot_df['width_group_label'] = plot_df['width_ms_group'].apply(
            lambda w: f"{w} ms (x{per_width_factors.get(str(w), overall_factor):.2f})"
        )
        sns.scatterplot(
            data=plot_df, x='expected_det_snr', y='det_snr', hue='width_group_label',
            alpha=0.7, s=40, ax=plt.gca()
        )
        plt.legend(title='Inj. Width (Factor)')
    else:
        plt.scatter(detected_df['expected_det_snr'], detected_df['det_snr'], s=30, alpha=0.6, label='All Matched Pulses')
        plt.legend()

    plt.xlabel('Scaled Injected SNR (Expected Detected SNR)')
    plt.ylabel('Detected SNR')
    plt.title('Scaled Injected vs. Detected SNR')
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.savefig(outdir / 'snr_comparison.png')
    plt.close()


def plot_completeness(matched_df: pd.DataFrame, outdir: Path):
    """
    Plots completeness vs. expected SNR with a smooth cubic spline fit for each
    pulse width group.
    """
    if 'expected_det_snr' not in matched_df.columns:
        print("Skipping completeness plot: 'expected_det_snr' column not available.")
        return

    plt.style.use('seaborn-v0_8-whitegrid')
    # Get a list of unique widths to assign colors consistently
    unique_widths = sorted(matched_df['width_ms_group'].unique())
    # Using a colormap for distinct colors for each width
    colors_palette = plt.cm.viridis(np.linspace(0, 1, len(unique_widths)))
    width_color_map = dict(zip(unique_widths, colors_palette))

    all_completeness_data = []
    fig, ax1 = plt.subplots(figsize=(8, 5))
    for width_ms, group in matched_df.groupby('width_ms_group'):
        if len(group) < 10:
            continue

        max_snr = group['expected_det_snr'].max()
        if pd.isna(max_snr) or max_snr < 1:
            continue

        bins = np.arange(0.5, np.ceil(max_snr) + 1.5, 1)
        bin_centers = bins[:-1] + 0.5

        total_in_bin, _ = np.histogram(group['expected_det_snr'], bins=bins)
        detected_in_bin, _ = np.histogram(group[group['is_detected']]['expected_det_snr'], bins=bins)

        # Use bins with >= 5 points for a stable fit
        completeness_for_fit = np.divide(detected_in_bin, total_in_bin, out=np.full_like(total_in_bin, np.nan, dtype=float), where=(total_in_bin >= 5))
        fit_mask = ~np.isnan(completeness_for_fit)

        current_color = width_color_map.get(width_ms, 'black')
        label_text = f'{width_ms} ms ({len(group)})'

        # --- Spline Fitting and Plotting ---
        if fit_mask.sum() < 4:
            print(f"Warning: Not enough data points ({fit_mask.sum()}) for cubic interpolation for width {width_ms} ms. Plotting points only.", file=sys.stderr)
            completeness_for_plot = np.divide(detected_in_bin, total_in_bin, out=np.full_like(total_in_bin, np.nan, dtype=float), where=(total_in_bin > 0))
            plot_mask = ~np.isnan(completeness_for_plot)
            if plot_mask.sum() > 0:
                ax1.scatter(bin_centers[plot_mask], completeness_for_plot[plot_mask], color=current_color, marker='x', s=40, label=label_text)
        else:
            try:
                x_fit = bin_centers[fit_mask]
                y_fit = completeness_for_fit[fit_mask]
                y_fit_monotonic = np.maximum.accumulate(y_fit)

                interp_func = interpolate.interp1d(x_fit, y_fit_monotonic, kind='cubic', bounds_error=False, fill_value=(y_fit_monotonic[0], y_fit_monotonic[-1]))

                x_smooth = np.linspace(x_fit.min(), x_fit.max(), 300)
                y_smooth = np.clip(interp_func(x_smooth), 0, 1)

                ax1.plot(x_smooth, y_smooth, linestyle='-', label=label_text, color=current_color, linewidth=2.5, zorder=10)

            except (RuntimeError, ValueError) as e:
                print(f"Warning: Could not create interpolation for width {width_ms} ms. Plotting points only. Reason: {e}", file=sys.stderr)

        # Plot original data points on top of the curve
        completeness_for_plot = np.divide(detected_in_bin, total_in_bin, out=np.full_like(total_in_bin, np.nan, dtype=float), where=(total_in_bin > 0))
        plot_mask = ~np.isnan(completeness_for_plot)
        
        strong_points_mask = (total_in_bin >= 10) & plot_mask
        weak_points_mask = (total_in_bin > 0) & (total_in_bin < 10) & plot_mask

        ax1.scatter(bin_centers[strong_points_mask], completeness_for_plot[strong_points_mask],
                    color=current_color, alpha=0.9, s=50, marker='o', zorder=15, label=None)
        ax1.scatter(bin_centers[weak_points_mask], completeness_for_plot[weak_points_mask],
                    color=current_color, alpha=0.4, s=30, marker='o', zorder=15, label=None)

        # Store data for saving (only for bins with >= 5 bursts)
        official_completeness = np.divide(detected_in_bin, total_in_bin, out=np.full_like(total_in_bin, np.nan, dtype=float), where=(total_in_bin >= 5))
        for i in np.where(total_in_bin >= 5)[0]:
            all_completeness_data.append({
                'width_ms': width_ms,
                'snr_bin_center': bin_centers[i],
                'completeness': official_completeness[i],
                'n_detected': detected_in_bin[i],
                'n_total': total_in_bin[i]
            })

    # Add horizontal line for the 90% completeness level
    ax1.axhline(y=0.9, color='gray', linestyle=':', alpha=0.9, zorder=0, linewidth=2)

    ax1.set_xlabel('Expected Detected SNR (Binned)')
    ax1.set_ylabel('Completeness')
    ax1.set_ylim(-0.05, 1.05)
    ax1.set_title('Pipeline Completeness by Pulse Width')
    ax1.legend(title='Width (count)')
    ax1.grid(True, linestyle=':', alpha=0.7)

    fig.tight_layout()
    fig.savefig(outdir / 'completeness_by_width.png')

    # Create a zoomed-in version
    ax1.set_xlim(-0.5, 15.5)
    ax1.set_title('Pipeline Completeness by Pulse Width (Zoomed < 15 SNR)')
    fig.savefig(outdir / 'completeness_by_width_zoomed.png')
    plt.close(fig)

    # Save the collected data to a CSV file
    if all_completeness_data:
        completeness_df = pd.DataFrame(all_completeness_data)
        completeness_df.to_csv(outdir / 'completeness_data.csv', index=False, float_format='%.4f')
        print(f"Saved completeness data to '{outdir / 'completeness_data.csv'}'")


def plot_corner(df: pd.DataFrame, filename: str, title: str):
    """Generate and save a corner plot."""
    if len(df) < 2:
        print(f"Skipping corner plot '{filename}': not enough data points.")
        return

    df_plot = df.copy()
    df_plot['is_detected'] = df_plot['is_detected'].astype(int)
    df_plot['inj_width_ms'] = df_plot['inj_width_s'] * 1000.0
    # Add detected width in ms for plotting
    df_plot['det_width_ms'] = df_plot['det_width_s'] * 1000.0
    
    # Include both injected and detected parameters for matched pulses
    cols_to_plot = ['inj_dm', 'inj_snr', 'inj_width_ms', 'det_dm', 'det_snr', 'det_width_ms', 'is_detected']
    
    grid = sns.pairplot(df_plot[cols_to_plot], hue='is_detected', corner=True, plot_kws={'s': 14, 'alpha': 0.7})
    grid.fig.suptitle(title, y=1.03)
    grid.fig.tight_layout()
    grid.fig.savefig(filename)
    plt.close()


def plot_matched_pulses(matched_df: pd.DataFrame, fil_filepath: Path, outdir: Path, num_to_plot: int, max_plot_duration_s: float):
    """
    Generate side-by-side waterfall plots for a sample of matched pulses to allow
    visual verification of the matching quality.
    """
    if FilReader is None:
        print("Warning: sigpyproc is not installed. Skipping visualization.", file=sys.stderr)
        return

    matched_pulses = matched_df[matched_df['is_detected']].head(num_to_plot)
    if matched_pulses.empty:
        print("No matched pulses to visualize.")
        return

    try:
        fil_reader = FilReader(str(fil_filepath))
        tsamp = fil_reader.header.tsamp
        total_nsamps = fil_reader.header.nsamples
        f_top_ghz = fil_reader.header.ftop / 1000.0  # Convert MHz to GHz
        f_bottom_ghz = fil_reader.header.fbottom / 1000.0 # Convert MHz to GHz
    except Exception as e:
        print(f"Error opening filterbank file for visualization: {e}", file=sys.stderr)
        return

    vis_dir = outdir / 'match_visualizations'
    vis_dir.mkdir(exist_ok=True)

    for idx, pulse in matched_pulses.iterrows():
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6), sharey=True)

        # Calculate plot width based on DM delay, with a buffer
        dm_delay_s = get_disp_delay(f_bottom_ghz, f_top_ghz, pulse['inj_dm'])
        plot_width_s = min(max_plot_duration_s, max(0.2, dm_delay_s + 5 * pulse['inj_width_s'] + 0.05))

        # --- Plot centered on Injected Time ---
        center_time_inj = pulse['inj_time']
        start_time_req_inj = center_time_inj - (plot_width_s / 2)
        end_time_req_inj = center_time_inj + (plot_width_s / 2)

        # Clip the plot window to the file boundaries
        start_samp_inj = max(0, int(start_time_req_inj / tsamp))
        end_samp_inj = min(total_nsamps, int(end_time_req_inj / tsamp))
        nsamps_to_read_inj = end_samp_inj - start_samp_inj

        if nsamps_to_read_inj <= 0:
            ax1.text(0.5, 0.5, 'Data out of bounds', ha='center', va='center')
        else:
            data_inj = fil_reader.read_block(start_samp_inj, nsamps_to_read_inj).data
            plot_start_time_inj = start_samp_inj * tsamp
            plot_end_time_inj = end_samp_inj * tsamp
            ax1.imshow(data_inj, aspect='auto', cmap='viridis',
                             extent=[plot_start_time_inj, plot_end_time_inj,
                                     fil_reader.header.fbottom, fil_reader.header.ftop])

        ax1.set_title(f"Injected\n"
                      f"Time: {pulse['inj_time']:.4f}s, DM: {pulse['inj_dm']:.2f}\n"
                      f"S/N: {pulse['inj_snr']:.2f}, Width: {pulse['inj_width_s']*1000:.2f}ms")
        ax1.set_xlabel("Time (s)")
        ax1.set_ylabel("Frequency (MHz)")

        # --- Plot centered on Detected Time ---
        center_time_det = pulse['det_time']
        start_time_req_det = center_time_det - (plot_width_s / 2)
        end_time_req_det = center_time_det + (plot_width_s / 2)

        start_samp_det = max(0, int(start_time_req_det / tsamp))
        end_samp_det = min(total_nsamps, int(end_time_req_det / tsamp))
        nsamps_to_read_det = end_samp_det - start_samp_det

        det_width_val = pulse.get('det_width_s', np.nan)
        det_width_str = f"{det_width_val*1000:.2f}ms" if pd.notna(det_width_val) else "N/A"

        if nsamps_to_read_det <= 0:
            ax2.text(0.5, 0.5, 'Data out of bounds', ha='center', va='center')
        else:
            data_det = fil_reader.read_block(start_samp_det, nsamps_to_read_det).data
            plot_start_time_det = start_samp_det * tsamp
            plot_end_time_det = end_samp_det * tsamp
            ax2.imshow(data_det, aspect='auto', cmap='viridis',
                       extent=[plot_start_time_det, plot_end_time_det,
                               fil_reader.header.fbottom, fil_reader.header.ftop])

        ax2.set_title(f"Detected\n"
                      f"Time: {pulse['det_time']:.4f}s, DM: {pulse['det_dm']:.2f}\n"
                      f"S/N: {pulse['det_snr']:.2f}, Width: {det_width_str}")
        ax2.set_xlabel("Time (s)")

        fig.suptitle(f"Match Visualization for Injection Index {idx}", fontsize=16)
        fig.tight_layout(rect=[0, 0.03, 1, 0.95])

        plot_filename = vis_dir / f"match_vis_inj_{idx}.png"
        plt.savefig(plot_filename)
        plt.close(fig)
    print(f"Saved visualizations to '{vis_dir}'")

def main():
    parser = argparse.ArgumentParser(
        description="Analyze single-pulse search pipeline performance against injected pulses.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument('--inject-file', type=Path, required=True, help='Path to Ra_one .injinf file')
    parser.add_argument('--detection-file', type=Path, required=True, help='Path to search pipeline output file')
    parser.add_argument('--det-format', required=True, choices=['presto', 'transientx'], help='Format of the detection file')
    parser.add_argument('--fil-file', type=Path, help='Path to the filterbank file. If not provided, assumes it has the same basename as the inject-file.')
    parser.add_argument('--outdir', type=Path, default=Path('G_one_results'), help='Output directory for plots and tables')
    parser.add_argument('--prepdata-downsamp', type=int, default=1,
                        help='Downsampling factor used in prepdata. This is needed to correctly calculate PRESTO pulse widths.')
    
    # Tolerances
    parser.add_argument('--time-tol-factor', type=float, default=1.5, help='Time tolerance as a factor of injected width')
    parser.add_argument('--min-time-tol', type=float, default=0.002, help='Minimum absolute time tolerance (s)')
    parser.add_argument('--dm-tol-abs', type=float, default=0.5, help='Absolute DM tolerance for matching (pc/cm³)')
    parser.add_argument('--dm-tol-rel', type=float, default=0.05, help='Relative DM tolerance as a fraction of injected DM')
    parser.add_argument('--width-tol-factor', type=float, default=2.0, help='(Unused) Width tolerance factor. Kept for compatibility.')

    # Analysis params
    parser.add_argument('--snr-scaling-factor', type=float, default=1.0,
                        help='Manually provide an overall SNR scaling factor. If not given, it will be calculated automatically.')
    parser.add_argument('--snr-scale-min-count', type=int, default=10, help='Min points for SNR scaling fit')
    parser.add_argument('--det-snr-cutoff', type=float, default=10.0,
                        help='Minimum detected SNR for candidates to be included in SNR scaling factor calculation.')
    parser.add_argument('--width-rounding-precision', type=float, default=0.01,
                        help='Rounding precision for pulse widths (in ms) when grouping for completeness plots. '
                             'E.g., 0.01 for rounding to 2 decimal places, 1.0 for rounding to nearest integer.')

    # Visualization params
    parser.add_argument('--visualize', action='store_true',
                        help='Generate side-by-side waterfall plots for a few matched pulses to verify matching.')
    parser.add_argument('--num-visualize', type=int, default=5,
                        help='Number of matched pulses to visualize if --visualize is used.')
    parser.add_argument('--max-plot-duration', type=float, default=2.0,
                        help='Maximum duration (in seconds) for each visualization plot window.')
    
    args = parser.parse_args()
    args.outdir.mkdir(exist_ok=True)

    # Determine filterbank file path and tsamp
    fil_file = args.fil_file
    if not fil_file:
        fil_file = args.inject_file.with_suffix('.fil')
        print(f"Filterbank file not specified. Assuming: {fil_file}")

    tsamp = None
    if not fil_file.exists():
        parser.error(
            f"Filterbank file not found at '{fil_file}'. "
            "Please ensure it's in the same directory as the .injinf file or provide the path using --fil-file."
        )

    try:
        if FilReader is None:
            raise ImportError("sigpyproc is not installed, cannot read filterbank file.")
        fil_reader_for_tsamp = FilReader(str(fil_file))
        tsamp = fil_reader_for_tsamp.header.tsamp
        print(f"Using tsamp={tsamp:.6f}s from filterbank file '{fil_file}'.")
    except Exception as e:
        parser.error(f"Could not read tsamp from --fil-file '{fil_file}': {e}")

    # 1. Load data
    try:
        injections = load_injections(args.inject_file)
        detections = load_detections(args.detection_file, args.det_format, tsamp, args.prepdata_downsamp)
    except (FileNotFoundError, ValueError) as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)

    print(f"Loaded {len(injections)} injections and {len(detections)} detections.")

    # 2. Match detections to injections
    tolerances = { # width_tol_factor is unused in matching, but kept for compatibility
        'time_factor': args.time_tol_factor,
        'min_time_tol': args.min_time_tol,
        'dm_abs': args.dm_tol_abs,
        'dm_rel': args.dm_tol_rel,
        'width_factor': args.width_tol_factor,
    }
    matched_df = match_detections(injections, detections, tolerances)
    # Store metadata for later use in plotting functions

    print(f"Matched {matched_df['is_detected'].sum()} injections to detections.")

    # Calculate width_ms_group BEFORE calling compute_snr_scale
    matched_df['width_ms_group'] = (matched_df['inj_width_s'] * 1000.0 / args.width_rounding_precision).round(0) * args.width_rounding_precision

    # 3. Analyze SNR and calculate expected SNR
    if args.snr_scaling_factor is not None:
        print(f"Using manual SNR scaling factor: {args.snr_scaling_factor:.3f}")
        overall_scale_factor = args.snr_scaling_factor
        per_width_factors = {} # No per-width factors in manual mode
    else:
        print("Automatically calculating SNR scaling factor(s)...")
        snr_analysis = compute_snr_scale(matched_df, args.snr_scale_min_count, args.det_snr_cutoff)
        per_width_factors = snr_analysis['per_width_factors']
        overall_scale_factor = snr_analysis['overall_scale_factor']

        if snr_analysis.get('warning'):
            print(snr_analysis['warning'], file=sys.stderr)
    
    if np.isnan(overall_scale_factor):
        print("Warning: Could not determine overall SNR scale factor. Using a factor of 1.0 for completeness calculations.", file=sys.stderr)
        scale_factor_for_completeness = 1.0
    else:
        if args.snr_scaling_factor is None: # Only print if it was auto-calculated
            print(f"Overall SNR scale factor (median ratio of det_snr/inj_snr for det_snr > {args.det_snr_cutoff}): {overall_scale_factor:.3f}")
        scale_factor_for_completeness = overall_scale_factor

    if per_width_factors:
        print("Per-width SNR scale factors (width_ms_group: factor):")
        for width_ms, factor in per_width_factors.items():
            if not np.isnan(factor):
                print(f"  {width_ms} ms: {factor:.3f}")
            else:
                print(f"  {width_ms} ms: NaN")
    elif args.snr_scaling_factor is None: # Only print if auto and none were found
        print("Could not calculate per-width SNR scale factors (not enough data in width groups).")

    # Apply per-width scale factors where available, otherwise use the overall factor.
    if not matched_df.empty and 'width_ms_group' in matched_df.columns:
        # Map per-width factors, falling back to overall_scale_factor if a specific width has no factor
        scale_factors_to_apply = matched_df['width_ms_group'].astype(str).map(per_width_factors).fillna(scale_factor_for_completeness)
        matched_df['expected_det_snr'] = matched_df['inj_snr'] * scale_factors_to_apply
    else:
        print("Applying overall scale factor for 'expected_det_snr' calculation (no width groups or empty dataframe).")
        matched_df['expected_det_snr'] = matched_df['inj_snr'] * scale_factor_for_completeness

    # 4. Find 90% recovery thresholds
    thresholds = find_recovery_thresholds(matched_df)

    # 5. Generate outputs
    matched_df.to_csv(args.outdir / 'injections_matched.csv', index=False, float_format='%.4f')
    with open(args.outdir / '90_thresholds.json', 'w') as f:
        json.dump(thresholds, f, indent=2)
    print(f"Saved 90% recovery SNR thresholds to '{args.outdir / '90_thresholds.json'}'")

    detected_df = matched_df[matched_df['is_detected']]
    plot_snr_comparison(detected_df, per_width_factors, scale_factor_for_completeness, args.outdir)
    plot_completeness(matched_df, args.outdir)

    # All injections corner plot
    plot_corner(
        matched_df,
        args.outdir / 'corner_detected.png',
        'Injected Parameters (All) - Colored by Detection Status'
    )

    # High-confidence corner plot
    if thresholds:
        mask = matched_df.apply(
            lambda row: thresholds.get(str(row['width_ms_group'])) is not None and
                        row['expected_det_snr'] >= thresholds[str(row['width_ms_group'])],
            axis=1
        )
        high_conf_df = matched_df[mask]
        plot_corner(
            high_conf_df,
            args.outdir / 'corner_detected_ge90.png',
            'Injections with Expected Detected SNR ≥ 90% Recovery Threshold'
        )

    # 6. Visualize a few matches if requested
    if args.visualize:
        print(f"\nGenerating visualizations for up to {args.num_visualize} matched pulses...")
        plot_matched_pulses(
            matched_df,
            fil_filepath=fil_file,
            outdir=args.outdir,
            num_to_plot=args.num_visualize,
            max_plot_duration_s=args.max_plot_duration
        )

    print(f"\nAnalysis complete. Results are in '{args.outdir}'")


if __name__ == '__main__':
    main()
