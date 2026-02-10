import numpy as np
import matplotlib.pyplot as plt
import sys
from scipy.signal import fftconvolve

def _generate_gaussian_bandpass(nchan: int, bw_chan: float, central_freq_chan: float) -> np.ndarray:
    """
    Generates a Gaussian bandpass shape with a peak of 1.
    """
    if bw_chan <= 0:
        return np.ones(nchan)
    x = np.arange(nchan)
    gaussian = np.exp(-0.5 * ((x - central_freq_chan) / bw_chan) ** 2)
    max_val = np.max(gaussian)
    if max_val > 1e-9: # Avoid division by zero
        return gaussian / np.max(gaussian)
    return gaussian


def get_disp_delay_s(freq_lo_ghz: float, freq_hi_ghz: float, dm: float) -> float:
    """
    Estimate the dispersion time delay (in seconds) between the low and high frequency
    for a source at a given DM.

    Args:
        freq_lo_ghz (float): Lowest frequency in GHz.
        freq_hi_ghz (float): Highest frequency in GHz.
        dm (float): Dispersion Measure in pc/cm^3.

    Returns:
        float: Time delay in seconds.
    """
    # Dispersion constant D ~ 4.15e-3 s * GHz^2 / (pc cm^-3)
    # delay (s) = D * DM * (1/f_lo^2 - 1/f_hi^2)
    DISPERSION_CONSTANT_S_GHZ2 = 4.15e-3
    return DISPERSION_CONSTANT_S_GHZ2 * dm * (1/freq_lo_ghz**2 - 1/freq_hi_ghz**2)

def _dedisperse_array(
    data_array: np.ndarray,
    freqs_ghz: np.ndarray,
    dm: float,
    tsamp_s: float,
    ref_freq_ghz: float
) -> np.ndarray:
    """Dedisperses a 2D array by applying integer time shifts to each channel."""
    dedispersed_array = np.zeros_like(data_array)
    delays_s = get_disp_delay_s(freqs_ghz, ref_freq_ghz, dm)
    shifts = np.round(delays_s / tsamp_s).astype(int)
    ntime = data_array.shape[1]

    for i, shift in enumerate(shifts):
        if 0 <= shift < ntime:
            dedispersed_array[i, :ntime - shift] = data_array[i, shift:]

    return dedispersed_array

def _calculate_presto_snr(
    pulse_array: np.ndarray,
    std_per_channel: np.ndarray,
    effective_width_s: float, # The effective, observed width of the pulse in seconds (an
    tsamp_s: float, # The sampling time in seconds.
    # New args for dedispersion
    freqs_ghz: np.ndarray,
    dm: float,
    ref_freq_ghz: float
) -> float:
    """
    Calculates SNR of a pulse array given per-channel standard deviation,
    mimicking PRESTO's `single_pulse_search.py` definition more accurately.
    The signal is summed over a boxcar window applied to the dedispersed time series.

    Args:
        pulse_array (np.ndarray): 2D array (nchan, ntime) of the pulse signal.
        std_per_channel (np.ndarray): 1D array of noise standard deviation for each channel (nchan,).
        effective_width_s (float): The effective, observed width of the pulse in seconds (an
            approximation including intrinsic width, smearing, and scattering), used for the boxcar.
        tsamp_s (float): The sampling time in seconds.
        freqs_ghz (np.ndarray): Array of channel frequencies in GHz.
        dm (float): Dispersion Measure in pc/cm^3.
        ref_freq_ghz (float): Reference frequency for dispersion in GHz.

    Returns:
        float: The calculated PRESTO-like SNR.
    """
    # 1. Dedisperse the pulse array and create the time series
    dedispersed_pulse = _dedisperse_array(pulse_array, freqs_ghz, dm, tsamp_s, ref_freq_ghz)
    dedispersed_timeseries = np.sum(dedispersed_pulse, axis=0)

    # 2. Define the boxcar width in samples
    boxcar_width_samples = max(1, int(round(effective_width_s / tsamp_s)))

    # 3. Find the sum of the signal within the optimal boxcar window
    # To mimic PRESTO, we find the peak of the time series and sum over the boxcar centered on it.
    if len(dedispersed_timeseries) > 0 and np.max(dedispersed_timeseries) > 0:
        peak_idx = np.argmax(dedispersed_timeseries)
        
        # Center the boxcar on the peak
        start_idx = peak_idx - boxcar_width_samples // 2
        end_idx = start_idx + boxcar_width_samples
        
        # Ensure window is within bounds
        start_idx = max(0, start_idx)
        end_idx = min(len(dedispersed_timeseries), end_idx)
        
        total_signal = np.sum(dedispersed_timeseries[start_idx:end_idx])
    else:
        total_signal = 0.0

    # 4. Calculate the total noise in the denominator
    # PRESTO's single_pulse_search.py calculates the RMS of a mean-subtracted
    # time series, which is equivalent to its standard deviation (std).
    # Assuming independent noise between channels, Var(sum(X_i)) = sum(Var(X_i)),
    # so std(sum(X_i)) = sqrt(sum(std(X_i)^2)).
    total_noise_std = np.sqrt(np.sum(std_per_channel**2))

    # 5. Calculate SNR
    denominator = total_noise_std * np.sqrt(boxcar_width_samples)
    if denominator == 0:
        return np.inf if total_signal > 0 else 0.0
    
    return total_signal / denominator

def generate_dispersed_gaussian_pulse(
    nchan: int,
    ntime: int,
    tsamp_s: float,
    freq_lo_ghz: float,
    freq_hi_ghz: float,
    dm: float,
    width_s: float, # Intrinsic, pre-broadening pulse width (FWHM) in seconds
    injected_snr: float,
    std_per_channel: np.ndarray, # Array of noise standard deviation for each channel (nchan,)
    scattering_ms: float, # Scattering time in ms at highest frequency
    ref_freq_ghz: float = None, # Reference frequency for dispersion (usually highest)
    # Parameters for bandwidth shaping
    bandwidth_mode: str = 'broadband',
    bw_chan: float = 0,
    central_freq_chan: float = 0
) -> (np.ndarray, float):
    """
    Generates a dispersed Gaussian pulse in a 2D (frequency, time) array.
    The final pulse amplitude is scaled to match the target SNR, even for complex
    or narrowband morphologies.
    The pulse shape includes intrinsic Gaussian width, intra-channel smearing, and a
    exponential scattering tail applied via convolution for physical accuracy.

    Args:
        nchan (int): Number of frequency channels.
        ntime (int): Number of time samples.
        tsamp_s (float): Sampling time in seconds.
        freq_lo_ghz (float): Lowest frequency in GHz.
        freq_hi_ghz (float): Highest frequency in GHz.
        dm (float): Dispersion Measure in pc/cm^3.
        width_s (float): The intrinsic, pre-broadening Gaussian pulse width (FWHM) in seconds.
            The final, observed width will be larger due to DM smearing and scattering.
        injected_snr (float): Desired SNR of the injected pulse.
        std_per_channel (np.ndarray): 1D array of noise standard deviation for each channel (nchan,).
        scattering_ms (float): Scattering time in milliseconds at the highest frequency.
        ref_freq_ghz (float, optional): Reference frequency for dispersion in GHz.
            If None, uses freq_hi_ghz. Defaults to None.
        bandwidth_mode (str, optional): 'broadband' or 'narrowband'. Defaults to 'broadband'.
        bw_chan (float, optional): Bandwidth of the pulse in channels (for 'narrowband' mode).
        central_freq_chan (float, optional): Central frequency of the pulse in channel number (for 'narrowband' mode).

    Returns:
        tuple[np.ndarray, float]: A tuple containing:
            - pulse_data (np.ndarray): 2D array (nchan, ntime) with the final, scaled pulse.
            - calculated_snr (float): The actual SNR of the injected pulse.
    """
    if std_per_channel.shape[0] != nchan:
        raise ValueError(f"std_per_channel must have {nchan} elements, but got {std_per_channel.shape[0]}.")

    # --- Setup ---
    # Frequency and time arrays
    freqs_ghz = np.linspace(freq_hi_ghz, freq_lo_ghz, nchan)
    if ref_freq_ghz is None:
        ref_freq_ghz = freq_hi_ghz
    times_s = np.arange(ntime) * tsamp_s
    
    # Convert FWHM to standard deviation for Gaussian
    FWHM_TO_SIGMA = 1 / (2 * np.sqrt(2 * np.log(2)))
    sigma_intrinsic_s = width_s * FWHM_TO_SIGMA

    # Scattering time in seconds at the reference frequency
    tau_sc_ref_s = scattering_ms / 1000.0

    # --- Vectorized Per-Channel Calculations ---
    # 1. Dispersion delay for each channel
    delays_s = get_disp_delay_s(freqs_ghz, ref_freq_ghz, dm)
    print(f"Max dispersion delay across band: {np.max(delays_s):.6f} s")
    # 2. Intra-channel dispersion smearing
    channel_bw_ghz = (freq_hi_ghz - freq_lo_ghz) / nchan
    # Formula for smearing is dt_s = 8.3e-3 * DM * BW_GHz / f_GHz^3
    with np.errstate(divide='ignore', invalid='ignore'):
        tau_dm_smear_s = 8.3e-3 * dm * channel_bw_ghz / (freqs_ghz**3)
    tau_dm_smear_s[~np.isfinite(tau_dm_smear_s)] = np.inf

    # 3. Effective standard deviation (sigma) from intrinsic width and DM smearing.
    sigma_pre_scatter_sq = (
        sigma_intrinsic_s**2 +
        (tau_dm_smear_s * FWHM_TO_SIGMA)**2
    )
    sigma_pre_scatter_s = np.sqrt(sigma_pre_scatter_sq)

    # Warn if effective width is less than one sample and clamp to minimum
    min_sigma_s = tsamp_s * FWHM_TO_SIGMA
    if np.any(sigma_pre_scatter_s < min_sigma_s):
        undersampled_channels = np.where(sigma_pre_scatter_s < min_sigma_s)[0]
        print(
            f"Warning: Effective pulse width in {len(undersampled_channels)} channels "
            f"is less than one sample. Clamping to minimum.",
            file=sys.stderr
        )

    # --- Effective Width and SNR Setup ---
    # The effective width is an *approximation* of the final observed FWHM, used to
    # determine the optimal boxcar for the SNR calculation. It is calculated by
    # adding the intrinsic width, DM smearing, and scattering time in quadrature.
    # While physically scattering is a convolution, adding its timescale in quadrature
    # is a standard and effective approximation for estimating the final width.
    # We calculate this at the middle of the band for a representative value.
    mid_chan_idx = nchan // 2
    # Scattering time follows f^-4 scaling
    tau_sc_channel_s = tau_sc_ref_s * (ref_freq_ghz / freqs_ghz)**4
    width_pre_scatter_mid_s = sigma_pre_scatter_s[mid_chan_idx] / FWHM_TO_SIGMA
    tau_sc_mid_s = tau_sc_channel_s[mid_chan_idx]
    effective_width_s = np.sqrt(width_pre_scatter_mid_s**2 + tau_sc_mid_s**2)

    # 4. Arrival time for each channel
    t_dedispersed_ref_s = times_s[ntime // 2]  
    t_arrival_channel_s = t_dedispersed_ref_s + delays_s

    # --- Generate Pulse Profiles ---
    # First, generate the Gaussian profiles from intrinsic width and DM smearing
    time_diff = times_s[np.newaxis, :] - t_arrival_channel_s[:, np.newaxis]
    exponent = -0.5 * (time_diff / sigma_pre_scatter_s[:, np.newaxis])**2
    gaussian_profiles_pre_scatter = np.exp(exponent)


    # The kernel should be defined on a simple time axis, independent of pulse arrival time.
    kernel_times = np.arange(ntime) * tsamp_s
    # Add a small epsilon to prevent division by zero when scattering is zero.
    # This creates a one-sided exponential kernel for each channel.
    scattering_kernels = np.exp(-kernel_times[np.newaxis, :] / (tau_sc_channel_s[:, np.newaxis] + 1e-12))

    # Normalize each kernel to have a sum of 1.
    kernel_sums = np.sum(scattering_kernels, axis=1, keepdims=True)
    np.divide(scattering_kernels, kernel_sums, out=scattering_kernels, where=kernel_sums != 0)

    shifted_scattering_kernels = np.roll(scattering_kernels, ntime // 2, axis=1)
    '''
    #Plot Convolution kernels for verification
    plt.figure(figsize=(10, 6))
    for i in range(0, nchan, max(1, nchan // 10)):
        plt.plot(kernel_times * 1000, shifted_scattering_kernels[i], label=f'Channel {i} ({freqs_ghz[i]:.2f} GHz)')
    plt.xlabel('Time (ms)')
    plt.ylabel('Amplitude')
    plt.title('Scattering Convolution Kernels for Selected Channels')
    plt.legend()
    plt.grid(True)
    plt.savefig('scattering_kernels.png', dpi=300)
    plt.close()
    '''
    # Use scipy's fftconvolve with axes=1 to perform a vectorized 1D convolution for each channel.
    # This is much faster than a Python loop and correctly models the pulse position.
    gaussian_profiles_nominal = fftconvolve(
        gaussian_profiles_pre_scatter, shifted_scattering_kernels, mode='same', axes=1
    )

    # --- Apply Spectral Morphology to the Unscaled Pulse ---
    # This creates the final shape of the pulse before amplitude scaling.
    pulse_nominal_shaped = gaussian_profiles_nominal.copy()

    if bandwidth_mode == 'narrowband':
        bandpass_shape = _generate_gaussian_bandpass(nchan, bw_chan, central_freq_chan)
        pulse_nominal_shaped *= bandpass_shape[:, np.newaxis]

    if bandwidth_mode == 'complex':
        # Generates multiple narrow-band or broad-band drifting pulses randomly
        num_pulses = np.random.randint(2, 6) # Randomly choose number of sub-pulses between 2 and 5
        pulse_nominal_shaped.fill(0) # Reset pulse data
        for _ in range(num_pulses):
            bw_chan_sub = np.random.uniform(nchan/10, nchan/2) # Random bandwidth 
            central_freq_chan_sub = np.random.uniform(0, nchan) # Random central frequency channel
            bandpass_shape_sub = _generate_gaussian_bandpass(nchan, bw_chan_sub, central_freq_chan_sub)
            # Randomly shift the pulse in time
            time_shift_samples = np.random.randint(-5*width_s//tsamp_s, 5*width_s//tsamp_s) #-10*width_s//tsamp_s, 10*width_s//tsamp_s)
            # Use the base broadband pulse as the component
            shifted_pulse = np.roll(gaussian_profiles_nominal, time_shift_samples, axis=1)
            # Zero out the rolled-over parts to avoid edge effects
            if time_shift_samples > 0:
                shifted_pulse[:, :time_shift_samples] = 0
            elif time_shift_samples < 0:
                shifted_pulse[:, time_shift_samples:] = 0
            # Add the component with a random relative amplitude
            pulse_nominal_shaped += (shifted_pulse * bandpass_shape_sub[:, np.newaxis] * np.random.uniform(0.5, 1.0))

    # --- Amplitude Scaling ---
    # To scale the pulse to the target SNR, we first calculate the SNR of the
    # unscaled nominal pulse (as if its amplitude were 1).
    nominal_snr = _calculate_presto_snr(
        pulse_nominal_shaped, std_per_channel, effective_width_s, tsamp_s,
        freqs_ghz, dm, ref_freq_ghz
    )

    # The final pulse amplitude is linear with SNR. We can find the required
    # scaling factor `A_peak` to reach the target `injected_snr`.
    if nominal_snr > 1e-9:
        A_peak = injected_snr / nominal_snr
    else:
        A_peak = 0.0
        print("Warning: Nominal SNR is zero. Cannot scale pulse. Returning zero-amplitude pulse.", file=sys.stderr)

    # Generate the final pulse data by scaling the shaped nominal pulse
    pulse_data = (A_peak * pulse_nominal_shaped).astype(np.float32)

    # --- Final SNR Calculation ---
    # The final SNR is simply the target SNR. No need to recalculate.
    # We can return `injected_snr` or `A_peak * nominal_snr` for verification.
    final_snr = injected_snr if A_peak != 0.0 else 0.0

    return pulse_data, final_snr

if __name__ == '__main__':
    # Example Usage:
    nchan = 1024
    ntime = 4096
    tsamp_s = 0.000327 # 327 us
    freq_lo_ghz = 0.550 # GHz
    freq_hi_ghz = 0.750 # GHz
    dm = 100.0 # pc/cm^3
    width_s = 10/1000
    injected_snr = 10.0
    
    # Assume uniform RMS for all channels for simplicity in this example
    std_per_channel = np.full(nchan, 0.1)
    
    scattering_ms = 0.01 # 0.1 ms scattering at highest frequency to make the tail visible

    print(f"Generating pulse with target SNR: {injected_snr}")
    pulse_array, actual_snr = generate_dispersed_gaussian_pulse(
        nchan, ntime, tsamp_s, freq_lo_ghz, freq_hi_ghz, dm, width_s, # type: ignore
        injected_snr, std_per_channel, scattering_ms,
        # Example of narrowband injection
        bandwidth_mode='complex',
        bw_chan=nchan/4, # Pulse covers 1/4 of the band
        central_freq_chan=nchan/2 # Centered in the band
    )

    print(f"Calculated SNR of injected pulse: {actual_snr:.2f}")
    
    # Visualization
    plt.figure(figsize=(12, 8))
    plt.imshow(pulse_array, aspect='auto', cmap='viridis',
               extent=[0, ntime * tsamp_s, freq_lo_ghz, freq_hi_ghz])
    plt.colorbar(label='Amplitude')
    plt.xlabel('Time (s)')
    plt.ylabel('Frequency (GHz)')
    plt.title(f'Injected Dispersed Gaussian Pulse (Target SNR: {injected_snr:.1f}, Actual SNR: {actual_snr:.1f})')
    plt.savefig('dispersed_gaussian_pulse.png', dpi=300)
    plt.show()
      
    # Plot dedispersed time series
    print("\nDedispersing array for time series plot...")
    # Create the frequency array as used in the simulation
    freqs_for_dedisp = np.linspace(freq_hi_ghz, freq_lo_ghz, nchan)
    ref_freq_for_dedisp = freq_hi_ghz # The reference frequency used in simulation

    # Dedisperse the pulse array
    dedispersed_pulse_array = _dedisperse_array(
        pulse_array, freqs_for_dedisp, dm, tsamp_s, ref_freq_for_dedisp
    )
    # Sum over frequency to get the time series
    dedispersed_timeseries = np.sum(dedispersed_pulse_array, axis=0)

    plt.figure(figsize=(10, 4))
    plt.plot(np.arange(ntime) * tsamp_s, dedispersed_timeseries)
    plt.xlabel('Time (s)')
    plt.ylabel('Dedispersed Amplitude')
    plt.title('Dedispersed Pulse Profile')
    plt.grid(True)
    plt.savefig('dedispersed_pulse_profile.png', dpi=300)
    plt.show()

    #plot zoomed version of dedispersed time series
    pulse_center_idx = ntime // 2
    zoom_window_s = 0.1 # 10 ms window
    zoom_window_samples = int(zoom_window_s / tsamp_s)
    start_idx = max(0, pulse_center_idx - zoom_window_samples // 2)
    end_idx = min(ntime, pulse_center_idx + zoom_window_samples // 2)
    plt.figure(figsize=(10, 4))
    plt.plot(np.arange(start_idx, end_idx) * tsamp_s, dedispersed_timeseries[start_idx:end_idx])
    #plot verticle line at center 
    plt.axvline(pulse_center_idx * tsamp_s, color='r', linestyle='--')
    plt.xlabel('Time (s)')
    plt.ylabel('Dedispersed Amplitude')

    plt.title('Zoomed Dedispersed Pulse Profile')
    plt.grid(True)
    plt.savefig('dedispersed_pulse_profile_zoomed.png', dpi=300)
    plt.show()
