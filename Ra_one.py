#from turtle import down
import numpy as np
import matplotlib.pyplot as plt
import copy
from .simulator import generate_dispersed_gaussian_pulse, _dedisperse_array
import your
from scipy.ndimage import shift
from matplotlib.backends.backend_pdf import PdfPages
import argparse
import os
import pathlib
import simpulse
import warnings
from sigpyproc.readers import FilReader
from sigpyproc.block import FilterbankBlock
from sigpyproc.io.fileio import FileWriter
from sigpyproc.io import sigproc
from copy import copy, deepcopy
from scipy import signal
from scipy.stats import norm
import random
import json



def load_config(config_path="injection_config.json"):
    """
    Load injection parameters configuration from JSON file.
    """
    with open(config_path, 'r') as f:
        config = json.load(f)
    return config

def get_width_interval(config):
    return config['width']['lower'], config['width']['upper']

def power_law_distribution(lower, upper, n, slope=-1.5):
    """
    Generates 'n' random numbers from a power-law distribution.

    The distribution is defined by P(x) ~ x^slope over the range [upper, lower].
    Note: For a decaying distribution (more small values), the slope must be negative.
    The config file uses 'lower' for the max value and 'upper' for the min value.
    
    Args:
        lower (float): The maximum value of the range (e.g., 30).
        upper (float): The minimum value of the range (e.g., 5).
        n (int): The number of samples to generate.
        slope (float): The power-law index.

    Returns:
        np.ndarray: An array of 'n' samples.
    """
    # Use inverse transform sampling. For a distribution P(x) ~ x^slope,
    # the inverse of the cumulative distribution function (CDF) is used to
    # transform a uniform random variable `u` into a power-law distributed one.
    
    # Generate uniform random numbers in [0, 1]
    u = np.random.uniform(0, 1, n)
    
    # The power-law index from the PDF, used in the CDF calculation
    k = slope + 1
    
    if k == 0: # This corresponds to a slope of -1
        # Special case for P(x) ~ 1/x, which is a log-uniform distribution.
        log_lower = np.log(lower)
        log_upper = np.log(upper)
        return np.exp(u * (log_lower - log_upper) + log_upper)

    # Apply the inverse transform formula: x = [u * (x_max^k - x_min^k) + x_min^k] ^ (1/k)
    samples = (u * (lower**k - upper**k) + upper**k)**(1/k)
    return samples

def get_fluence_interval(width):
    lower_bound = (1.3*(width**0.5))
    return lower_bound,lower_bound*5

def noise_std(array):
    # The PRESTO toolchain (prepdata) mean-subtracts the time series before
    # processing. Therefore, to correctly estimate the PRESTO SNR, we must
    # characterize the noise using its standard deviation, which is insensitive
    # to the DC offset of the raw data.
    return np.nanstd(array, axis=1)

def check(out_name):
    '''
    Checks if out_name + .fil exists or not 
    out_name : pathlib object
    returns pathlib object  
    '''
    i = 0
    temp = out_name
    while temp.with_suffix('.fil').is_file():
        temp1 = out_name.stem + '_' + str(i)
        temp = out_name.with_name(temp1)
        i += 1
    return temp.with_suffix('.fil')

def get_disp_delay(freq_lo, freq_hi, DM):
    """Estimate the dispersion time delay (in ms) between the high and low freq 
    channel for source at given DM.
    Args:
        freq_lo (float): Frequency corresponding to the lowest channel in GHz.
        freq_hi (float): Frequency corresponding to the highest channel in GHz.
        DM (float): Source DM.
    Returns:
        float: Time delay in sec.
    """
    return (4.15*(1/freq_lo**2 - 1/freq_hi**2)*DM)/1000

def genrate_input_param(n, total_time, fre, nchan, config):
    input_param = np.zeros([n, 8])
    
    # Load parameters from config
    # Handle DM - can be fixed or variable
    if config['dm']['mode'] == 'variable':
        dm = np.random.uniform(config['dm']['lower'], config['dm']['upper'], n)
    else:
        dm = np.full(n, config['dm']['value'])
    
    # Handle Width generation based on config
    width_config = config.get('width', {})
    width_mode = width_config.get('mode', 'uniform')  # Default to uniform for backward compatibility

    if width_mode == 'array':
        width_values = width_config.get('values')
        if not width_values:
            raise ValueError("Width mode is 'array' but no 'values' are provided in the config.")
        width_ms = np.random.choice(width_values, n)
    else:  # Default to uniform distribution
        width_ms = np.random.uniform(width_config['lower'], width_config['upper'], n)
    width = width_ms / 1000.0  # Convert ms to seconds

    BW = np.random.uniform(config['bandwidth']['lower'], config['bandwidth']['upper'], n) #in chanel_number
    central_frequency = np.random.uniform(
        config['central_frequency']['lower_offset'], 
        nchan - config['central_frequency']['upper_offset'], 
        n
    )
    
    # Generate target SNR using config parameters
    if config['snr']['distribution'] == 'power_law':
        target_snr = power_law_distribution(
            config['snr']['lower'],
            config['snr']['upper'],
            n,
            slope=config['snr']['power_law_slope']
        )
    else:
        target_snr = np.random.uniform(config['snr']['lower'], config['snr']['upper'], n)
    
    # Handle scattering measure - can be fixed or variable
    if config['scattering_measure']['mode'] == 'variable':
        sm = np.random.uniform(config['scattering_measure']['lower'], config['scattering_measure']['upper'], n)
    else:
        sm = config['scattering_measure']['value']
    
    spectral_index = config['spectral_index']['value']
    
    total_dm_delay = get_disp_delay(fre[0], fre[-1], np.sum(dm))
    print(total_time,total_dm_delay)
    residual_time = total_time - np.abs(total_dm_delay) - 30
    if residual_time < 0:
        warnings.warn(
            "################################################## Too Many Injections #####################################################")
    diff_bet_two_burst = residual_time/n
    for i in range(n):
        if i == 0:
            input_param[0, 0] = config['time_spacing']['initial_time']
        else:
            input_param[i, 0] = np.abs(get_disp_delay(
                fre[0], fre[-1], np.sum(dm[:i]))) + i*diff_bet_two_burst + config['time_spacing']['initial_time']
    input_param[:, 1] = dm
    input_param[:, 2] = width
    input_param[:, 3] = target_snr
    input_param[:, 4] = sm if isinstance(sm, (int, float)) else sm
    input_param[:, 5] = spectral_index
    input_param[:, 6] = BW
    input_param[:, 7] = central_frequency
    return input_param

def _dedisperse_for_vis(
    data_array: np.ndarray,
    freqs_ghz: np.ndarray,
    dm: float,
    tsamp_s: float,
    ref_freq_ghz: float
) -> np.ndarray:
    """
    Dedisperses a 2D array for visualization using circular shifts (np.roll)
    to preserve the noise baseline. This is a local version to avoid modifying
    the main simulator's behavior, which uses zero-padding.
    """
    dedispersed_array = np.zeros_like(data_array)
    # Use the existing get_disp_delay function which works with arrays
    delays_s = get_disp_delay(freqs_ghz, ref_freq_ghz, dm)
    shifts = np.round(delays_s / tsamp_s).astype(int)

    for i, shift in enumerate(shifts):
        # A positive shift corresponds to a time delay. To correct this, we shift
        # the channel's data to the left (a negative roll) to align it.
        dedispersed_array[i, :] = np.roll(data_array[i, :], -shift)

    return dedispersed_array

def inject(data, inject_numpy_file, index_of_injection, samp, k, fre_in_GHz, samp_ms, T_total_till_now=0, visualize=False, config=None):
    '''
    This function will inject pulse with given pa.rametrs in input data
    data = numpy data array 0,0 corresponds to lowest frequency and lowest time bin 
    '''
    tobs = (int(len(data.data[0]))*(samp_ms/1000))
    nfreq = int(len(data.data[:, 0]))
    freq_lo_MHz = fre_in_GHz[-1]*1000
    freq_hi_MHz = fre_in_GHz[0]*1000
    dm = inject_numpy_file[index_of_injection, 1]
    sm = inject_numpy_file[index_of_injection, 4]  # 4
    intrinsic_width = inject_numpy_file[index_of_injection, 2]  # in sec
    target_snr = inject_numpy_file[index_of_injection, 3]
    spectral_index = inject_numpy_file[index_of_injection, 5]
    central_frequency = inject_numpy_file[index_of_injection, 7]
    BW = inject_numpy_file[index_of_injection, 6]
    width_in_samples = max(1, int(np.round(intrinsic_width / (samp_ms / 1000))))
    time_in_sec = inject_numpy_file[index_of_injection, 0]
    sample_number_arr_time = int(time_in_sec/((samp_ms/1000)))
    number_of_samples_in_burst = int(
        (get_disp_delay(freq_lo_MHz/1000, freq_hi_MHz/1000, dm))/((samp_ms/1000))) + 2*width_in_samples
    if number_of_samples_in_burst < 1024:
        number_of_samples_in_burst = 1024
 
    noise_std_per_channel = noise_std(
        data.data[:, sample_number_arr_time-number_of_samples_in_burst:sample_number_arr_time + number_of_samples_in_burst])
    
    bandwidth_mode = 'broadband'
    if config and 'bandwidth' in config:
        bandwidth_mode = config['bandwidth'].get('mode', 'narrowband')

    # The simulator function now handles bandpass shaping and returns the final pulse array
    # and the *actual* SNR after all effects have been applied.
    # The 'target_snr' parameter here is used as the target SNR before bandpass effects.
    nump_array, final_snr = generate_dispersed_gaussian_pulse(
        nchan=nfreq,
        ntime=2*number_of_samples_in_burst,
        tsamp_s=samp_ms/1000,
        freq_lo_ghz=freq_lo_MHz/1000,
        freq_hi_ghz=freq_hi_MHz/1000,
        dm=dm,
        width_s=intrinsic_width,
        injected_snr=target_snr,
        std_per_channel=noise_std_per_channel,
        scattering_ms=sm,
        ref_freq_ghz=freq_hi_MHz/1000,
        bandwidth_mode=bandwidth_mode,
        bw_chan=BW,
        central_freq_chan=central_frequency,
    )   
    # The pulse is now added to the data. The simulator centers the pulse in the middle
    # of the `nump_array`, and we inject it centered around `sample_number_arr_time`.
    data.data[:,sample_number_arr_time - number_of_samples_in_burst:sample_number_arr_time + number_of_samples_in_burst] += nump_array
    
    # The SNR returned by the simulator is the one we want to log, as it accounts for all effects.
    snr = final_snr

    if visualize:
        # 1. Extract the data window for plotting. This window contains the injected pulse.
        start_sample = sample_number_arr_time - 64
        end_sample = sample_number_arr_time + number_of_samples_in_burst
        plot_data = data.data[:, start_sample:end_sample]
        
        # 2. Create time axis for the plot in seconds
        tsamp_s = samp_ms / 1000.0
        time_axis_s = np.arange(plot_data.shape[1]) * tsamp_s
        
        # 3. Dedisperse the data window to get the time series
        ref_freq_ghz = freq_hi_MHz / 1000.0
        dedispersed_data = _dedisperse_for_vis(
            plot_data, fre_in_GHz, dm, tsamp_s, ref_freq_ghz
        )
        time_series = np.sum(dedispersed_data, axis=0)
        
        # 4. Create the plot with two subplots (time series and waterfall)
        fig, (ax_ts, ax_waterfall) = plt.subplots(
            2, 1, figsize=(8, 6), sharex=True, 
            gridspec_kw={'height_ratios': [1, 3], 'hspace': 0}
        )
        
        # Plot 1: Time series on top
        ax_ts.plot(time_axis_s, time_series, color='black', linewidth=1.0)
        ax_ts.set_ylabel('Intensity \n (arb. units)')
        ax_ts.margins(x=0)
        # Hide y-axis ticks and labels for a cleaner look
        ax_ts.tick_params(axis='y', labelleft=False, left=False)
        
        # Plot 2: Waterfall plot on bottom
        ax_waterfall.imshow(
            plot_data, 
            aspect='auto', 
            cmap='viridis',
            extent=[time_axis_s[0], time_axis_s[-1], freq_lo_MHz/1000, freq_hi_MHz/1000]
        )
        ax_waterfall.set_xlabel('Time (s)')
        ax_waterfall.set_ylabel('Frequency (GHz)')
        
        # Save the figure
        plt.savefig(f'injected_pulse_DM_{dm:.2f}_SNR_{snr:.2f}.png', dpi=300, bbox_inches='tight')
        plt.close(fig)
    return data, snr, time_in_sec


def Pulse_injection(args,s):
    # Load configuration
    config = load_config(args.config)
    
    # Make a log file
    file_name = args.filepath + '.fil'
    fname_ = args.filepath
    file1 = open(args.log, 'a')

    your_object = your.Your(file_name)
    bw = np.abs(your_object.your_header.bw)
    low = your_object.your_header.center_freq-(bw/2)
    print(low)
    width_channel = np.abs(your_object.your_header.bw /
                           your_object.your_header.native_nchans)
    nchan = your_object.your_header.native_nchans
    freq = np.linspace(low+bw, low, nchan)
    freq = freq*1e-3  # in GHz

    downsample = 1
    samp_ms = your_object.your_header.tsamp*1000

    del(your_object)
    if file1.tell() == 0:
        file1.writelines(
            '#outfile_name, final_snr, time_in_sec, dm, width_s, target_snr, sm, spectral_index, central_freq_chan, bw_chan\n')
    filepath_obj = pathlib.Path(args.filepath)
    datafile = FilReader(
        args.filepath+'.fil')

    header_dict = sigproc.parse_header(datafile.filename)
    #datafile.header.nsamples = datafile.header.nsamples/40 
    header_dict['nsamples'] = int(datafile.header.nsamples/args.spilt)
    ##datafile.header.nsamples = int(datafile.header.nsamples/args.spilt)
    object.__setattr__(datafile.header, "nsamples", int(datafile.header.nsamples / args.spilt))
    print(datafile.header.nsamples)
    data = FilterbankBlock(
        np.zeros((datafile.header.nchans, datafile.header.nsamples)), datafile.header)
    print(datafile.header.nsamples*s, datafile.header.nsamples*(s+1))
    #datafile.header.nsamples = int(datafile.header.nsamples/args.spilt)
    print(int((datafile.header.nsamples*datafile.header.tsamp)*s), datafile.header.nsamples)
    data = datafile.read_block(int((datafile.header.nsamples*datafile.header.tsamp)*s), datafile.header.nsamples)
    samp = 1

    print('filterbank data loaded')
    if args.p:
        inject_numpy_file = np.loadtxt(args.p)
    else:
        inject_numpy_file = genrate_input_param(
            args.n, datafile.header.nsamples*datafile.header.tsamp, freq, nchan, config)
    
    print('Injection param created')
    print(inject_numpy_file)
    for i in range(args.nf):
        # data = copy.deepcopy(data_copy)
        # sub_dir_path = filepath_obj.relative_to(
        #    filepath_obj.parent.parent.parent.parent.parent.parent.parent)
        outfile = pathlib.Path(args.o)  # .joinpath(sub_dir_path).parent
        os.makedirs(str(outfile), exist_ok=True)
        outfile_name_temp = filepath_obj.name +'_injected'+ '_npulses_' + str(args.n) + "_avgDM_" + str(
            "{:0>7.2f}".format(np.mean(inject_numpy_file[i:i+args.n, 1]))).replace('.', '_')
        outfname = outfile / outfile_name_temp
        outfname = check(outfname)
        
        file2 = open(str(outfname).replace('.fil', '.injinf'), 'w')
        file2.writelines(
            '#DM final_snr time_s width_s target_snr\n')
        for k in range(args.n):
            j = i*args.n+k
            print(k)
            data, snr, time_in_sec = inject(
                data, inject_numpy_file, j, samp, k, freq, samp_ms, 0, visualize=args.visualize, config=config)
            print(snr)
            # Columns: outfname, final_snr, time, dm, width, target_snr, sm, spec_idx, cent_freq, bw
            file1.writelines([str(outfname), ' ', str(round(snr, 3)), ' ', str(round(time_in_sec, 3)), ' ', str(inject_numpy_file[j, 1]), ' ',
                              str(inject_numpy_file[j, 2]), ' ', str(inject_numpy_file[j, 3]), ' ', str(inject_numpy_file[j, 4]), ' ',
                              str(inject_numpy_file[j, 5]), ' ', str(inject_numpy_file[j, 7]), ' ', str(inject_numpy_file[j, 6]), ' \n'] )
            # Columns: dm, final_snr, time, width, target_snr
            file2.writelines([str(inject_numpy_file[j, 1]), ' ', str(round(snr, 3)), ' ', str(round(time_in_sec, 3)), ' ',
                              str(inject_numpy_file[j, 2]), ' ', str(round(inject_numpy_file[j, 3], 4)), '\n'])
        file2.close()
        print(header_dict) 
        '''
        header_dict['nbits'] = 16
        data.data[:,:] = data.data.astype(np.int16)
        fil_fileobj = FileWriter(str(outfname),  mode="w", nbits = 16  )
        fil_fileobj.write( sigproc.encode_header(header_dict) )
        array_testing = data.data.transpose().flatten()
        #array_testing = array_testing.astype(np.int16)
        fil_fileobj.cwrite( array_testing )
        fil_fileobj.close()
        file1.close()
        '''
        header_dict['nbits'] = 32
        fil_fileobj = FileWriter(str(outfname),  mode="w", nbits = 32  )
        fil_fileobj.write( sigproc.encode_header(header_dict) )
        array_testing = data.data.transpose().flatten()
        fil_fileobj.cwrite( array_testing )
        fil_fileobj.close()
        file1.close()
    return

def main():
    parser = argparse.ArgumentParser(description='Injects simulated pulses into filterbank data.')
    parser.add_argument('--filepath', type=str, required=True,
                        help='Filterbank file path without ".fil"')
    parser.add_argument('--log', type=str, required=True, help='log file')
    parser.add_argument('--p', type=str, required=False, help='Parameter file')
    parser.add_argument('--o', type=str, required=True,
                        help='Output directory')
    parser.add_argument('--n', type=int, required=False, default=1,
                        help='Number of pulses to inject per file')
    parser.add_argument('--nf', type=int, required=True,
                        help='Number of output files to make ".fil" ')
    parser.add_argument('--spilt', type=int, required=True,
                        help='Number of output files to make split  ".fil" ')
    parser.add_argument('--config', type=str, required=False, default='injection_config.json',
                        help='Path to JSON configuration file with parameter bounds (default: injection_config.json)')
    parser.add_argument('--visualize', action='store_true',
                        help='Save plots of injected pulses')
    args = parser.parse_args()
    for s in range(args.spilt):
            Pulse_injection(args,s)

if __name__ == '__main__':
    main()
