import argparse
import os
import your
import pandas as pd
import numpy as np
def main():
    # Argument parsing
    parser = argparse.ArgumentParser(description="Convert .fil and .injinf files to CSV.")
    parser.add_argument("-f", "--fil_files", required=True, nargs='+', help="Paths to input .fil files")
    parser.add_argument("-i", "--info_files", required=True, nargs='+', help="Paths to input .injinf files")
    parser.add_argument("-o", "--output_dir", required=True, help="Directory to save output CSV file")
    parser.add_argument("-rfi", "--rfi", action='store_true', help="if 4th index contain width in sample")

    args = parser.parse_args()

    # Resolve paths
    fil_files = [os.path.abspath(f) for f in args.fil_files]
    info_files = [os.path.abspath(f) for f in args.info_files]
    output_dir = os.path.abspath(args.output_dir)

    # Sanity check
    if len(fil_files) != len(info_files):
        raise ValueError("Number of .fil and .injinf files must be equal.")

    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Determine output filename
    if len(fil_files) > 1:
        csv_file_path = os.path.join(output_dir, "combined_output.csv")
    else:
        base_name = os.path.splitext(os.path.basename(fil_files[0]))[0]
        csv_file_path = os.path.join(output_dir, base_name + ".csv")

    dataframes = []

    for fil_file, info_file in zip(fil_files, info_files):
        print(f" Processing FIL: {fil_file}")
        print(f" With INFO: {info_file}")

        # Load .injinf file
        info_df = pd.read_csv(info_file, sep='\s+')
        print(info_df)
        # Extract data
        DM = info_df.iloc[:, 0].values
        snr = info_df.iloc[:, 1].values
        stime = info_df.iloc[:, 2].values
        width_time = info_df.iloc[:, 3].values
        width_in_samp = info_df.iloc[:, 4].values
        label = np.ones(len(DM)) #changed to np.zeros, temp ones

        # Process .fil file
        your_object = your.Your(fil_file)
        tsamp = your_object.your_header.tsamp
        nchans = your_object.your_header.nchans
        print(nchans)
        width_samp = (width_time / tsamp) #converting sample width from sec to samples
        chans_to_rem = int(0.025*nchans)
        print(chans_to_rem)
        # Define the ranges to exclude
        mask = np.concatenate([
            np.arange(0, chans_to_rem+1),      # Channels 0 to ~30
            np.arange(nchans - chans_to_rem, nchans)  # Channels ~994 to 1024
        ])
        np.savetxt(f"{output_dir}/rfi_mask.txt", mask, fmt='%d')
        # Create DataFrame
        if args.rfi :
            width = np.array([int(np.log2(int(i))) for i in width_in_samp])
        else :
            width = np.array([int(np.log2(i)) for i in width_samp if i > 0])

        df = pd.DataFrame({
            "file": [fil_file] * len(DM),
            "snr": snr,
            "width": width,
            "dm": DM,
            "label": label,
            "stime": stime,
            "chan_mask_path": f"{output_dir}/rfi_mask.txt",
            "num_files": 1
        })

        dataframes.append(df)

    # Combine and save
    final_df = pd.concat(dataframes, ignore_index=True)
    final_df.to_csv(csv_file_path, index=False)

    print(f" CSV file saved at: {csv_file_path}")

if __name__ == "__main__":
    main()
