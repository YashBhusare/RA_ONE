<p align="center">
  <img src="logo.png" alt="RA_ONE logo" width="220" />
</p>

# RA_ONE: A Radio Burst Injection and Recovery Analysis Toolkit

`RA_ONE` is a Python toolkit for injecting realistic single-pulse radio bursts into filterbank data and analysing recovery by search pipelines (e.g., PRESTO).

## Core Components

- **`Ra_one.py`** — pulse injection script (uses `simulator.py`).
- **`G_one.py`** — recovery analysis and plotting.
- **`Raftaar.py`** — wrapper to run injection → PRESTO → analysis.
- Utilities: `plot_combined_completeness.py`, `completeness_analyser.py`, `converter.py`, `singlepulsetocsv.py`, etc.

## Requirements

- Python packages are listed in `requirements.txt` (install with `pip install -r requirements.txt`).
- External: a working PRESTO installation with `prepdata` and `single_pulse_search.py` available in `PATH`.

## Installation

1. Clone the repository:
```bash
git clone https://github.com/your-username/RA_ONE.git
cd RA_ONE
```

2. Install the package (editable mode recommended):
```bash
pip install -e .
```

After installation the CLI scripts (`ra-one`, `g-one`, `raftaar`, ...) will be available.

## Docker (optional)

A published image is available on Docker Hub under `ybhusare/ra-one`. Users can pull and run it directly (no GitHub push needed):

```bash
docker pull ybhusare/ra-one:latest
docker run -it --rm -v "${PWD}:/data" ybhusare/ra-one:latest bash
```


## Workflow

The `raftaar` script runs the full pipeline. Example:
```bash
raftaar /path/to/input.fil --output-dir ./analysis_run --n-pulses 500 --sps-threshold 6.0 --config injection_config.json
```

Manual steps:
- Inject: `ra-one --filepath /path/to/input --log master.log --o injection_output --n 500 --nf 1 --spilt 1 --config injection_config.json`
- PRESTO: `prepdata -o presto_search -dm 100.0 injection_output/input_injected...fil` then `single_pulse_search.py -t 6.0 presto_search.dat`
- Analyse: `g-one --inject-file ...injinf --detection-file ...singlepulse --det-format presto --outdir analysis_output --visualize`

## Configuration & Editing

Configuration for injections is stored in the repository root file [injection_config.json](injection_config.json). Edit that file to change how pulses are generated and sampled. Key fields and where to change them:

- `dm`: Dispersion Measure settings
  - `mode`: `fixed` or `variable` — change `mode` and `value`/`lower`/`upper` in `injection_config.json`.
  - `value`: DM used when `mode` is `fixed`.

- `width`: Pulse width (ms)
  - `mode`: `array` (choose from `values`) or `uniform` (sample between `lower` and `upper`). Edit `values` or bounds in `injection_config.json`.

- `snr`: Target signal-to-noise distribution
  - `distribution`: `power_law` or `uniform`.
  - `lower`, `upper`, `power_law_slope`: tune these in `injection_config.json`.

- Other parameters you can edit directly in `injection_config.json`:
  - `bandwidth`, `central_frequency`, `scattering_measure`, `spectral_index`, `time_spacing` — these control spectral/profile and timing properties.

Example: open and edit the file in-place:

```bash
# edit the config (any editor)
nano injection_config.json
# or use an IDE/text editor and save changes
```

Usage notes:
- To run injections with a config file: use the `--config` option with `ra-one` (or supply the same flag to `Ra_one.py` if running directly). Example:

```bash
# when package is installed and entrypoints are available
ra-one --filepath /path/to/input.fil --config injection_config.json --o injection_output

# or run the script directly (from repo root)
python Ra_one.py --filepath /path/to/input.fil --config injection_config.json --o injection_output
```

- `raftaar` can be used to run the full pipeline (injection → PRESTO → analysis) but is intentionally described briefly here — it accepts the same `--config` flag.

- If you prefer to test inside Docker (recommended when host lacks some dependencies), the published image already includes PRESTO and transient_X installed, so PRESTO steps and transient-related utilities are available inside the container.



## Post-Analysis

- `plot_combined_completeness.py` and `completeness_analyser.py` help combine and visualise results across runs.

---

If you want I can also add a concise example command showing a minimal end-to-end run, or add a `Dockerfile` for reproducible builds. Which would you prefer?
