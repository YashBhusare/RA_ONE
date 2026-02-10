<p align="center">
  <img src="logo.png" alt="RA_ONE logo" width="220" />
</p>

# RA_ONE: A Radio Burst Injection and Recovery Analysis Toolkit

`RA_ONE` is a comprehensive Python-based toolkit designed for testing and characterizing the performance of single-pulse (or "burst") search pipelines in radio astronomy. It provides a full-cycle workflow:

1. **Injection**: Generates and injects physically realistic, simulated radio bursts into filterbank data (`.fil` files).
2. **Automation**: Includes a wrapper script to automate the process of running an external search pipeline (e.g., PRESTO) on the injected data.
3. **Analysis**: Compares the catalog of injected pulses against the candidates detected by the search pipeline to produce detailed recovery statistics and diagnostic plots.

This allows astronomers and software developers to quantify the completeness and sensitivity of their detection pipelines as a function of pulse parameters like Signal-to-Noise Ratio (S/N), pulse width, and Dispersion Measure (DM).

## Core Components

- **`Ra_one.py`**: The pulse injection script. It uses `simulator.py` to create pulses with effects like dispersion, scattering, and intra-channel smearing. Injection parameters are controlled via `injection_config.json`.
- **`G_one.py`**: The recovery analysis script. It matches detected candidates to injected pulses and generates a suite of outputs, including completeness curves, S/N comparison plots, and recovery threshold reports.
- **`Raftaar.py`**: An automated workflow script that orchestrates the entire `inject -> search -> analyze` pipeline, providing a simple entry point for a full analysis run.
- **Analysis Utilities**: Includes scripts like `plot_combined_completeness.py` to aggregate results from multiple runs and `completeness_analyser.py` for alternative analysis methods.

## Requirements

### Python Dependencies

The Python packages required are listed in `requirements.txt`. You can install them using pip:

```bash
pip install -r requirements.txt
```

### External Dependencies

This toolkit is designed to wrap around an existing search pipeline. **PRESTO** is the default supported pipeline.

- You must have a working installation of **PRESTO**.
- The PRESTO executables (`prepdata`, `single_pulse_search.py`) must be available in your system's `PATH`.

## Installation

To make the scripts available as system-wide command-line tools, you can install `RA_ONE` as a Python package.

1. **Clone the repository:**
    ```bash
    git clone https://github.com/your-username/RA_ONE.git
    cd RA_ONE
    ```

2. **Install the package:**
    Use `pip` to install the project in editable mode. This is recommended as it allows you to modify the code without reinstalling.
    ```bash
    pip install -e .
    ```

After installation, the scripts will be available as commands (e.g., `raftaar`, `g-one`).

## Workflow

The easiest way to run a full analysis is with the `Raftaar.py` wrapper script.

### Automated Workflow with `raftaar`

The `raftaar` command automates all steps: injection, de-dispersion with `prepdata`, searching with `single_pulse_search.py`, and analysis with `g-one`.

1. **Configure Injection Parameters**: Edit the `injection_config.json` file to define the properties of the pulses you want to inject (DM, width, S/N distribution, etc.). For `raftaar`, the DM mode must be set to `"fixed"`.

2. **Run the Pipeline**:
    ```bash
    raftaar /path/to/your/input.fil \
        --output-dir ./analysis_run_1ms \
        --n-pulses 500 \
        --sps-threshold 6.0 \
        --max-width-ms 10.0 \
        --config injection_config.json
    ```

This will create an `analysis_run_1ms` directory with the following structure:

```
analysis_run_1ms/
├── 1_injection_output/   # Injected .fil and .injinf files
├── 2_presto_output/      # PRESTO .dat and .singlepulse files
└── 3_analysis_results/   # G_one.py outputs (plots, CSVs, JSON)
```

### Manual Workflow

You can also run each step manually for more control.

1. **Inject Pulses (`ra-one`)**:
    ```bash
    ra-one --filepath /path/to/input --log master.log --o injection_output --n 500 --nf 1 --spilt 1 --config injection_config.json
    ```
    This creates `injection_output/input_injected_...fil` and `...injinf`.

2. **Run Your Search Pipeline**:
    (Example with PRESTO)
    ```bash
    prepdata -o presto_search -dm 100.0 -nobary injection_output/input_injected_...fil
    single_pulse_search.py -t 6.0 presto_search.dat
    ```

3. **Analyze Detections (`g-one`)**:
    ```bash
    g-one --inject-file injection_output/input_injected_...injinf \
          --detection-file presto_search.singlepulse \
          --det-format presto \
          --outdir analysis_output \
          --visualize
    ```

## Post-Analysis

After running one or more analyses, you can use the helper scripts to combine and visualize the results:

- **`ra-completeness-plotter`**: Combines `completeness_data.csv` from multiple runs into a single plot.
- **`ra-completeness-analyser`**: Performs a completeness analysis in the fluence-width plane.
