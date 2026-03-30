# ROAST (Risk-Aware Outlier Exposure Selective Training)

This is the code repository for the ROAST framework, which improves Anomaly Detector (AD) performance against evasion attacks in safety-critical domains like healthcare. Link to paper: [To be added once arXiv link is avaialable]


### Repository Overview
This repository unifies and orchestrates multiple model pipelines across three distinct time-series datasets used for the ROAST experiments:
- **OhioT1DM:** Continuous Glucose Monitoring (CGM) forecasting.
- **MIMIC:** Clinical time-series analysis (MIMIC-IV).
- **PhysioNetCinC:** Sepsis Early Prediction on ICU Data.

The project features automated data preprocessing, isolated virtual environments, unified adversarial attack integration (e.g., URET and FGSM), risk profiling, and selective training for anomaly detectors.

## Dataset Acquisition

### 1. OhioT1DM
The OhioT1DM dataset is protected under an NDA and therefore we cannot disclose it publicly. However, it can be obtained from Ohio University by following the instructions listed [here](https://webpages.charlotte.edu/rbunescu/data/ohiot1dm/OhioT1DM-dataset.html).
- **Action Required:** Place the dataset at the following path: `./OhioT1DM/data/raw/`

### 2. MIMIC-IV
The MIMIC dataset access is restricted and subject to meeting the following requirements:
- [Be a PhysioNet credentialed user](https://physionet.org/settings/credentialing/)
- [Complete required training: CITI Data or Specimens Only Research](https://physionet.org/content/mimiciv/view-required-training/2.0/#1)
- [Submit your training](https://physionet.org/settings/training/)
- [Sign the data use agreement for the project](https://physionet.org/sign-dua/mimiciv/2.0/)
- Download the dataset using the following command:
    ```bash
    wget -r -N -c -np --user [YOUR_PHYSIONET_USERNAME] --ask-password https://physionet.org/files/mimiciv/2.0/
    ```
- **Action Required:** Place the dataset at the following path: `./MIMIC/mimiciv/`

### 3. PhysioNetCinC
The PhysioNetCinC sepsis prediction dataset is publicly available and can be accessed from the [PhysioNet website](https://physionet.org/content/challenge-2019/1.0.0/), as long as you conform to the terms of the [Creative Commons Attribution 4.0 International Public License](https://physionet.org/content/challenge-2019/view-license/1.0.0/). 

- Download the dataset using the following command:
    ```bash
    wget --user [YOUR_PHYSIONET_USERNAME] --ask-password -r -N -c -np -nH --cut-dirs=3 --reject "index.html*" https://physionet.org/files/challenge-2019/1.0.0/training/
    ```
- **Action Required:** Place the dataset at the following path: `./PhysioNetCinC/input/`

## Project Setup

Run the `setup.sh` script to build isolated Python virtual environments for each respective project and install their dependencies:

```bash
bash setup.sh
```

## Configuration

Control the entire execution pipeline through the central `pipeline_config.yml` configuration point. Here, you can toggle preprocessing and model training independently:

```yaml
# OhioT1DM Pipeline
ohiot1dm_preprocess: false
ohiot1dm_model: false
ohiot1dm_dataset: 2020  # Options: 2018, 2020
ohiot1dm_attack_type: "URET"    # Options: URET, FGSM

# MIMIC Pipeline
mimic_preprocess: false
mimic_model_train: false
mimic_model_test: false
mimic_attack_type: "URET"  # Options: URET, FGSM

# PhysioNetCinC Pipeline
physionetcinc_model: false
physionetcinc_dataset: "A"  # Options: A, B
physionetcinc_attack_type: "URET"  # Options: URET, FGSM
```

## Running the Pipeline

Once your configuration is configured, orchestrate execution using `run_pipeline.sh`. The script applies overrides directly from the command-line arguments:

```bash
# Run using the configuration provided in pipeline_config.yml
./run_pipeline.sh

# Or directly override specific flags on the command line
./run_pipeline.sh --ohiot1dm_dataset=2018 --ohiot1dm_model=true --mimic_model=false
```