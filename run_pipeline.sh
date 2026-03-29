#!/bin/bash
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONFIG_FILE="$SCRIPT_DIR/pipeline_config.yml"

# Check for yq and install if missing
if ! command -v yq &>/dev/null; then
    echo "'yq' not found. Installing..."
    if [[ "$OSTYPE" == "darwin"* ]]; then
        # macOS: use Homebrew
        if ! command -v brew &>/dev/null; then
            echo "Error: Homebrew not found. Install it from https://brew.sh then re-run."
            exit 1
        fi
        brew install yq
    else
        # Linux: download prebuilt binary
        YQ_BIN="/usr/local/bin/yq"
        sudo wget -q https://github.com/mikefarah/yq/releases/latest/download/yq_linux_amd64 -O "$YQ_BIN"
        sudo chmod +x "$YQ_BIN"
    fi
    if ! command -v yq &>/dev/null; then
        echo "Error: Failed to install yq."
        exit 1
    fi
    echo "'yq' installed successfully."
fi

# ---------------------------
# Load flags from YAML config
# ---------------------------
RUN_OHIOT1DM_PRE=$(yq e '.ohiot1dm_preprocess' "$CONFIG_FILE")
RUN_OHIOT1DM_MOD=$(yq e '.ohiot1dm_model' "$CONFIG_FILE")
RUN_OHIOT1DM_DATASET=$(yq e '.ohiot1dm_dataset' "$CONFIG_FILE")

RUN_MIMIC_PRE=$(yq e '.mimic_preprocess' "$CONFIG_FILE")
RUN_MIMIC_MOD_TRAIN=$(yq e '.mimic_model_train' "$CONFIG_FILE")
RUN_MIMIC_MOD_TEST=$(yq e '.mimic_model_test' "$CONFIG_FILE")

RUN_PHYSIONET_MOD=$(yq e '.physionetcinc_model' "$CONFIG_FILE")
RUN_PHYSIONET_DATASET=$(yq e '.physionetcinc_dataset' "$CONFIG_FILE")

# ---------------------------
# Parse command-line overrides
# ---------------------------
for arg in "$@"; do
    case $arg in
        --ohiot1dm_preprocess=*) RUN_OHIOT1DM_PRE="${arg#*=}" ;;
        --ohiot1dm_model=*)      RUN_OHIOT1DM_MOD="${arg#*=}" ;;
        --ohiot1dm_dataset=*)    RUN_OHIOT1DM_DATASET="${arg#*=}" ;;
        --mimic_preprocess=*)    RUN_MIMIC_PRE="${arg#*=}" ;;
        --mimic_model_train=*)   RUN_MIMIC_MOD_TRAIN="${arg#*=}" ;;
        --mimic_model_test=*)    RUN_MIMIC_MOD_TEST="${arg#*=}" ;;
        --physionetcinc_model=*)      RUN_PHYSIONET_MOD="${arg#*=}" ;;
        --physionetcinc_dataset=*)    RUN_PHYSIONET_DATASET="${arg#*=}" ;;
        -h|--help)
            echo "Usage: $0 [--ohiot1dm_preprocess=true|false] [--ohiot1dm_model=true|false]"
            echo "       [--ohiot1dm_dataset=2018|2020]"
            echo "       [--mimic_preprocess=true|false] [--mimic_model_train=true|false] [--mimic_model_test=true|false]"
            echo "       [--physionetcinc_model=true|false]"
            echo "       [--physionetcinc_dataset=A|B]"
            exit 0
            ;;
        *) echo "Unknown option: $arg"; exit 1 ;;
    esac
done

# Convert all flags to lowercase
RUN_OHIOT1DM_PRE=$(echo "$RUN_OHIOT1DM_PRE" | tr '[:upper:]' '[:lower:]')
RUN_OHIOT1DM_MOD=$(echo "$RUN_OHIOT1DM_MOD" | tr '[:upper:]' '[:lower:]')

RUN_MIMIC_PRE=$(echo "$RUN_MIMIC_PRE" | tr '[:upper:]' '[:lower:]')
RUN_MIMIC_MOD_TRAIN=$(echo "$RUN_MIMIC_MOD_TRAIN" | tr '[:upper:]' '[:lower:]')
RUN_MIMIC_MOD_TEST=$(echo "$RUN_MIMIC_MOD_TEST" | tr '[:upper:]' '[:lower:]')

RUN_PHYSIONET_MOD=$(echo "$RUN_PHYSIONET_MOD" | tr '[:upper:]' '[:lower:]')

# ---------------------------
# Helper to run scripts inside environments
# ---------------------------
run_in_env() {
    local env_dir=$1
    local target_dir=$2
    local cmd=$3

    echo ">>> Activating environment $env_dir and running script in $target_dir ..."
    source "$SCRIPT_DIR/$target_dir/$env_dir/bin/activate"
    cd "$SCRIPT_DIR/$target_dir"
    eval "$cmd"
    cd "$SCRIPT_DIR"
    deactivate || true
}

# ---------------------------
# Run pipeline stages
# ---------------------------
if [ "$RUN_OHIOT1DM_PRE" = "true" ]; then
    echo "Preprocessing OhioT1DM dataset..."
    run_in_env "venv_ohiot1dm" "OhioT1DM" "python convert_data.py data/raw data/processed"
fi

if [ "$RUN_OHIOT1DM_MOD" = "true" ]; then
    echo "Running OhioT1DM model for dataset ${RUN_OHIOT1DM_DATASET}..."
    run_in_env "venv_ohiot1dm" "OhioT1DM" "python drtf.py data/processed/${RUN_OHIOT1DM_DATASET}data output/${RUN_OHIOT1DM_DATASET}"
fi

if [ "$RUN_MIMIC_PRE" = "true" ]; then
    echo "Preprocessing MIMIC dataset..."
    run_in_env "venv_mimic" "MIMIC" "jupyter nbconvert --execute --inplace mainPipeline.ipynb"
fi

if [ "$RUN_MIMIC_MOD_TRAIN" = "true" ]; then
    echo "Running MIMIC model training..."
    run_in_env "venv_mimic" "MIMIC" "python run.py --train_test 1"
fi

if [ "$RUN_MIMIC_MOD_TEST" = "true" ]; then
    echo "Running MIMIC model testing..."
    run_in_env "venv_mimic" "MIMIC" "python run.py --train_test 0"
fi

if [ "$RUN_PHYSIONET_MOD" = "true" ]; then
    echo "Running PhysioNetCinC model for dataset ${RUN_PHYSIONET_DATASET}..."
    run_in_env "venv_physionetcinc" "PhysioNetCinC" "python driver.py inputs/training_set${RUN_PHYSIONET_DATASET} outputs/training_set${RUN_PHYSIONET_DATASET}"
fi

echo "Pipeline completed successfully."
