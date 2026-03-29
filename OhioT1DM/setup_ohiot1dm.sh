#!/usr/bin/env bash
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BASE_DIR="$SCRIPT_DIR"
VENV_DIR="$BASE_DIR/venv_ohiot1dm"

echo ">>> Creating OhioT1DM environment"
python3.9 -m venv "$VENV_DIR"

echo ">>> Activating OhioT1DM environment"
source "$VENV_DIR/bin/activate"

PIP="$VENV_DIR/bin/pip"

echo ">>> Installing requirements"
$PIP install -r "$SCRIPT_DIR/requirements.txt"

echo ">>> Copying OhioT1DM data"
mkdir -p "$BASE_DIR/data"
cp -r ~/Downloads/OhioT1DM/raw_data "$BASE_DIR/data/raw" || true    # Important: Replace ~/Downloads/OhioT1DM/raw_data with the actual path to the raw data
# mkdir -p "$BASE_DIR/data/processed"
# cp -r ~/Downloads/OhioT1DM/processed_data/2020data "$BASE_DIR/data/processed/" || true
# cp -r ~/Downloads/OhioT1DM/processed_data/2018data "$BASE_DIR/data/processed/" || true

# echo ">>> Copying pretrained models"
# cp -r ~/Downloads/OhioT1DM/models/PRETRAINS "$BASE_DIR/" || true
echo ">>> Downloading pretrained models"
$PIP install gdown
gdown 1VXO2wT7M0htjVGbd6hk2935q52AteI7A -O "$BASE_DIR/PRETRAINS.zip" || true
unzip -o "$BASE_DIR/PRETRAINS.zip" -d "$BASE_DIR/" || true

echo ">>> Installing URET"
cd "$SCRIPT_DIR/URET"
$PIP install -e .
cd "$BASE_DIR"

echo ">>> OhioT1DM setup complete"