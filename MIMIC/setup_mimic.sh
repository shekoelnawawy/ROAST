#!/usr/bin/env bash
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BASE_DIR="$SCRIPT_DIR"
VENV_DIR="$BASE_DIR/venv_mimic"

echo ">>> Creating MIMIC environment"
python3.7 -m venv "$VENV_DIR"

echo ">>> Activating MIMIC environment"
source "$VENV_DIR/bin/activate"

PIP="$VENV_DIR/bin/pip"

echo ">>> Installing requirements"
$PIP install -r "$SCRIPT_DIR/requirements.txt"

echo ">>> Installing PyTorch"
$PIP install torch==1.12.1+cu113 torchvision==0.13.1+cu113 \
  -f https://download.pytorch.org/whl/torch_stable.html

echo ">>> Copying MIMIC data"
cp -r ~/Downloads/MIMIC/original/data "$BASE_DIR/" || true
cp -r ~/Downloads/MIMIC/original/mimiciv "$BASE_DIR/" || true
cp -r ~/Downloads/MIMIC/original/saved_models "$BASE_DIR/" || true
cp -r ~/Downloads/MIMIC/original/utils "$BASE_DIR/" || true

echo ">>> Installing URET"
cd "$SCRIPT_DIR/URET"
$PIP install -e .
cd "$BASE_DIR"

echo ">>> MIMIC setup complete"