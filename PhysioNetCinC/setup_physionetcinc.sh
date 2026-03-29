#!/usr/bin/env bash
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BASE_DIR="$SCRIPT_DIR"
VENV_DIR="$BASE_DIR/venv_physionetcinc"

echo ">>> Creating PhysioNetCinc environment"
python3.9 -m venv "$VENV_DIR"

echo ">>> Activating PhysioNetCinc environment"
source "$VENV_DIR/bin/activate"

PIP="$VENV_DIR/bin/pip"

echo ">>> Installing requirements"
$PIP install -r "$SCRIPT_DIR/requirements.txt"

echo ">>> Copying PhysioNetCinc data"
rm -rf "$BASE_DIR/inputs" || true
cp -r ~/Downloads/Sepsis/files/challenge-2019/1.0.0/training "$BASE_DIR/inputs" || true

echo ">>> Installing URET"
cd "$SCRIPT_DIR/URET"
$PIP install -e .
cd "$BASE_DIR"

echo ">>> PhysioNetCinc setup complete"