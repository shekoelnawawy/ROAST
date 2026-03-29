#!/usr/bin/env bash
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo ">>> Running all setup scripts"

bash "$SCRIPT_DIR/OhioT1DM/setup_ohiot1dm.sh"
bash "$SCRIPT_DIR/MIMIC/setup_mimic.sh"
bash "$SCRIPT_DIR/PhysioNetCinC/setup_physionetcinc.sh"

echo ">>> All setups completed"