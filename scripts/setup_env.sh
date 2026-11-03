#!/usr/bin/env bash
# Create the project virtualenv and install dependencies.
#
# The venv lives OUTSIDE this folder by default because the project sits in a
# OneDrive-synced directory and a torch install is tens of thousands of files.
# Override with VENV_DIR=/some/path.
set -euo pipefail
cd "$(dirname "$0")/.."

VENV_DIR="${VENV_DIR:-$HOME/.venvs/magnet-ssm}"

uv venv --python 3.12 "$VENV_DIR"
uv pip install --python "$VENV_DIR/bin/python" -r requirements.txt

echo
echo "Activate with:  source $VENV_DIR/bin/activate"
echo "Then run:       python -m pytest -q"
