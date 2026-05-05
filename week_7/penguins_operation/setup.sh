#!/usr/bin/env bash
set -e

echo "==> Creating virtual environment in .venv/"
python3 -m venv .venv

echo "==> Installing dependencies..."
.venv/bin/pip install --quiet --upgrade pip
.venv/bin/pip install --quiet -r requirements.txt

echo ""
echo "Done! Activate the environment with:"
echo "  source .venv/bin/activate"
echo ""
echo "Then start the activity:"
echo "  python penguin_flow.py run --max_depth 2"
