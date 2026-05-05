#!/usr/bin/env bash
# Sets up the Getflix Adventure environment.
# Run once from the netflix_adventure/ directory:
#   bash setup.sh
set -euo pipefail
cd "$(dirname "$0")"

echo "==> Creating virtual environment (.venv)..."
python3 -m venv .venv

echo "==> Installing dependencies..."
.venv/bin/pip install --upgrade pip --quiet
.venv/bin/pip install -r requirements.txt

echo "==> Downloading MovieLens 100K data..."
.venv/bin/python download_data.py

echo ""
echo "Setup complete."
echo ""
echo "Activate the environment:"
echo "  source .venv/bin/activate"
echo ""
echo "Then open the walkthroughs and work through them in order:"
echo "  open explore.html"
echo "  open clustering_walkthrough.html    # Step 1 of 3"
echo "  open recommender_walkthrough.html   # Step 2 of 3"
echo "  open timeseries_walkthrough.html    # Step 3 of 3"
echo ""
echo "Each walkthrough ends with a Metaflow command to run — that trains"
echo "your model and saves it so the Flask app can serve it."
echo ""
echo "Once you've run all three pipelines, start the Flask app:"
echo "  python flask_app/app.py"
echo "  open flask_app/static/index.html"
