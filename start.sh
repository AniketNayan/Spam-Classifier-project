#!/usr/bin/env bash
set -ex   # -e exit on error, -x print commands

echo "### START: startup script running on $(hostname) ###"

# ensure a local nltk_data folder exists
NLTK_DIR="./nltk_data"
mkdir -p "$NLTK_DIR"
ls -l .

echo ">>> Checking/downloading NLTK data..."
python - <<'PY'
import nltk, os
nltk.data.path.append(os.path.abspath("nltk_data"))

# list of packages we will ensure are present
needed = ['punkt', 'punkt_tab', 'stopwords']
for pkg in needed:
    try:
        nltk.data.find(pkg)
        print(pkg + " OK")
    except LookupError:
        print("Downloading " + pkg + "...")
        nltk.download(pkg, download_dir="nltk_data")
print("NLTK ready")
PY

echo ">>> Listing project files:"
ls -la

echo ">>> Python version:"
python --version

echo ">>> Pip freeze (first 40 lines):"
pip freeze | sed -n '1,40p' || true

echo ">>> Starting gunicorn (binding to 0.0.0.0:${PORT:-10000})..."
exec gunicorn app:app \
  --bind 0.0.0.0:${PORT:-10000} \
  --workers 2 \
  --threads 2 \
  --log-level info \
  --access-logfile - \
  --error-logfile -
