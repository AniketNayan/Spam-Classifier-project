#!/usr/bin/env bash
set -e

# ensure a local nltk_data folder exists and downloads required corpora if missing
NLTK_DIR="./nltk_data"
mkdir -p "$NLTK_DIR"

python - <<'PY'
import nltk, os, sys
nltk.data.path.append(os.path.abspath("nltk_data"))
needed = ['punkt', 'stopwords']
for pkg in needed:
    try:
        nltk.data.find(pkg)
    except LookupError:
        print(f"Downloading {pkg}...")
        nltk.download(pkg, download_dir="nltk_data")
print("NLTK ready")
PY

# start Gunicorn, binding to Render's $PORT so Render can route traffic
exec gunicorn app:app --bind 0.0.0.0:${PORT:-10000} --workers 2 --threads 2
