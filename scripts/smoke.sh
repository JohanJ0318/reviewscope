#!/usr/bin/env bash
# End-to-end run on synthetic data with small models. CI runs this; it takes a few minutes on CPU.
# For the real dataset, see "Running on the real data" in the README.
set -euo pipefail

BASE=${BASE:-sentence-transformers/paraphrase-MiniLM-L3-v2}
EMBEDDER=${EMBEDDER:-sentence-transformers/paraphrase-MiniLM-L3-v2}

python -m reviewscope.sample_data --reviews 20000 --out data/raw/sample.csv
python -m reviewscope.etl_spark --raw data/raw/sample.csv --out data/processed
python -m reviewscope.train_lora --base "$BASE" --mode lora --epochs 1 --max-len 64 --max-train 10000
python -m reviewscope.train_lora --base "$BASE" --mode head --epochs 1 --max-len 64 --max-train 10000
python -m reviewscope.calibration
python -m reviewscope.export_onnx --embedder "$EMBEDDER" --max-len 64 --repeats 20
python -m reviewscope.build_index
python -m reviewscope.rag_bedrock --app-id "$(python -c "import pandas as pd; print(pd.read_parquet('artifacts/index/meta.parquet').app_id.mode()[0])")" \
  --dry-run "What do players say about performance?" | head -5
