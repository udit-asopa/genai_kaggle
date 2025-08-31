from pathlib import Path
import os

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROC_DIR = DATA_DIR / "processed"
MODELS_DIR = ROOT / "models"
INDEX_DIR = MODELS_DIR / "faiss_index"
OUTPUTS_DIR = ROOT / "outputs"

for d in [RAW_DIR, PROC_DIR, INDEX_DIR, OUTPUTS_DIR]:
    d.mkdir(parents=True, exist_ok=True)

INDEX_PATH = INDEX_DIR / "index.faiss"
TEXTS_PATH = INDEX_DIR / "texts.jsonl"

EMB_MODEL_NAME = os.environ.get("EMB_MODEL", "sentence-transformers/all-MiniLM-L6-v2")