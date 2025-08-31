import numpy as np, faiss, typer
import json
from sentence_transformers import SentenceTransformer
from .config import PROC_DIR, INDEX_PATH, TEXTS_PATH, EMB_MODEL_NAME
from .utils import read_jsonl

app = typer.Typer()

def ensure_embeddings(model_name: str = EMB_MODEL_NAME) -> SentenceTransformer:
    return SentenceTransformer(model_name)

def load_texts() -> list[str]:
    return [r["text"] for r in read_jsonl(TEXTS_PATH)]

def retrieve(query: str, top_k: int = 4, emb_model: str = EMB_MODEL_NAME) -> list[str]:
    model = ensure_embeddings(emb_model)
    qv = model.encode([query], normalize_embeddings=True)
    index = faiss.read_index(str(INDEX_PATH))
    D, I = index.search(np.array(qv, dtype="float32"), top_k)
    texts = load_texts()
    return [texts[i] for i in I[0]]

@app.command()
def index(emb_model: str = EMB_MODEL_NAME):
    chunks_path = PROC_DIR / "chunks.jsonl"
    if not chunks_path.exists():
        typer.echo("No chunks found. Run: python app.py prepare")
        raise typer.Exit(code=1)

    rows = read_jsonl(chunks_path)
    texts = [r["text"] for r in rows]
    model = ensure_embeddings(emb_model)
    embs = model.encode(texts, normalize_embeddings=True, batch_size=64, show_progress_bar=True).astype("float32")
    dim = embs.shape[1]
    idx = faiss.IndexFlatIP(dim)
    idx.add(embs)
    faiss.write_index(idx, str(INDEX_PATH))
    with TEXTS_PATH.open("w", encoding="utf-8") as f:
        for t in texts:
            f.write(json.dumps({"text": t}, ensure_ascii=False) + "\n")
    typer.echo(f"Built FAISS index with {len(texts)} chunks.")
