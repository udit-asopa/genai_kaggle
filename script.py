from __future__ import annotations
import os, json, typer
from pathlib import Path
from typing import List, Tuple
from dotenv import load_dotenv

import numpy as np
import faiss
import pandas as pd

import kagglehub
from sentence_transformers import SentenceTransformer
from openai import OpenAI

from transformers import pipeline

_qa_pipeline = pipeline("question-answering", model="deepset/roberta-base-squad2")
_summarizer = pipeline("summarization", model="facebook/bart-large-cnn") #sshleifer/distilbart-cnn-12-6
#switch to "facebook/bart-large-cnn" for higher quality but slower responses

app = typer.Typer(help="GenAI mini project: RAG + summarization + recommendations + eval")


# --- Paths ---
ROOT = Path(__file__).parent
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

# --- Utilities ---
def clean_text(t: str) -> str:
    return " ".join((t or "").split())

def load_openai() -> OpenAI | None:
    key = os.getenv("OPENAI_API_KEY")
    return OpenAI(api_key=key) if key else None

def ensure_embeddings(model_name: str = EMB_MODEL_NAME) -> SentenceTransformer:
    return SentenceTransformer(model_name)

def write_jsonl(path: Path, rows: List[dict]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

def read_jsonl(path: Path) -> List[dict]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            rows.append(json.loads(line))
    return rows

# --- 1) Kaggle download ---
@app.command()
def download():
    """
    Download SQuAD dataset via KaggleHub. Reads KAGGLE_USERNAME/KAGGLE_KEY from .env if needed.
    """
    load_dotenv()
    path = kagglehub.dataset_download("stanfordu/stanford-question-answering-dataset")
    typer.echo(f"Kaggle dataset path: {path}")
    # Copy JSONs to data/raw for clarity
    src = Path(path)
    for fn in ["train-v1.1.json", "dev-v1.1.json"]:
        src_file = src / fn
        dst_file = RAW_DIR / fn
        if src_file.exists():
            dst_file.write_bytes(src_file.read_bytes())
            typer.echo(f"Copied {fn} -> {dst_file}")
        else:
            typer.echo(f"WARNING: {fn} not found at {src_file}")

# --- 2) Prepare (chunk contexts + QAs) ---
def load_squad_json(path: Path) -> Tuple[List[str], List[dict]]:
    """
    Return (contexts, qas) from SQuAD v1.1 JSON.
    Each QA row: {'question', 'answer', 'context'}
    """
    data = json.loads(path.read_text(encoding="utf-8"))
    contexts, qas = [], []
    for art in data.get("data", []):
        for para in art.get("paragraphs", []):
            context = clean_text(para.get("context", ""))
            if not context:
                continue
            contexts.append(context)
            for qa in para.get("qas", []):
                if qa.get("is_impossible"):
                    continue
                question = clean_text(qa.get("question", ""))
                answers = qa.get("answers", [])
                ans_text = clean_text(answers[0]["text"]) if answers else ""
                qas.append({"question": question, "answer": ans_text, "context": context})
    return contexts, qas

def chunk_words(text: str, chunk_size: int = 180, overlap: int = 30) -> List[str]:
    # word-based chunking (fast, simple, no regex)
    words = text.split()
    chunks = []
    i = 0
    step = max(1, chunk_size - overlap)
    while i < len(words):
        chunk = " ".join(words[i:i + chunk_size])
        if chunk:
            chunks.append(chunk)
        i += step
    return chunks

@app.command()
def prepare(max_docs: int = 200, chunk_size: int = 180, overlap: int = 30):
    """
    Build processed chunks and QAs from SQuAD.
    - Uses train-v1.1.json for contexts
    - Uses dev-v1.1.json for evaluation QAs (smaller, good for quick eval)
    """
    train_json = RAW_DIR / "train-v1.1.json"
    dev_json = RAW_DIR / "dev-v1.1.json"
    if not train_json.exists() or not dev_json.exists():
        typer.echo("Missing SQuAD files. Run: python app.py download")
        raise typer.Exit(code=1)

    train_contexts, _ = load_squad_json(train_json)
    dev_contexts, dev_qas = load_squad_json(dev_json)

    # Limit docs for speed
    train_contexts = train_contexts[:max_docs]

    # Chunk train contexts
    chunk_rows = []
    for ctx in train_contexts:
        for ch in chunk_words(ctx, chunk_size, overlap):
            chunk_rows.append({"text": ch})
    write_jsonl(PROC_DIR / "chunks.jsonl", chunk_rows)

    # Save dev QAs for eval
    write_jsonl(PROC_DIR / "qas.jsonl", dev_qas)
    typer.echo(f"Wrote {len(chunk_rows)} chunks and {len(dev_qas)} dev QAs.")

# --- 3) Build FAISS index ---
@app.command()
def index(emb_model: str = EMB_MODEL_NAME):
    """
    Create embeddings for chunks and build FAISS index.
    """
    chunks_path = PROC_DIR / "chunks.jsonl"
    if not chunks_path.exists():
        typer.echo("No chunks found. Run: python app.py prepare")
        raise typer.Exit(code=1)

    rows = read_jsonl(chunks_path)
    texts = [r["text"] for r in rows]
    model = ensure_embeddings(emb_model)
    embs = model.encode(texts, normalize_embeddings=True, batch_size=64, show_progress_bar=True)
    embs = np.array(embs, dtype="float32")
    dim = embs.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(embs)
    faiss.write_index(index, str(INDEX_PATH))
    # Save texts alongside
    with TEXTS_PATH.open("w", encoding="utf-8") as f:
        for t in texts:
            f.write(json.dumps({"text": t}, ensure_ascii=False) + "\n")
    typer.echo(f"Built FAISS index with {len(texts)} chunks.")

def _load_texts() -> List[str]:
    return [r["text"] for r in read_jsonl(TEXTS_PATH)]

def _retrieve(query: str, top_k: int = 4, emb_model: str = EMB_MODEL_NAME) -> List[str]:
    model = ensure_embeddings(emb_model)
    qv = model.encode([query], normalize_embeddings=True)
    index = faiss.read_index(str(INDEX_PATH))
    D, I = index.search(np.array(qv, dtype="float32"), top_k)
    texts = _load_texts()
    return [texts[i] for i in I[0]]

def _llm_answer(prompt: str, context: str = "", temperature: float = 0.2) -> str:
    """
    Local QA answerer using Hugging Face pipeline.
    - prompt: the question
    - context: retrieved chunks concatenated
    """
    if not context.strip():
        return "(No context provided)"
    result = _qa_pipeline(question=prompt, context=context)
    return result["answer"]

# --- 4) RAG Q&A ---
@app.command("rag-qa")
def rag_qa(question: str, top_k: int = 4):
    """
    Retrieval-Augmented Question Answering using Hugging Face QA pipeline.
    """
    # Retrieve top-k chunks
    ctxs = _retrieve(question, top_k=top_k)
    context = " ".join(ctxs)

    # Run local QA model
    if not context.strip():
        ans = "(No context retrieved)"
    else:
        result = _qa_pipeline(question=question, context=context)
        ans = result["answer"]

    # Display
    typer.echo("\n=== Retrieved Contexts ===")
    for i, c in enumerate(ctxs, 1):
        typer.echo(f"[{i}] {c[:220]}...")
    typer.echo("\n=== Answer ===\n" + ans)

# --- 5) Summarize ---
@app.command()
def summarize(topic: str, max_chunks: int = 6, max_chars: int = 3000):
    """
    Summarize the most relevant chunks for a topic using a local Hugging Face summarizer.
    - topic: what you want summarized (used to retrieve relevant chunks)
    - max_chunks: how many retrieved chunks to include
    - max_chars: safety cap to avoid very long inputs
    """
    # 1) get candidate texts
    try:
        # prefer retrieval for relevance
        ctxs = _retrieve(topic, top_k=max_chunks)
    except Exception:
        # fallback: read straight from index texts if retrieval isn't available
        texts = _load_texts()
        ctxs = texts[:max_chunks]

    if not ctxs:
        typer.echo("(No chunks found. Did you run prepare + index?)")
        raise typer.Exit(code=1)

    # 2) keep input size reasonable (HF models have input limits)
    #    join small summaries of each chunk to maintain coverage
    per_chunk_summaries = []
    for c in ctxs:
        snippet = c[:1200]  # prevent single-chunk overflow
        out = _summarizer(snippet, max_length=120, min_length=40, do_sample=False)[0]["summary_text"]
        per_chunk_summaries.append(out)

    merged = (" ".join(per_chunk_summaries))[:max_chars]

    # 3) final pass to produce a tight overview
    final = _summarizer(
        f"Topic: {topic}\n\nContext: {merged}",
        max_length=150,
        min_length=60,
        do_sample=False
    )[0]["summary_text"]

    typer.echo(final)

# --- 6) Recommendations (semantic NN) ---
@app.command()
def recommend(query: str, top_k: int = 5):
    """
    Return nearest-neighbor chunks by embeddings (semantic recommendation).
    """
    ctxs = _retrieve(query, top_k=top_k)
    for i, txt in enumerate(ctxs, 1):
        typer.echo(f"[{i}] {txt[:200]}...")

# --- 7) Eval (retrieval hit@K using dev set QAs) ---
@app.command()
def evaluate(top_k: int = 5, limit: int = 50):
    """
    Simple retrieval evaluation: for each dev QA question, check if any retrieved chunk
    contains a snippet from its gold context (coarse proxy hit@K).
    """
    qas_path = PROC_DIR / "qas.jsonl"
    if not qas_path.exists():
        typer.echo("No dev QAs found. Run: python app.py prepare")
        raise typer.Exit(code=1)

    qas = [r for r in read_jsonl(qas_path) if r.get("question")]
    qas = qas[:limit]
    hits = 0
    total = len(qas)
    for qa in qas:
        q = qa["question"]
        gold_ctx = qa["context"]
        retrieved = _retrieve(q, top_k=top_k)
        # proxy: if the first 120 chars of gold context appear in any retrieved chunk
        gold_snip = gold_ctx[:120]
        hit = any(gold_snip in r for r in retrieved)
        hits += int(hit)

    precision_at_k = hits / total if total else 0.0
    metrics = {"retrieval_precision_at_{}".format(top_k): precision_at_k, "samples": total}
    (OUTPUTS_DIR / "eval_metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    typer.echo(json.dumps(metrics, indent=2))

if __name__ == "__main__":
    load_dotenv()
    app()
