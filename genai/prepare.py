from typing import List, Tuple
from pathlib import Path
import json, typer
from .config import RAW_DIR, PROC_DIR
from .utils import clean_text, write_jsonl

app = typer.Typer()

def load_squad_json(path: Path) -> Tuple[List[str], List[dict]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    contexts, qas = [], []
    for art in data.get("data", []):
        for para in art.get("paragraphs", []):
            context = clean_text(para.get("context", ""))
            if not context: continue
            contexts.append(context)
            for qa in para.get("qas", []):
                if qa.get("is_impossible"): continue
                q = clean_text(qa.get("question", ""))
                ans = clean_text(qa.get("answers", [{}])[0].get("text", "")) if qa.get("answers") else ""
                qas.append({"question": q, "answer": ans, "context": context})
    return contexts, qas

def chunk_words(text: str, chunk_size: int = 180, overlap: int = 30) -> List[str]:
    words = text.split()
    chunks, i = [], 0
    step = max(1, chunk_size - overlap)
    while i < len(words):
        chunk = " ".join(words[i:i+chunk_size])
        if chunk: chunks.append(chunk)
        i += step
    return chunks

@app.command()
def prepare(max_docs: int = 200, chunk_size: int = 180, overlap: int = 30):
    train_json = RAW_DIR / "train-v1.1.json"
    dev_json = RAW_DIR / "dev-v1.1.json"
    if not train_json.exists() or not dev_json.exists():
        typer.echo("Missing SQuAD files. Run: python app.py download")
        raise typer.Exit(code=1)

    train_contexts, _ = load_squad_json(train_json)
    _, dev_qas = load_squad_json(dev_json)

    train_contexts = train_contexts[:max_docs]

    chunk_rows = []
    for ctx in train_contexts:
        for ch in chunk_words(ctx, chunk_size, overlap):
            chunk_rows.append({"text": ch})

    write_jsonl(PROC_DIR / "chunks.jsonl", chunk_rows)
    write_jsonl(PROC_DIR / "qas.jsonl", dev_qas)
    typer.echo(f"Wrote {len(chunk_rows)} chunks and {len(dev_qas)} dev QAs.")
