import typer
from transformers import pipeline
from .indexer import retrieve
from .config import TEXTS_PATH
from .utils import read_jsonl

app = typer.Typer()
summarizer = pipeline("summarization", model="facebook/bart-large-cnn")

@app.command()
def summarize(topic: str, max_chunks: int = 6, max_chars: int = 3000):
    try:
        ctxs = retrieve(topic, top_k=max_chunks)
    except Exception:
        texts = [r["text"] for r in read_jsonl(TEXTS_PATH)]
        ctxs = texts[:max_chunks]

    if not ctxs:
        typer.echo("(No chunks found. Did you run prepare + index?)")
        raise typer.Exit(code=1)

    # Summarize per-chunk then merge, then final pass
    partials = []
    for c in ctxs:
        snippet = c[:1200]
        out = summarizer(snippet, max_length=120, min_length=40, do_sample=False)[0]["summary_text"]
        partials.append(out)

    merged = (" ".join(partials))[:max_chars]
    final = summarizer(f"Topic: {topic}\n\nContext: {merged}",
                       max_length=150, min_length=60, do_sample=False)[0]["summary_text"]
    typer.echo(final)
