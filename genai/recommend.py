import typer
from .indexer import retrieve

app = typer.Typer()

@app.command()
def recommend(query: str, top_k: int = 5):
    ctxs = retrieve(query, top_k=top_k)
    for i, txt in enumerate(ctxs, 1):
        typer.echo(f"[{i}] {txt[:200]}...")
