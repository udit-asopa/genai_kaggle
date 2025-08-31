import json, typer
from .config import PROC_DIR, OUTPUTS_DIR
from .utils import read_jsonl
from .indexer import retrieve

app = typer.Typer()

@app.command()
def evaluate(top_k: int = 5, limit: int = 50):
    qas_path = PROC_DIR / "qas.jsonl"
    if not qas_path.exists():
        typer.echo("No dev QAs found. Run: python app.py prepare")
        raise typer.Exit(code=1)

    qas = [r for r in read_jsonl(qas_path) if r.get("question")][:limit]
    hits, total = 0, len(qas)
    for qa in qas:
        q = qa["question"]
        gold_ctx = qa["context"]
        retrieved = retrieve(q, top_k=top_k)
        gold_snip = gold_ctx[:120]
        hit = any(gold_snip in r for r in retrieved)
        hits += int(hit)

    precision_at_k = hits / total if total else 0.0
    metrics = {f"retrieval_precision_at_{top_k}": precision_at_k, "samples": total}
    (OUTPUTS_DIR / "eval_metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    typer.echo(json.dumps(metrics, indent=2))
