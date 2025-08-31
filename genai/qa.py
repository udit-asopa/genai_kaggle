import typer
from transformers import pipeline
from .indexer import retrieve

app = typer.Typer()
qa_pipeline = pipeline("question-answering", model="deepset/roberta-base-squad2")

@app.command("rag-qa")
def rag_qa(question: str, top_k: int = 4):
    ctxs = retrieve(question, top_k=top_k)
    if not ctxs:
        typer.echo("(No context retrieved)")
        raise typer.Exit(code=1)

    # QA per chunk; pick best answer by score (prevents long-context truncation)
    candidates = []
    for c in ctxs:
        try:
            out = qa_pipeline(question=question, context=c)
            candidates.append((out.get("score", 0.0), out.get("answer", ""), c))
        except Exception as e:
            candidates.append((0.0, f"(error: {e})", c))

    best_score, ans, _ = max(candidates, key=lambda x: x[0]) if candidates else (0.0, "(no answer)", "")

    typer.echo("\n=== Retrieved Contexts ===")
    for i, c in enumerate(ctxs, 1):
        typer.echo(f"[{i}] {c[:220]}...")
    typer.echo("\n=== Answer ===\n" + ans)
