import typer
import subprocess

app = typer.Typer()

@app.command()
def prepare(dataset: str = "squad", max_docs: int = 25):
    """Prepare dataset (chunks + QAs)."""
    subprocess.run(["python", "01_prepare_data.py", "--dataset", dataset, "--max-docs", str(max_docs)])

@app.command()
def build_index():
    """Build FAISS index."""
    subprocess.run(["python", "02_build_index.py"])

@app.command()
def rag_qa(question: str):
    """Run Retrieval-Augmented QA."""
    subprocess.run(["python", "03_rag_qa.py", "--question", question])

@app.command()
def summarize(topic: str):
    """Summarize retrieved docs."""
    subprocess.run(["python", "04_summarize.py", "--topic", topic])

@app.command()
def recommend(query: str):
    """Semantic recommendations (top-K)."""
    subprocess.run(["python", "05_recommend.py", "--query", query])

@app.command()
def eval():
    """Evaluate retrieval and QA performance."""
    subprocess.run(["python", "06_eval.py"])

if __name__ == "__main__":
    app()
