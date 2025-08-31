from pathlib import Path
import typer, kagglehub
from dotenv import load_dotenv
from .config import RAW_DIR

app = typer.Typer()

@app.command()
def download():
    """Download SQuAD via KaggleHub; copy train/dev JSONs into data/raw."""
    load_dotenv()
    path = kagglehub.dataset_download("stanfordu/stanford-question-answering-dataset")
    typer.echo(f"Kaggle dataset path: {path}")
    src = Path(path)
    for fn in ["train-v1.1.json", "dev-v1.1.json"]:
        src_file = src / fn
        dst_file = RAW_DIR / fn
        if src_file.exists():
            dst_file.write_bytes(src_file.read_bytes())
            typer.echo(f"Copied {fn} -> {dst_file}")
        else:
            typer.echo(f"WARNING: {fn} not found at {src_file}")
