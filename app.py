from dotenv import load_dotenv
import typer
from genai.download import app as download_app
from genai.prepare import app as prepare_app
from genai.indexer import app as index_app
from genai.qa import app as qa_app
from genai.summarize import app as summarize_app
from genai.recommend import app as recommend_app
from genai.evaluate import app as eval_app

cli = typer.Typer(help="GenAI mini project: RAG + summarization + recommendations + eval")
cli.add_typer(download_app, name="download")
cli.add_typer(prepare_app, name="prepare")
cli.add_typer(index_app, name="index")
cli.add_typer(qa_app, name="qa")
cli.add_typer(summarize_app, name="summarize")
cli.add_typer(recommend_app, name="recommend")
cli.add_typer(eval_app, name="evaluate")

if __name__ == "__main__":
    load_dotenv()
    cli()
