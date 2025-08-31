Commands


```
pixi install

# 1) Download SQuAD via KaggleHub (reads .env if needed)
python script.py download

# 2) Prepare chunks (train) + keep dev QAs for eval
python script.py prepare --max-docs 200

# 3) Build FAISS index
python script.py index

# 4) RAG Q&A (set OPENAI_API_KEY in .env to use LLM answers)
python script.py rag-qa "What is the SQuAD dataset?"

# 5) Summarize
python script.py summarize "Overview of SQuAD dataset"

# 6) Semantic recommendations
python script.py recommend "Wikipedia articles about history"

# 7) Evaluate retrieval quickly
python script.py evaluate --top-k 5 --limit 50
```

OR

```
# 1) Download SQuAD
python app.py download download

# 2) Prepare chunks + dev QAs
python app.py prepare prepare --max-docs 200

# 3) Build FAISS index
python app.py index index

# 4) RAG QA (HF local QA)
python app.py qa rag-qa "Who are the Fighting Irish?"

# 5) Summarize relevant chunks
python app.py summarize summarize "Beyoncé Super Bowl performance"

# 6) Semantic recommendations
python app.py recommend recommend "American football"

# 7) Evaluate retrieval hit@5
python app.py evaluate evaluate --top-k 5 --limit 50
```
