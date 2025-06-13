# RAG PDF Project (Chroma Version)

A simple Retrieval-Augmented Generation (RAG) pipeline using:

- 🧠 `sentence-transformers` for embeddings
- 🗃️ `Chroma` for persistent vector storage
- 📄 `LangChain` for PDF parsing and orchestration

## 🔧 Setup

```bash
python -m venv .venv
source .venv/bin/activate  # or .venv\Scripts\activate on Windows
pip install -r requirements.txt

Make sure you have a PDF file in the data folder

run the embed.py script to split and load the vector store
run query.py to ask questions about the pdf file 