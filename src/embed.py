from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from ingest import load_and_split_pdfs
import os


def build_chroma_vectorstore():
    docs = load_and_split_pdfs("data")
    print(f"✅ Loaded and split {len(docs)} chunks.")

    embedding = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vectorstore = Chroma.from_documents(
        documents=docs,
        embedding=embedding,
        persist_directory="chroma"
    )

    print("Chroma vector store saved.")


if __name__ == "__main__":
    build_chroma_vectorstore()
