from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os


def load_and_split_pdfs(data_dir):
    abs_path = os.path.abspath(data_dir)
    print(f"🔍 Looking for PDFs in: {abs_path}")

    if not os.path.exists(abs_path):
        raise FileNotFoundError(f"❌ Folder does not exist: {abs_path}")

    documents = []
    for file in os.listdir(data_dir):
        if file.endswith(".pdf"):
            loader = PyPDFLoader(os.path.join(data_dir, file))
            documents.extend(loader.load())

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000, chunk_overlap=200)
    return splitter.split_documents(documents)


if __name__ == "__main__":
    docs = load_and_split_pdfs("data")
    print(f"Loaded and split {len(docs)} chunks.")
