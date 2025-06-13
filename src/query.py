from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA
from langchain_ollama import OllamaLLM


def query_rag(question):
    embedding = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vectorstore = Chroma(
        persist_directory="chroma",
        embedding_function=embedding
    )
    retriever = vectorstore.as_retriever()

    qa_chain = RetrievalQA.from_chain_type(
        llm=OllamaLLM(model="llama3"),
        retriever=retriever
    )

    return qa_chain.invoke(question)


if __name__ == "__main__":
    while True:
        q = input("Ask a question (or 'exit'): ")
        if q.lower() == 'exit':
            break
        answer = query_rag(q)
        print("Answer:", answer)
