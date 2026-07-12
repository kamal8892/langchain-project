import os
from pathlib import Path

from dotenv import load_dotenv
from langchain_unstructured import UnstructuredLoader
from langchain_pinecone import PineconeVectorStore, PineconeEmbeddings
from langchain_text_splitters import CharacterTextSplitter

load_dotenv()

if __name__ == "__main__":
    print("Ingesting...")
    file_path = Path(__file__).resolve().parent / "mediumblog1.txt"
    loader = UnstructuredLoader(file_path=str(file_path), chunking_strategy="basic", max_characters=1000000)
    document = loader.load()

    print("splitting...")
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    texts = text_splitter.split_documents(document)
    print(f"created {len(texts)} chunks")

    pinecone_api_key = os.environ.get("PINECONE_API_KEY")
    if not pinecone_api_key:
        raise EnvironmentError("Missing PINECONE_API_KEY in your .env file.")

    embeddings = PineconeEmbeddings(
        model="llama-text-embed-v2",
        pinecone_api_key=pinecone_api_key,
        top_k=5,
    )

    index_name = os.environ.get("INDEX_NAME")
    if not index_name:
        raise EnvironmentError("Missing INDEX_NAME. Set INDEX_NAME=rag in your .env file.")

    print(f"ingesting into Pinecone index: {index_name}")
    PineconeVectorStore.from_documents(
        texts, embeddings, index_name=index_name
    )
    print("finish")