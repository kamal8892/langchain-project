import os

from dotenv import load_dotenv
from langchain_pinecone import PineconeVectorStore, PineconeEmbeddings

load_dotenv()

embeddings = PineconeEmbeddings(
    model="llama-text-embed-v2",
    pinecone_api_key=os.environ.get("PINECONE_API_KEY"),
)

vectorstore = PineconeVectorStore(
    index_name=os.environ.get("INDEX_NAME"),
    embedding=embeddings,
)

retriever = vectorstore.as_retriever()
