import os
from pathlib import Path

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_pinecone import PineconeVectorStore, PineconeEmbeddings
from langchain_text_splitters import CharacterTextSplitter
from langchain_unstructured import UnstructuredLoader
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

load_dotenv()


def get_env(name: str, default: str | None = None) -> str:
    value = os.environ.get(name, default)
    if not value:
        raise EnvironmentError(f"Missing environment variable {name}")
    return value


def load_document_chunks() -> list:
    file_path = Path(__file__).resolve().parent / "mediumblog1.txt"
    loader = UnstructuredLoader(
        file_path=str(file_path), chunking_strategy="basic", max_characters=1000000
    )
    document = loader.load()
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    chunks = text_splitter.split_documents(document)
    print(f"Loaded {len(chunks)} chunks from {file_path.name}")
    return chunks


def build_vector_store(chunks: list) -> PineconeVectorStore:
    pinecone_api_key = get_env("PINECONE_API_KEY")
    embedding_model = os.environ.get("PINECONE_EMBEDDING_MODEL", "llama-text-embed-v2")
    index_name = get_env("PINECONE_INDEX_NAME")

    embeddings = PineconeEmbeddings(
        model=embedding_model,
        pinecone_api_key=pinecone_api_key,
    )
    vector_store = PineconeVectorStore.from_documents(
        chunks, embeddings, index_name=index_name
    )
    print(f"Connected to Pinecone index: {index_name}")
    return vector_store


def format_docs(docs) -> str:
    return "\n\n".join(doc.page_content for doc in docs)


def create_rag_chain(vector_store: PineconeVectorStore):
    llm_model = os.environ.get("OPENAI_CHAT_MODEL", "gpt-4o-mini")
    llm = ChatOpenAI(model=llm_model, temperature=0)
    retriever = vector_store.as_retriever(search_kwargs={"k": 4})

    prompt = ChatPromptTemplate.from_template(
        "Answer the question based only on the following context:\n\n"
        "{context}\n\nQuestion: {question}"
    )

    chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )
    return chain


def main() -> None:
    print("Starting RAG pipeline...")
    get_env("PINECONE_API_KEY")
    get_env("PINECONE_INDEX_NAME")

    chunks = load_document_chunks()
    vector_store = build_vector_store(chunks)
    qa_chain = create_rag_chain(vector_store)

    question = input("Enter your question: ").strip()
    if not question:
        print("No question provided. Exiting.")
        return

    answer = qa_chain.invoke(question)
    print("\n=== RAG Answer ===")
    print(answer)


if __name__ == "__main__":
    main()