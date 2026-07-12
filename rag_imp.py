import os
from pathlib import Path
from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI
from langchain_pinecone import PineconeVectorStore, PineconeEmbeddings
from langchain_unstructured import UnstructuredLoader
from langchain_text_splitters import CharacterTextSplitter

load_dotenv()

print("Initializing components...")

pinecone_api_key = os.environ.get("PINECONE_API_KEY")
if not pinecone_api_key:
    raise EnvironmentError("Missing PINECONE_API_KEY in your .env file.")

pinecone_index_name = os.environ.get("PINECONE_INDEX_NAME")
if not pinecone_index_name:
    raise EnvironmentError("Missing PINECONE_INDEX_NAME in your .env file.")

embeddings = PineconeEmbeddings(
    model="llama-text-embed-v2",
    pinecone_api_key=pinecone_api_key,
)
llm = ChatOpenAI()

vectorstore = PineconeVectorStore(
    index_name=pinecone_index_name, embedding=embeddings
)

retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

prompt_template = ChatPromptTemplate.from_template(
    """Answer the question based only on the following context:

{context}

Question: {question}

Provide a detailed answer:"""
)

file_path = Path(__file__).resolve().parent / "mediumblog1.txt"
loader = UnstructuredLoader(file_path=str(file_path), chunking_strategy="basic", max_characters=1000000)
document = loader.load()

print("splitting...")
text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
texts = text_splitter.split_documents(document)
print(f"created {len(texts)} chunks")

print(f"ingesting into Pinecone index: {pinecone_index_name}")
PineconeVectorStore.from_documents(
    texts, embeddings, index_name=pinecone_index_name
)
print("finish")


def format_docs(docs):
    """Format retrieved documents into a single string."""
    return "\n\n".join(doc.page_content for doc in docs)


def retrieval_chain_without_lcel(query: str):
    """
    Simple retrieval chain without LCEL.
    Manually retrieves documents, formats them, and generates a response.

    Limitations:
    - Manual step-by-step execution
    - No built-in streaming support
    - No async support without additional code
    - Harder to compose with other chains
    - More verbose and error-prone
    """
    # Step 1: Retrieve relevant documents
    docs = retriever.invoke(query)

    # Step 2: Format documents into context string
    context = format_docs(docs)

    # Step 3: Format the prompt with context and question
    messages = prompt_template.format_messages(context=context, question=query)

    # Step 4: Invoke LLM with the formatted messages
    response = llm.invoke(messages)

    # Step 5: Return the content
    return response.content


if __name__ == "__main__":
    print("Retrieving...")

    # Query
    query = "what is Pinecone in machine learning?"

    # ========================================================================
    # Option 0: Raw invocation without RAG
    # ========================================================================
    print("\n" + "=" * 70)
    print("IMPLEMENTATION 0: Raw LLM Invocation (No RAG)")
    print("=" * 70)
    result_raw = llm.invoke([HumanMessage(content=query)])
    print("\nAnswer:")
    print(result_raw.content)

    # ========================================================================
    # Option 1: Use implementation WITHOUT LCEL
    # ========================================================================
    print("\n" + "=" * 70)
    print("IMPLEMENTATION 1: Without LCEL")
    print("=" * 70)
    result_without_lcel = retrieval_chain_without_lcel(query)
    print("\nAnswer:")
    print(result_without_lcel)


 # ========================================================================
    # Option 0: Raw invocation without RAG
    # ========================================================================
    print("\n" + "=" * 70)
    print("IMPLEMENTATION 0: Raw LLM Invocation (RAG)")