# import os
# from typing import Any

# from dotenv import load_dotenv
# from langchain_openai import ChatOpenAI, OpenAIEmbeddings
# from langchain_pinecone import PineconeVectorStore
# from pinecone import Pinecone

# load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), "..", ".env"))


# def run_llm(query: str) -> Any:
#     openai_api_key = os.getenv("OPENAI_API_KEY")
#     pinecone_api_key = os.getenv("PINECONE_API_KEY")
#     pinecone_index_name = os.getenv("PINECONE_INDEX_NAME")

#     if not openai_api_key:
#         raise EnvironmentError(
#             "Missing OPENAI_API_KEY. Set it in your environment or .env file before running the script."
#         )
#     if not pinecone_api_key:
#         raise EnvironmentError(
#             "Missing PINECONE_API_KEY. Set it in your environment or .env file before running the script."
#         )
#     if not pinecone_index_name:
#         raise EnvironmentError(
#             "Missing PINECONE_INDEX_NAME. Set it in your environment or .env file before running the script."
#         )

#     pc = Pinecone(api_key=pinecone_api_key)

#     embeddings = OpenAIEmbeddings(api_key=openai_api_key)
#     docsearch = PineconeVectorStore.from_existing_index(
#         index_name=pinecone_index_name, embedding=embeddings
#     )
#     chat = ChatOpenAI(api_key=openai_api_key, verbose=True, temperature=0)

#     docs = docsearch.similarity_search(query=query, k=4)
#     context = "\n\n".join(doc.page_content for doc in docs)
#     prompt = (
#         "Answer the question using only the provided context.\n\n"
#         f"Context:\n{context}\n\nQuestion:\n{query}"
#     )

#     response = chat.invoke(prompt)
#     return response.content if hasattr(response, "content") else response


# if __name__ == "__main__":
#     print(run_llm(query="What is LangChain?"))







import os
from typing import Any, Dict

from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain.messages import ToolMessage
from langchain.tools import tool
from langchain_pinecone import PineconeVectorStore, PineconeEmbeddings
from langchain_groq import ChatGroq

load_dotenv()

pinecone_api_key = os.getenv("PINECONE_API_KEY")
pinecone_index_name = os.getenv("PINECONE_INDEX_NAME", "langchain-doc-index")

if not pinecone_api_key:
    raise EnvironmentError("Missing PINECONE_API_KEY in your .env file.")

embeddings = PineconeEmbeddings(
    model="llama-text-embed-v2",
    pinecone_api_key=pinecone_api_key,
)

vectorstore = PineconeVectorStore(
    index_name=pinecone_index_name, embedding=embeddings
)
model = ChatGroq(model="llama-3.1-8b-instant", temperature=0)


@tool(response_format="content_and_artifact")
def retrieve_context(query: str):
    """Retrieve relevant documentation to help answer user queries about LangChain."""
    retrieved_docs = vectorstore.as_retriever().invoke(query, k=4)
    print(f"Retrieved {len(retrieved_docs)} document(s) for query: {query}")
    if not retrieved_docs:
        print("No documents were retrieved from Pinecone.")

    serialized = "\n\n".join(
        (f"Source: {doc.metadata.get('source', 'Unknown')}\n\nContent: {doc.page_content}")
        for doc in retrieved_docs
    )

    return serialized, retrieved_docs


def run_llm(query: str) -> Dict[str, Any]:
    """
    Run the RAG pipeline to answer a query using retrieved documentation.
    
    Args:
        query: The user's question
        
    Returns:
        Dictionary containing:
            - answer: The generated answer
            - context: List of retrieved documents
    """
    system_prompt = (
        "You are a helpful AI assistant that answers questions about LangChain documentation. "
        "You have access to a tool that retrieves relevant documentation. "
        "Use the tool to find relevant information before answering questions. "
        "Always cite the sources you use in your answers. "
        "If you cannot find the answer in the retrieved documentation, say so."
    )
    
    agent = create_agent(model, tools=[retrieve_context], system_prompt=system_prompt)
    
  
    messages = [{"role": "user", "content": query}]
    
   
    response = agent.invoke({"messages": messages})
    
  
    answer = response["messages"][-1].content
    
    
    context_docs = []
    for message in response["messages"]:
        
        if isinstance(message, ToolMessage) and hasattr(message, "artifact"):
            
            if isinstance(message.artifact, list):
                context_docs.extend(message.artifact)
    
    return {
        "answer": answer,
        "context": context_docs
    }

if __name__ == '__main__':
    result = run_llm(query="what are deep agents?")
    print(result)