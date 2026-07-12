import os 
import json
from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain.tools import tool
from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI
from tavily import TavilyClient

load_dotenv()

# Initialize Tavily client
tavily_api_key = os.getenv("TAVILY_API_KEY")
tavily_client = TavilyClient(api_key=tavily_api_key)

@tool
def search_query(query: str) -> str:
    """
    Tool that searches over the internet using Tavily Search API
    Args:
        query: The query to search
    Returns:
        The search results formatted as a string
    """
    print(f"Searching for: {query}")
    try:
        # Perform Tavily search
        response = tavily_client.search(
            query=query,
            max_results=5,
            include_answer=True
        )
        
        # Format the results
        results = []
        
        # Add the answer if available
        if response.get("answer"):
            results.append(f"Answer: {response['answer']}\n")
        
        # Add search results
        if response.get("results"):
            results.append("Search Results:")
            for idx, result in enumerate(response["results"], 1):
                results.append(f"\n{idx}. {result.get('title', 'No title')}")
                results.append(f"   URL: {result.get('url', 'No URL')}")
                results.append(f"   Content: {result.get('content', 'No content')[:200]}...")
        
        return "\n".join(results) if results else "No results found"
    
    except Exception as e:
        return f"Error during search: {str(e)}"




llm = ChatOpenAI()
tools = [search_query]
agent = create_agent(
    model=llm,
    tools=tools
)

def main():
    print("Hello welcome to LangChain Tavily Search")
    result = agent.invoke({"messages": [HumanMessage(content="What is the weather in pune")]})
    print(result)

if __name__ == "__main__":
    main()
