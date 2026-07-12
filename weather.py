import os 
from dotenv import load_dotenv
load_dotenv()

from langchain.agents import create_agent
from langchain.tools import tool
from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI
from tavily import TavilyClient 

tavily = TavilyClient()

@tool
def search(query:str)->str:
    """
    Tool that search over internet
    Args: 
        query: The query to search for
    Returns:
        The search result 
    """
    print(f"Searching for {query}")
    return tavily.search(query=query)

llm = ChatOpenAI(model='gpt-5')
tools = [search]
agents = create_agent(model=llm,tools=tool)

if __name__ == "__main__":
    search()

