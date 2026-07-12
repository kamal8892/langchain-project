from typing import List, Optional, Any
from pydantic import BaseModel, Field
from dotenv import load_dotenv

load_dotenv()

from langchain.agents import create_agent
from langchain.tools import tool
from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI
from langchain_tavily import TavilySearch   




class Source(BaseModel):
    """
    Represents a source of information for a search result.
    """
    title: str = Field(..., description="The title of the source")
    url: str = Field(..., description="The URL of the source")
    content: Optional[str] = Field(None, description="The content of the source")

class AgentResponse(BaseModel):
    """
    Represents the response from the agent after invoking a search query.
    """
    answer: Optional[str] = Field(None, description="The answer provided by the agent")
    sources: List[Source] = Field(default_factory=list, description="List of sources related to the search query")


def main():
    print("Hello welcome to LangChain Tavily Search")
    result = agent.invoke({"messages": [HumanMessage(content="search for the 3 job posting for python developer in pune can u given me link and content of the job posting") ]})
    print(result)


llm = ChatOpenAI(model='gpt-4o-mini')
tools = [TavilySearch()]
agent = create_agent(model=llm, tools=tools, response_format=AgentResponse)

if __name__ == "__main__":
    main()