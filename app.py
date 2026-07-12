import os
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

from dotenv import load_dotenv
load_dotenv()

from langchain.agents import create_agent
from langchain.tools import tool
from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI
from tavily_tools import search_query

llm = ChatOpenAI(model='gpt-4o-mini')
tools = [search_query]
agent = create_agent(model=llm, tools=tools)

def main():
    print("Hello welcome to LangChain Tavily Search")
    result = agent.invoke({"messages": [HumanMessage(content="search for the 3 job posting for python developer in pune can u given me link and content of the job posting") ]})
    print(result)

if __name__ == "__main__":
    main()
