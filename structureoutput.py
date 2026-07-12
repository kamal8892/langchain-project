from dotenv import load_dotenv
load_dotenv()

from pydantic import BaseModel, Field
from langchain.agents import create_agent
from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI
from langchain.tools import tool 
# Create LLM
llm = ChatOpenAI(model="gpt-4o-mini")

# Create agent with empty tools list
agent = create_agent(
    model=llm,
    tools=[]
)

result = agent.invoke({
    "messages": [HumanMessage(content="Extract contact info from: John Doe, john@example.com, (555) 123-4567")]
})

print("Result:")
print(result)


# Data model for structured output

from dataclasses import dataclass
from langchain.agents import create_agent


@dataclass
class ContactInfo:
    """Contact information for a person."""
    name: str 
    email: str 
    phone: str 

agent = create_agent(
    model="gpt-5.5",
    tools=[],
    response_format=ContactInfo  
)

result = agent.invoke({
    "messages": [{"role": "user", "content": "Extract contact info from: John Doe, john@example.com, (555) 123-4567"}]
})

result["structured_response"]



# TypedDict for structured output

from typing_extensions import TypedDict
from langchain.agents import create_agent


class ContactInfo(TypedDict):
    """Contact information for a person."""
    name: str # The name of the person
    email: str # The email address of the person
    phone: str # The phone number of the person

agent = create_agent(
    model="gpt-4o-mini",
    tools=[],
    response_format=ContactInfo  # Auto-selects ProviderStrategy
)

result = agent.invoke({
    "messages": [{"role": "user", "content": "Extract contact info from: John Doe, john@example.com, (555) 123-4567"}]
})

result["structured_response"]


# JSON Schema for structured output

from langchain.agents import create_agent
from langchain.schema import ProviderStrategy

contact_info_schema = {
    "type": "object",
    "description": "Contact information for a person.",
    "properties": {
        "name": {"type": "string", "description": "The name of the person"},
        "email": {"type": "string", "description": "The email address of the person"},
        "phone": {"type": "string", "description": "The phone number of the person"}
    },
    "required": ["name", "email", "phone"]
}

agent = create_agent(
    model="gpt-4o-mini",
    tools=[],
    response_format=ProviderStrategy(contact_info_schema)
)

result = agent.invoke({
    "messages": [{"role": "user", "content": "Extract contact info from: John Doe, john@example.com, (555) 123-4567"}]
})

result["structured_response"]


