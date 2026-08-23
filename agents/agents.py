from dotenv import load_dotenv
from langchain_groq import ChatGroq
from tools.tools import web_search , scrape_url
from langchain.agents import create_agent
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate


load_dotenv()

# Model Setup
model = ChatGroq(
    model = "openai/gpt-oss-120b",
    temperature=0
)

# Web Search Agent

def build_search_agent():
    return create_agent(
        model= model,
        tools=[web_search]
    )


# Scrape URL Agent

def build_reader_agent():
    return create_agent(
        model= model ,
        tools=[scrape_url]
    )