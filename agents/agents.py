from dotenv import load_dotenv
from langchain_groq import ChatGroq
from tools.tools import web_search , scrape_url
from langchain.agents import create_agent
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from tools.tools import human_approval

load_dotenv()

# Model Setup
model = ChatGroq(
    model = "openai/gpt-oss-20b",
    temperature=0
)

# Web Search Agent

def build_search_agent():
    return create_agent(
        model= model,
        tools=[web_search],
        middleware=[human_approval],
        system_prompt="""
        You are a web research search agent.

        Your task is to search the web using the web_search tool.

        Rules:
        - Always use the web_search tool to find information.
        - Do not write a research report.
        - Do not summarize the search results.
        - Return the tool results directly.
        - Preserve the Title, URL, and Content for every result.
        """
    )


# Scrape URL Agent

def build_reader_agent():
    return create_agent(
        model= model ,
        tools=[scrape_url]
    )


parser = StrOutputParser()

# Writer Chain

writer_prompt = ChatPromptTemplate.from_messages(
    [

        ("system", "You are an expert research writer. Write clear, structured and insightful reports."),
    
        ("human", """Write a detailed research report on the topic below.

Topic: {topic}

Research Gathered:
{research}

Structure the report as:
- Introduction
- Key Findings (minimum 3 well-explained points)
- Conclusion
- Sources (list all URLs found in the research)

Be detailed, factual and professional."""),
    ]
)

writer_chain = writer_prompt | model | parser

#critic chain

critic_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "You are a sharp and constructive research critic. Be honest and specific."),
        ("human", """Review the research report below and evaluate it strictly.

Report:
{report}

Respond in this exact format:

Score: X/10

Strengths:
- ...
- ...

Areas to Improve:
- ...
- ...

One line verdict:
..."""),
    ]
)

critic_chain = critic_prompt | model | parser