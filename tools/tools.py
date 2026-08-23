from dotenv import load_dotenv
import os
import requests
from langchain.tools import tool
from rich import print
from tavily import TavilyClient

load_dotenv()

tavily = TavilyClient(
    api_key=os.getenv("TAVILY_API_KEY")
)

@tool
def web_search(query : str) -> str:
    """Search the web for recent and reliable information on a topic . Returns Titles , URLs and snippets."""
    results = tavily.search(
        query= query,
        max_results= 5
    )
    out = []
    for r in results['results']:
        out.append(
            f"Title: {r['title']} \nURL: {r['url']} \nContent: {r['content'][:400]}\n"
        )

    return '\n-----\n'.join(out)
