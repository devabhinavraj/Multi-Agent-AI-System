from dotenv import load_dotenv
import os
import requests
from langchain.tools import tool
from rich import print
from tavily import TavilyClient
from bs4 import BeautifulSoup
from langchain.agents.middleware import wrap_tool_call
from langchain_core.messages import ToolMessage

load_dotenv()

def _default_approval_handler(tool_name: str) -> bool:
    confirm = input(f"Agent wants to call '{tool_name}'. Approve? (yes/no): ")
    return confirm.strip().lower() != "no"


approval_handler = _default_approval_handler


@wrap_tool_call
def human_approval(request, handler):
    '''Ask for human approval before every tool call.'''
    try:
        tool_name = request.tool_call['name']

        approved = approval_handler(tool_name)

        if not approved:
            return ToolMessage(
                content="Tool call denied by user.",
                tool_call_id=request.tool_call['id']
            )
        return handler(request)

    except Exception as e:
        return ToolMessage(
            content=f"Tool execution failed: {str(e)}",
            tool_call_id=request.tool_call['id']
        )


tavily = TavilyClient(
    api_key=os.getenv("TAVILY_API_KEY")
)


@tool
def web_search(query: str) -> str:
    """Search the web for recent and reliable information on a topic . Returns Titles , URLs and snippets."""
    results = tavily.search(
        query=query,
        max_results=5
    )
    out = []
    for r in results['results']:
        out.append(
            f"Title: {r['title']}\nURL: {r['url']}\nContent: {r['content'][:400]}\n"
        )

    return '\n-----\n'.join(out)


@tool
def scrape_url(url: str) -> str:
    """Scrape and return clean text content from a given URL for deeper reading."""
    try:
        resp = requests.get(
            url,
            timeout=6,
            headers={"User-Agent": "Mozilla/5.0"}
        )
        soup = BeautifulSoup(
            resp.text, "html.parser"
        )
        for tag in soup(["script", "style", "nav", "footer"]):
            tag.decompose()
        return soup.get_text(separator=" ", strip=True)[:4000]
    except Exception as e:
        return f"Could not scrape URL {str(e)}"