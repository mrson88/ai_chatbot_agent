from typing import Any, Dict
import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin
import time

from dotenv import load_dotenv
from langchain_core.documents import Document

from graph.state import GraphState

load_dotenv()


def scrape_url(url: str, timeout: int = 5) -> str:
    """Scrape content from a URL"""
    try:
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
        }
        response = requests.get(url, headers=headers, timeout=timeout)
        response.raise_for_status()

        soup = BeautifulSoup(response.content, "html.parser")

        # Remove script and style elements
        for script in soup(["script", "style"]):
            script.decompose()

        # Get text
        text = soup.get_text(separator=" ", strip=True)

        # Clean up whitespace
        lines = (line.strip() for line in text.splitlines())
        chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
        text = " ".join(chunk for chunk in chunks if chunk)

        return text[:2000]  # Limit to 2000 chars
    except Exception as e:
        print(f"Error scraping {url}: {e}")
        return ""


def google_search_urls(query: str, num_results: int = 3) -> list:
    """Get search results using DuckDuckGo (no API key needed)"""
    try:
        from ddgs import DDGS

        results = []
        with DDGS() as ddgs:
            search_results = ddgs.text(query, max_results=num_results)
            for result in search_results:
                results.append({
                    "title": result.get("title", ""),
                    "url": result.get("href", ""),
                    "snippet": result.get("body", "")
                })
        return results
    except ImportError:
        print("DDGS library not found. Install with: pip install ddgs")
        return []


def web_search(state: GraphState) -> Dict[str, Any]:
    print("---WEB SEARCH---")
    question = state["question"]
    if "documents" in state:
        documents = state["documents"]
    else:
        documents = None

    print(f"Searching for: {question}")

    # Get search results
    search_results = google_search_urls(question, num_results=3)

    if not search_results:
        print("No search results found")
        if documents is None:
            documents = []
        return {
            "documents": documents,
            "question": question,
            "retry_count": state.get("retry_count", 0),
        }

    # Scrape content from top results
    web_content = []
    for result in search_results:
        print(f"Fetching: {result['title']}")
        content = scrape_url(result["url"])
        if content:
            web_content.append(f"Title: {result['title']}\n{content}")
        time.sleep(1)  # Be respectful to servers

    joined_result = "\n\n---\n\n".join(web_content)
    web_results = Document(page_content=joined_result)

    if documents is not None:
        documents.append(web_results)
    else:
        documents = [web_results]

    return {
        "documents": documents,
        "question": question,
        "retry_count": state.get("retry_count", 0),
    }


if __name__ == "__main__":
    web_search(state={"question": "agent memory", "documents": None})
