import importlib.util
import logging
import os
import requests

logger = logging.getLogger(__name__)

async def stream_output(log_type, step, content, websocket=None, with_data=False, data=None):
    """
    Stream output to the client.
    
    Args:
        log_type (str): The type of log
        step (str): The step being performed
        content (str): The content to stream
        websocket: The websocket to stream to
        with_data (bool): Whether to include data
        data: Additional data to include
    """
    if websocket:
        try:
            if with_data:
                await websocket.send_json({
                    "type": log_type,
                    "step": step,
                    "content": content,
                    "data": data
                })
            else:
                await websocket.send_json({
                    "type": log_type,
                    "step": step,
                    "content": content
                })
        except Exception as e:
            logger.error(f"Error streaming output: {e}")

def check_pkg(pkg: str) -> None:
    """
    Checks if a package is installed and raises an error if not.
    
    Args:
        pkg (str): The package name
    
    Raises:
        ImportError: If the package is not installed
    """
    if not importlib.util.find_spec(pkg):
        pkg_kebab = pkg.replace("_", "-")
        raise ImportError(
            f"Unable to import {pkg_kebab}. Please install with "
            f"`pip install -U {pkg_kebab}`"
        )


def build_domain_query(query: str, domains: list[str] | None) -> str:
    """Append domain filters to a search query.

    Args:
        query: The original search query.
        domains: List of domains to restrict the search to.

    Returns:
        The query string with ``site:`` filters appended if domains are
        provided.
    """
    if not domains:
        return query
    domain_query = " OR ".join([f"site:{domain}" for domain in domains])
    return f"{query} {domain_query}"


def jina_rerank(query: str, documents: list[dict], top_n: int | None = None) -> list[dict]:
    """Rerank documents using Jina AI reranker API."""
    api_key = os.environ.get("JINA_API_KEY")
    if not api_key:
        return documents
    try:
        payload = {
            "model": "jina-reranker-v1",
            "query": query,
            "documents": [d.get("body", "") for d in documents],
        }
        headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
        resp = requests.post("https://api.jina.ai/v1/rerank", json=payload, headers=headers, timeout=10)
        data = resp.json().get("data", [])
        ranked = sorted(zip(documents, data), key=lambda x: x[1]["relevance_score"], reverse=True)
        reranked = [doc for doc, _ in ranked]
        return reranked[:top_n] if top_n else reranked
    except Exception as e:
        logger.error(f"Jina rerank failed: {e}")
        return documents

# Valid retrievers for fallback
VALID_RETRIEVERS = [
    "tavily",
    "custom",
    "duckduckgo",
    "searchapi",
    "serper",
    "serpapi",
    "google",
    "searx",
    "bing",
    "arxiv",
    "semantic_scholar",
    "pubmed_central",
    "exa",
    "mcp",
    "mock",
    "jina",
]

def get_all_retriever_names():
    """
    Get all available retriever names
    :return: List of all available retriever names
    :rtype: list
    """
    try:
        current_dir = os.path.dirname(os.path.abspath(__file__))

        # Get all items in the current directory
        all_items = os.listdir(current_dir)

        # Filter out only the directories, excluding __pycache__
        retrievers = [
            item for item in all_items
            if os.path.isdir(os.path.join(current_dir, item)) and not item.startswith('__')
        ]

        return retrievers
    except Exception as e:
        logger.error(f"Error getting retrievers: {e}")
        return VALID_RETRIEVERS
