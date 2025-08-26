import logging
import os

import requests

from ..utils import build_domain_query


class JinaSearch:
    """Jina AI search retriever using the public search API."""

    def __init__(self, query: str, query_domains=None):
        self.query = query
        self.query_domains = query_domains or []
        self.api_key = os.environ.get("JINA_API_KEY")
        self.logger = logging.getLogger(__name__)

    def search(self, max_results: int = 10) -> list[dict[str, str]]:
        """Search via Jina AI API."""
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        payload = {
            "query": build_domain_query(self.query, self.query_domains),
            "top_k": max_results,
        }
        try:
            resp = requests.post(
                "https://api.jina.ai/v1/search",
                headers=headers,
                json=payload,
                timeout=10,
            )
            data = resp.json().get("data", [])
        except Exception as e:
            self.logger.error(f"Jina search failed: {e}")
            return []
        results = []
        for idx, item in enumerate(data, start=1):
            results.append(
                {
                    "id": idx,
                    "title": item.get("title", ""),
                    "href": item.get("url"),
                    "body": item.get("snippet", ""),
                }
            )
        return results
