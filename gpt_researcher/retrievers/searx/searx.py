import json
import os

from urllib.parse import urljoin

import requests

<<<<<<< HEAD

=======
>>>>>>> 1027e1d0 (Fix linting issues)
class SearxSearch:
    """
    SearxNG API Retriever
    """

    def __init__(self, query: str, query_domains=None):
        """
        Initializes the SearxSearch object
        Args:
            query: Search query string
        """
        self.query = query
        self.query_domains = query_domains or None
        self.base_url = self.get_searxng_url()

    def get_searxng_url(self) -> str:
        """
        Gets the SearxNG instance URL from environment variables
        Returns:
            str: Base URL of SearxNG instance
        """
        try:
            base_url = os.environ["SEARX_URL"]
            if not base_url.endswith("/"):
                base_url += "/"
            return base_url
        except KeyError:
            raise Exception(
                "SearxNG URL not found. Please set the SEARX_URL environment variable. "
                "You can find public instances at https://searx.space/"
            )

    def search(self, max_results: int = 10) -> list[dict[str, str]]:
        """
        Searches the query using SearxNG API
        Args:
            max_results: Maximum number of results to return
        Returns:
            List of dictionaries containing search results
        """
        search_url = urljoin(self.base_url, "search")
        # TODO: Add support for query domains
        params = {
            # The search query.
<<<<<<< HEAD
            "q": query,
=======
            'q': self.query,
>>>>>>> 1027e1d0 (Fix linting issues)
            # Output format of results. Format needs to be activated in searxng config.
            "format": "json",
        }

        try:
            response = requests.get(
                search_url, params=params, headers={"Accept": "application/json"}
            )
            response.raise_for_status()
            results = response.json()

            # Normalize results to match the expected format
            search_response = []
            for idx, result in enumerate(
                results.get("results", [])[:max_results], start=1
            ):
                search_response.append({
                    "id": idx,
                    "href": result.get("url", ""),
                    "body": result.get("content", ""),
                })

            return search_response

        except requests.exceptions.RequestException as e:
            raise Exception(f"Error querying SearxNG: {e!s}")
        except json.JSONDecodeError:
            raise Exception("Error parsing SearxNG response")
