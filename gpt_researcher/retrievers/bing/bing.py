# Bing Search Retriever

# libraries
import json
import logging
import os

import requests

from ..utils import build_domain_query


class BingSearch:
    """
    Bing Search Retriever
    """

    def __init__(self, query, query_domains=None):
        """
        Initializes the BingSearch object
        Args:
            query:
        """
        self.query = query
        self.query_domains = query_domains or []
        self.api_key = self.get_api_key()
        self.logger = logging.getLogger(__name__)

    def get_api_key(self):
        """
        Gets the Bing API key
        Returns:

        """
        try:
            api_key = os.environ["BING_API_KEY"]
        except KeyError:
            raise Exception(
                "Bing API key not found. Please set the BING_API_KEY environment variable."
            )
        return api_key

    def search(self, max_results=7) -> list[dict[str]]:
        """
        Searches the query
        Returns:

        """
        print(f"Searching with query {self.query}...")
        """Useful for general internet search queries using the Bing API."""

        # Search the query
        url = "https://api.bing.microsoft.com/v7.0/search"

        headers = {
            "Ocp-Apim-Subscription-Key": self.api_key,
            "Content-Type": "application/json",
        }
        query = build_domain_query(self.query, self.query_domains)
        params = {
            "responseFilter": "Webpages",
            "q": query,
            "count": max_results,
            "setLang": "en-GB",
            "textDecorations": False,
            "textFormat": "HTML",
            "safeSearch": "Strict",
        }

        resp = requests.get(url, headers=headers, params=params)

        # Preprocess the results
        if resp is None:
            return []
        try:
            search_results = json.loads(resp.text)
            results = search_results["webPages"]["value"]
        except Exception as e:
            self.logger.error(
                f"Error parsing Bing search results: {e}. Resulting in empty response."
            )
            return []
        if search_results is None:
            self.logger.warning(f"No search results found for query: {self.query}")
            return []
        search_results = []

        # Normalize the results to match the format of the other search APIs
        for idx, result in enumerate(results, start=1):
            # skip youtube results
            if "youtube.com" in result["url"]:
                continue
            search_result = {
                "id": idx,
                "title": result["name"],
                "href": result["url"],
                "body": result["snippet"],
            }
            search_results.append(search_result)

        return search_results
