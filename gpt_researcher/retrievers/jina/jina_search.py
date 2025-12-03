# Jina Search API Retriever

import os
import requests
import json
import urllib.parse

class JinaSearch:
    """
    Jina Search API Retriever
    """

    def __init__(self, query, headers=None, topic="general", query_domains=None):
        """
        Initializes the JinaSearch object.
        """
        self.query = query
        self.headers = headers or {}
        self.topic = topic
        self.base_url = "https://s.jina.ai/"
        self.api_key = self.get_api_key()
        self.query_domains = query_domains or None

    def get_api_key(self):
        """
        Gets the Jina API key
        """
        api_key = self.headers.get("jina_api_key")
        if not api_key:
            try:
                api_key = os.environ["JINA_API_KEY"]
            except KeyError:
                print(
                    "Jina API key not found. If you need a retriever, please set the JINA_API_KEY environment variable."
                )
                return ""
        return api_key

    def search(self, max_results=10):
        """
        Searches the query using Jina Search API
        """
        try:
            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Accept": "application/json"
            }
            # Jina Search API: https://s.jina.ai/<query>

            encoded_query = urllib.parse.quote(self.query)
            url = f"{self.base_url}{encoded_query}"

            response = requests.get(url, headers=headers, timeout=10)

            if response.status_code == 200:
                results = response.json()
                # The structure of Jina Search JSON response:
                # { "code": 200, "status": 200, "data": [ { "title": "...", "url": "...", "content": "..." }, ... ] }

                sources = results.get("data", [])
                if not sources:
                     # Fallback or empty
                     print("No results found with Jina Search API.")
                     return []

                # Limit to max_results
                sources = sources[:max_results]

                search_response = [
                    {"href": obj.get("url"), "body": obj.get("content"), "title": obj.get("title")} for obj in sources
                ]
                return search_response
            else:
                print(f"Jina Search failed with status code {response.status_code}")
                return []

        except Exception as e:
            print(f"Error: {e}. Failed fetching sources. Resulting in empty response.")
            return []
