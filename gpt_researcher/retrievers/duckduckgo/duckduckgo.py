from ..utils import check_pkg, build_domain_query


class Duckduckgo:
    """
    Duckduckgo API Retriever
    """
    def __init__(self, query, query_domains=None):
        check_pkg('duckduckgo_search')
        from duckduckgo_search import DDGS
        self.ddg = DDGS()
        self.query = query
        self.query_domains = query_domains or []

    def search(self, max_results=5):
        """
        Performs the search
        :param query:
        :param max_results:
        :return:
        """
        query = build_domain_query(self.query, self.query_domains)
        try:
            results = self.ddg.text(query, region='wt-wt', max_results=max_results)
        except Exception as e:
            print(f"Error: {e}. Failed fetching sources. Resulting in empty response.")
            results = []
        search_results = []
        for idx, result in enumerate(results, start=1):
            if "Deep search" in result.get("body", ""):
                continue
            result["id"] = idx
            search_results.append(result)
        return search_results
