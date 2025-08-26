import arxiv


class ArxivSearch:
    """
    Arxiv API Retriever
    """
<<<<<<< HEAD

    def __init__(self, query, sort="Relevance", query_domains=None):
=======
    def __init__(self, query, sort='Relevance', query_domains=None):
        """
        Initialize the ArxivSearch wrapper.
        
        Parameters:
        - query: Search query string passed to arXiv.
        - sort: 'Relevance' or 'SubmittedDate' (default 'Relevance'). If invalid, an AssertionError is raised with message "Invalid sort criterion".
        - query_domains: Accepted for API compatibility but ignored.
        
        Behavior:
        - Stores the arxiv module on self.arxiv and the query on self.query.
        - Maps the `sort` string to the corresponding arxiv.SortCriterion and stores it on self.sort.
        """
>>>>>>> 9a0c4dfe (📝 Add docstrings to `enhancements/highlevel-instructions`)
        self.arxiv = arxiv
        self.query = query
<<<<<<< HEAD
        assert sort in ["Relevance", "SubmittedDate"], "Invalid sort criterion"
        self.sort = (
            arxiv.SortCriterion.SubmittedDate
            if sort == "SubmittedDate"
            else arxiv.SortCriterion.Relevance
        )
=======
        assert sort in ['Relevance', 'SubmittedDate'], "Invalid sort criterion"
        self.sort = arxiv.SortCriterion.SubmittedDate if sort == 'SubmittedDate' else arxiv.SortCriterion.Relevance

>>>>>>> 1027e1d0 (Fix linting issues)

    def search(self, max_results=5):
        """
        Search arXiv for the instance's query and return a list of simplified result dictionaries.
        
        Performs an arXiv search using the query and sort criterion provided when the ArxivSearch instance was created, requesting up to max_results entries. Each returned item is a dictionary with keys:
        - "title": paper title (str)
        - "href": PDF URL (str)
        - "body": abstract/summary (str)
        
        Parameters:
            max_results (int): Maximum number of results to retrieve from the arXiv API (default 5).
        
        Returns:
            List[dict]: A list of result dictionaries containing "title", "href", and "body".
        """

        arxiv_gen = list(
            arxiv.Client().results(
                self.arxiv.Search(
                    query=self.query,  # +
                    max_results=max_results,
                    sort_by=self.sort,
                )
            )
        )

        search_result = []

        for result in arxiv_gen:
            search_result.append(
                {
                    "title": result.title,
                    "href": result.pdf_url,
                    "body": result.summary,
                }
            )

<<<<<<< HEAD
        return search_result
=======
            search_result.append({
                "title": result.title,
                "href": result.pdf_url,
                "body": result.summary,
            })

        return search_result
>>>>>>> 1027e1d0 (Fix linting issues)
