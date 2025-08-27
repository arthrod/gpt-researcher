from langchain_community.retrievers import ArxivRetriever


class ArxivScraper:
    def __init__(self, link, session=None):
        self.link = link
        self.session = session

    def scrape(self):
        """
<<<<<<< HEAD
        Scrape an arXiv entry from self.link and return assembled context, images list, and the document title.
        
        Uses the last path segment of self.link as the arXiv query, retrieves up to two documents via ArxivRetriever, and builds a context string containing the published date, authors, and the first document's page content.
        
=======
        The function scrapes relevant documents from Arxiv based on a given link and returns the content
        of the first document.

>>>>>>> newdev
        Returns:
            tuple:
                context (str): "Published: {Published}; Author: {Authors}; Content: {page_content}" built from the first retrieved document.
                image (list): Currently an empty list (placeholder for extracted images).
                title (str): The Title metadata of the first retrieved document.
        
        Notes:
            - Assumes at least one document is returned and that the first document contains 'Published', 'Authors', and 'Title' keys in metadata and a non-empty page_content. If these are missing, a KeyError or IndexError may occur.
        """
        query = self.link.split("/")[-1]
        retriever = ArxivRetriever(load_max_docs=2, doc_content_chars_max=None)
        docs = retriever.invoke(query)

        # Include the published date and author to provide additional context,
        # aligning with APA-style formatting in the report.
        context = f"Published: {docs[0].metadata['Published']}; Author: {docs[0].metadata['Authors']}; Content: {docs[0].page_content}"
        image = []

        return context, image, docs[0].metadata["Title"]
