<<<<<<< HEAD
from bs4 import BeautifulSoup
=======
>>>>>>> newdev
import requests

from bs4 import BeautifulSoup

from ..utils import extract_title, get_relevant_images


class WebBaseLoaderScraper:
    def __init__(self, link, session=None):
        self.link = link
        self.session = session or requests.Session()

    def scrape(self) -> tuple:
        """
<<<<<<< HEAD
        Scrape a web page's main text content, relevant image URLs, and HTML title.
        
        Uses a WebBaseLoader to load and concatenate document content for the given link, then performs an HTTP GET and parses the page with BeautifulSoup to extract relevant image URLs and the page title.
        
=======
        This Python function scrapes content from a webpage using a WebBaseLoader object and returns the
        concatenated page content.

>>>>>>> newdev
        Returns:
            tuple: (content, image_urls, title)
                - content (str): Concatenated page text produced by WebBaseLoader (empty string on failure).
                - image_urls (list[str]): List of relevant image URLs discovered in the page (empty list on failure).
                - title (str): The page title extracted from the HTML (empty string on failure).
        
        Notes:
            - The loader is configured to disable SSL verification for its requests.
            - Network operations are performed; on any exception the function prints an error and returns ("", [], "").
        """
        try:
            from langchain_community.document_loaders import WebBaseLoader

            loader = WebBaseLoader(self.link)
            loader.requests_kwargs = {"verify": False}
            docs = loader.load()
            content = ""

            for doc in docs:
                content += doc.page_content

            response = self.session.get(self.link)
            soup = BeautifulSoup(response.content, "html.parser")
            image_urls = get_relevant_images(soup, self.link)

            # Extract the title using the utility function
            title = extract_title(soup)

            return content, image_urls, title

        except Exception as e:
            print("Error! : " + str(e))
            return "", [], ""
