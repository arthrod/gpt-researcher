from .arxiv.arxiv import ArxivScraper
from .beautiful_soup.beautiful_soup import BeautifulSoupScraper
from .browser.browser import BrowserScraper
from .browser.nodriver_scraper import NoDriverScraper
from .firecrawl.firecrawl import FireCrawl
from .jina.jina_reader import JinaAIScraper
from .pymupdf.pymupdf import PyMuPDFScraper
from .scraper import Scraper
from .tavily_extract.tavily_extract import TavilyExtract
from .web_base_loader.web_base_loader import WebBaseLoaderScraper

__all__ = [
    "ArxivScraper",
    "BeautifulSoupScraper",
    "BrowserScraper",
    "FireCrawl",
<<<<<<< HEAD
    "JinaAIScraper",
=======
>>>>>>> 1027e1d0 (Fix linting issues)
    "NoDriverScraper",
    "PyMuPDFScraper",
    "Scraper",
    "TavilyExtract",
    "WebBaseLoaderScraper",
]
