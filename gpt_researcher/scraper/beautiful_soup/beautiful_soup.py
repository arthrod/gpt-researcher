from bs4 import BeautifulSoup

from ..utils import get_relevant_images, extract_title, get_text_from_soup, clean_soup

class BeautifulSoupScraper:

    def __init__(self, link, session=None):
        self.link = link
        self.session = session

    def scrape(self):
        """
        Scrape the webpage at self.link and return its cleaned text, relevant image URLs, and title.
        
        Performs an HTTP GET for the stored URL, parses the HTML with BeautifulSoup, applies cleaning, and extracts the page text, a list of relevant image URLs, and the page title.
        
        Returns:
            tuple:
                content (str): Cleaned text content of the page (empty string on failure).
                image_urls (list[str]): Relevant image URLs discovered on the page (empty list on failure).
                title (str): Page title (empty string on failure).
        """
        try:
            response = self.session.get(self.link, timeout=4)
            soup = BeautifulSoup(
                response.content, "lxml", from_encoding=response.encoding
            )

            soup = clean_soup(soup)

            content = get_text_from_soup(soup)

            image_urls = get_relevant_images(soup, self.link)

            # Extract the title using the utility function
            title = extract_title(soup)

            return content, image_urls, title

        except Exception as e:
            print("Error! : " + str(e))
            return "", [], ""