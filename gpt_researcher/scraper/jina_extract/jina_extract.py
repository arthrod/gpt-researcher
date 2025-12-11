import os
import requests

class JinaExtract:

    def __init__(self, link, session=None):
        self.link = link
        self.session = session
        self.api_key = self.get_api_key()

    def get_api_key(self) -> str:
        """
        Gets the Jina API key
        Returns:
        Api key (str)
        """
        try:
            api_key = os.environ["JINA_API_KEY"]
        except KeyError:
            raise Exception(
                "Jina API key not found. Please set the JINA_API_KEY environment variable.")
        return api_key

    def scrape(self) -> tuple:
        """
        This function extracts content from a specified link using the Jina Reader API.

        Returns:
          The `scrape` method returns a tuple containing the extracted content, a list of image URLs, and
        the title of the webpage specified by the `self.link` attribute.
        """

        try:
            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "X-Retain-Images": "none"
            }
            # Jina Reader API: https://r.jina.ai/<url>
            jina_url = f"https://r.jina.ai/{self.link}"

            response = requests.get(jina_url, headers=headers, timeout=10)

            if response.status_code == 200:
                content = response.text

                # Jina Reader returns the content directly.
                # We might need to parse title/images if Jina doesn't provide them in a structured way (it returns markdown).
                # However, the scraper interface expects (content, image_urls, title).
                # Jina Reader response is the markdown content.
                # Title is usually the first line or we can try to extract it.
                # Images might be in the markdown if X-Retain-Images is not none, but let's stick to text for now or simple extraction.

                # Let's try to get title from the first line if it looks like a title (# Title)
                lines = content.split('\n')
                title = ""
                if lines and lines[0].startswith('# '):
                    title = lines[0][2:].strip()

                # Image extraction from markdown is possible but let's return empty list for now unless we parse the markdown.
                # The existing scrapers return image_urls.

                return content, [], title
            else:
                print(f"Jina Reader failed with status code {response.status_code}")
                return "", [], ""

        except Exception as e:
            print("Error! : " + str(e))
            return "", [], ""
