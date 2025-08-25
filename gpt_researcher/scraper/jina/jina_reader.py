import requests

class JinaAIScraper:
    """Scrape web pages using the public Jina AI reader API."""

    def __init__(self, link: str, session=None):
        self.link = link
        self.session = session or requests.Session()

    def scrape(self):
        """Fetch and parse page content via Jina AI reader.

        Returns:
            tuple[str, list, str]: content, image urls, title
        """
        try:
            resp = self.session.get(f"https://r.jina.ai/{self.link}", timeout=10)
            text = resp.text.splitlines()
            title = ""
            content_lines = []
            capture = False
            for line in text:
                if line.startswith("Title: "):
                    title = line.replace("Title: ", "").strip()
                if line.startswith("Markdown Content:"):
                    capture = True
                    continue
                if capture:
                    content_lines.append(line)
            content = "\n".join(content_lines).strip()
            return content, [], title
        except Exception as e:
            import logging
            logging.getLogger(__name__).error(f"Failed to scrape {self.link} using Jina AI Reader: {e}")
            return "", [], ""
