from __future__ import annotations

import os
import pickle
import random
import string
import time
import traceback

from pathlib import Path
from sys import platform

from bs4 import BeautifulSoup

<<<<<<< HEAD
from .processing.scrape_skills import (scrape_pdf_with_pymupdf,
                                       scrape_pdf_with_arxiv)


from ..utils import get_relevant_images, extract_title, get_text_from_soup, clean_soup
=======
from ..utils import clean_soup, extract_title, get_relevant_images, get_text_from_soup
from .processing.scrape_skills import scrape_pdf_with_arxiv, scrape_pdf_with_pymupdf
>>>>>>> newdev

FILE_DIR = Path(__file__).parent.parent


class BrowserScraper:
    def __init__(self, url: str, session=None):
        self.url = url
        self.session = session
        self.selenium_web_browser = "chrome"
        self.headless = False
        self.user_agent = (
            "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/128.0.0.0 Safari/537.36"
        )
        self.driver = None
        self.use_browser_cookies = False
        self._import_selenium()  # Import only if used to avoid unnecessary dependencies
        self.cookie_filename = f"{self._generate_random_string(8)}.pkl"

    def scrape(self) -> tuple:
        """
        Scrape the configured URL using a Selenium WebDriver and return extracted text, images, and title.
        
        Attempts to start a browser driver, establish cookies (visits Google to seed cookies, loads saved/browser cookies when configured), injects a header overlay, and extract page content. Returns a tuple (text, image_urls, title) on success. If no URL is set, returns an explanatory error string in place of text and empty lists/strings for the other fields. On unexpected errors, returns an error message that includes the exception string and full stack trace as the first tuple element; image_urls and title will be empty in that case.
        
        The method ensures the WebDriver is quit and any temporary cookie file is cleaned up before returning.
        """
        if not self.url:
            print("URL not specified")
            return (
                "A URL was not specified, cancelling request to browse website.",
                [],
                "",
            )

        try:
            self.setup_driver()
            self._visit_google_and_save_cookies()
            self._load_saved_cookies()
            self._add_header()

            text, image_urls, title = self.scrape_text_with_selenium()
            return text, image_urls, title
        except Exception as e:
            print(f"An error occurred during scraping: {e!s}")
            print("Full stack trace:")
            print(traceback.format_exc())
<<<<<<< HEAD
            return f"An error occurred: {e!s}\n\nStack trace:\n{traceback.format_exc()}", [], ""
=======
            return (
                f"An error occurred: {e!s}\n\nStack trace:\n{traceback.format_exc()}",
                [],
                "",
            )
>>>>>>> newdev
        finally:
            if self.driver:
                self.driver.quit()
            self._cleanup_cookie_file()

    def _import_selenium(self):
        """
        Ensure Selenium is available and import required Selenium classes into module globals.
        
        This initializes the Selenium-related symbols used elsewhere in the module by importing:
        `webdriver`, `By`, `EC`, `WebDriverWait`, `TimeoutException`, `WebDriverException`,
        and browser option classes `ChromeOptions`, `FirefoxOptions`, `SafariOptions`.
        If Selenium is not installed, prints brief installation guidance and re-raises ImportError with a descriptive message.
        """
        try:
            global \
                webdriver, \
                By, \
                EC, \
                WebDriverWait, \
                TimeoutException, \
                WebDriverException
            from selenium import webdriver
            from selenium.common.exceptions import TimeoutException, WebDriverException
            from selenium.webdriver.common.by import By
            from selenium.webdriver.support import expected_conditions as EC
            from selenium.webdriver.support.wait import WebDriverWait

            global ChromeOptions, FirefoxOptions, SafariOptions
            from selenium.webdriver.chrome.options import Options as ChromeOptions
            from selenium.webdriver.firefox.options import Options as FirefoxOptions
            from selenium.webdriver.safari.options import Options as SafariOptions
        except ImportError as e:
            print(f"Failed to import Selenium: {e!s}")
            print("Please install Selenium and its dependencies to use BrowserScraper.")
            print("You can install Selenium using pip:")
            print("    pip install selenium")
            print("If you're using a virtual environment, make sure it's activated.")
            raise ImportError(
                "Selenium is required but not installed. See error message above for installation instructions."
            ) from e

    def setup_driver(self) -> None:
        # print(f"Setting up {self.selenium_web_browser} driver...")

        """
        Initialize and configure the Selenium WebDriver for the selected browser.
        
        Sets browser options (user agent, headless mode, JavaScript) and creates a WebDriver instance for Chrome, Firefox, or Safari.
        Applies Linux-specific Chrome arguments and Chrome download preferences when appropriate. If `use_browser_cookies` is true, loads browser cookies after the driver is created.
        
        Raises:
            Exception: Re-raises any exception raised while creating or configuring the WebDriver.
        """
        options_available = {
            "chrome": ChromeOptions,
            "firefox": FirefoxOptions,
            "safari": SafariOptions,
        }

        options = options_available[self.selenium_web_browser]()
        options.add_argument(f"user-agent={self.user_agent}")
        if self.headless:
            options.add_argument("--headless")
        options.add_argument("--enable-javascript")

        try:
            if self.selenium_web_browser == "firefox":
                self.driver = webdriver.Firefox(options=options)
            elif self.selenium_web_browser == "safari":
                self.driver = webdriver.Safari(options=options)
            else:  # chrome
                if platform == "linux" or platform == "linux2":
                    options.add_argument("--disable-dev-shm-usage")
                    options.add_argument("--remote-debugging-port=9222")
                options.add_argument("--no-sandbox")
                options.add_experimental_option("prefs", {"download_restrictions": 3})
                self.driver = webdriver.Chrome(options=options)

            if self.use_browser_cookies:
                self._load_browser_cookies()

            # print(f"{self.selenium_web_browser.capitalize()} driver set up successfully.")
        except Exception as e:
            print(f"Failed to set up {self.selenium_web_browser} driver: {e!s}")
            print("Full stack trace:")
            print(traceback.format_exc())
            raise

    def _load_saved_cookies(self):
        """Load saved cookies before visiting the target URL"""
        cookie_file = Path(self.cookie_filename)
        if cookie_file.exists():
            with open(self.cookie_filename, "rb") as f:
                cookies = pickle.load(f)
            for cookie in cookies:
                self.driver.add_cookie(cookie)
        else:
            print("No saved cookies found.")

    def _load_browser_cookies(self):
        """Load cookies directly from the browser"""
        try:
            import browser_cookie3
        except ImportError:
            print(
                "browser_cookie3 is not installed. Please install it using: pip install browser_cookie3"
            )
            return

        if self.selenium_web_browser == "chrome":
            cookies = browser_cookie3.chrome()
        elif self.selenium_web_browser == "firefox":
            cookies = browser_cookie3.firefox()
        else:
            print(f"Cookie loading not supported for {self.selenium_web_browser}")
            return

        for cookie in cookies:
            self.driver.add_cookie({
                "name": cookie.name,
                "value": cookie.value,
                "domain": cookie.domain,
            })

    def _cleanup_cookie_file(self):
        """
        Remove the cookie file created for this scraper, if present.
        
        Attempts to delete self.cookie_filename from disk. If the file does not exist the function returns quietly (it prints a notice); any failure to remove the file is printed to stdout.
        """
        cookie_file = Path(self.cookie_filename)
        if cookie_file.exists():
            try:
                os.remove(self.cookie_filename)
            except Exception as e:
                print(f"Failed to remove cookie file: {e!s}")
        else:
            print("No cookie file found to remove.")

    def _generate_random_string(self, length):
        """Generate a random string of specified length"""
        return "".join(random.choices(string.ascii_letters + string.digits, k=length))

    def _get_domain(self):
        """Extract domain from URL"""
        from urllib.parse import urlparse

        """Get domain from URL, removing 'www' if present"""
        domain = urlparse(self.url).netloc
        return domain[4:] if domain.startswith("www.") else domain

    def _visit_google_and_save_cookies(self):
        """
        Navigate the webdriver to https://www.google.com and persist any cookies to self.cookie_filename.
        
        This method loads Google (to trigger browser cookie setting), waits briefly for cookies to appear, then serializes the driver's cookies to the instance's cookie file. Side effects: navigates the active webdriver and writes a pickle file at self.cookie_filename. Exceptions are caught and logged; the method does not raise.
        """
        try:
            self.driver.get("https://www.google.com")
            time.sleep(2)  # Wait for cookies to be set

            # Save cookies to a file
            cookies = self.driver.get_cookies()
            with open(self.cookie_filename, "wb") as f:
                pickle.dump(cookies, f)

            # print("Google cookies saved successfully.")
        except Exception as e:
            print(f"Failed to visit Google and save cookies: {e!s}")
            print("Full stack trace:")
            print(traceback.format_exc())

    def scrape_text_with_selenium(self) -> tuple:
        """
        Load the configured URL with Selenium and extract page text, relevant image URLs, and the page title.
        
        This method waits for the page body to be present, scrolls to the bottom to trigger dynamic content loading, and then extracts content depending on the URL type:
        - For URLs ending with `.pdf`: extract text from the PDF using the internal PDF scraper and return no images or title.
        - For ArXiv pages (URL contains "arxiv"): extract the arXiv PDF text via the arXiv scraper and return no images or title.
        - For regular HTML pages: return the page text, a list of relevant image URLs, and the extracted title.
        
        Returns:
            tuple: A 3-tuple (text, image_urls, title).
                - text (str): Extracted textual content, or the error marker string "Page load timed out" if the initial page load timed out.
                - image_urls (list): List of image URLs relevant to the page (empty for PDF and arXiv extractions or on timeout).
                - title (str): Extracted page title (empty for PDF and arXiv extractions or on timeout).
        """
        self.driver.get(self.url)

        try:
            WebDriverWait(self.driver, 20).until(
                EC.presence_of_element_located((By.TAG_NAME, "body"))
            )
        except TimeoutException:
            print("Timed out waiting for page to load")
            print(f"Full stack trace:\n{traceback.format_exc()}")
            return "Page load timed out", [], ""

        self._scroll_to_bottom()

        if self.url.endswith(".pdf"):
            text = scrape_pdf_with_pymupdf(self.url)
            return text, [], ""
        elif "arxiv" in self.url:
            doc_num = self.url.split("/")[-1]
            text = scrape_pdf_with_arxiv(doc_num)
            return text, [], ""
        else:
            page_source = self.driver.execute_script(
                "return document.documentElement.outerHTML;"
            )
            soup = BeautifulSoup(page_source, "lxml")

            soup = clean_soup(soup)

            text = get_text_from_soup(soup)
            image_urls = get_relevant_images(soup, self.url)
            title = extract_title(soup)

        return text, image_urls, title

    def _scroll_to_bottom(self):
        """Scroll to the bottom of the page to load all content"""
        last_height = self.driver.execute_script("return document.body.scrollHeight")
        while True:
            self.driver.execute_script(
                "window.scrollTo(0, document.body.scrollHeight);"
            )
            time.sleep(2)  # Wait for content to load
            new_height = self.driver.execute_script("return document.body.scrollHeight")
            if new_height == last_height:
                break
            last_height = new_height

    def _scroll_to_percentage(self, ratio: float) -> None:
        """Scroll to a percentage of the page"""
        if ratio < 0 or ratio > 1:
            raise ValueError("Percentage should be between 0 and 1")
        self.driver.execute_script(
            f"window.scrollTo(0, document.body.scrollHeight * {ratio});"
        )

    def _add_header(self) -> None:
        """Add a header to the website"""
        with open(f"{FILE_DIR}/browser/js/overlay.js") as f:
            self.driver.execute_script(f.read())
