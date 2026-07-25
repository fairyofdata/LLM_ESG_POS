"""Naver News crawling used by the "recent news" page.

A headless Chrome (managed by ``webdriver_manager``) walks the Naver search
result pages and collects links that resolve to ``news.naver.com`` articles;
titles are then fetched with plain HTTP requests.
"""

import time

import requests
from bs4 import BeautifulSoup
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from webdriver_manager.chrome import ChromeDriverManager

_REQUEST_HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) Chrome/98.0.4758.102"
}


def setup_webdriver() -> webdriver.Chrome:
    """Create a headless Chrome driver suitable for CI/server environments."""
    options = webdriver.ChromeOptions()
    options.add_experimental_option("excludeSwitches", ["enable-automation"])
    options.add_experimental_option("useAutomationExtension", False)
    options.add_argument("--headless")
    options.add_argument("--no-sandbox")
    options.add_argument("--disable-dev-shm-usage")

    service = Service(ChromeDriverManager().install())
    driver = webdriver.Chrome(service=service, options=options)
    driver.implicitly_wait(3)
    return driver


def make_search_url(query: str, page: int) -> str:
    """Build the Naver news search URL for a 1-based result ``page``."""
    start = 1 + 10 * (page - 1)
    return (
        "https://search.naver.com/search.naver"
        f"?where=news&sm=tab_pge&query={query}&start={start}"
    )


def crawl_naver_news(query: str, article_count: int = 10) -> list[tuple[str, str]]:
    """Collect recent Naver news articles for a search keyword.

    Args:
        query: Search keyword.
        article_count: Number of articles to collect.

    Returns:
        List of ``(title, url)`` tuples.
    """
    driver = setup_webdriver()
    naver_urls: list[str] = []
    current_page = 1
    try:
        while len(naver_urls) < article_count:
            driver.get(make_search_url(query, current_page))
            time.sleep(1)

            for a_tag in driver.find_elements(By.CSS_SELECTOR, "a.info"):
                if len(naver_urls) >= article_count:
                    break
                a_tag.click()
                driver.switch_to.window(driver.window_handles[1])
                time.sleep(3)

                url = driver.current_url
                if "news.naver.com" in url:
                    naver_urls.append(url)

                driver.close()
                driver.switch_to.window(driver.window_handles[0])

            current_page += 1
    finally:
        driver.quit()
    return fetch_news_titles(naver_urls)


def fetch_news_titles(urls: list[str]) -> list[tuple[str, str]]:
    """Fetch article titles for a list of ``news.naver.com`` URLs."""
    news_list: list[tuple[str, str]] = []
    for url in urls:
        response = requests.get(url, headers=_REQUEST_HEADERS)
        html = BeautifulSoup(response.text, "html.parser")
        title_element = html.select_one(
            "div#ct > div.media_end_head.go_trans > div.media_end_head_title > h2"
        )
        title = title_element.get_text(strip=True) if title_element else "No title found"
        news_list.append((title, url))
    return news_list
