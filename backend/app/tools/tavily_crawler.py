from typing import Any, Dict, List, Optional
import httpx
from bs4 import BeautifulSoup
from app.tools.base import ToolExecutionError
from app.tools.serper_search import extract_core_keywords


class TavilyCrawlerTool:
    """
    Web Crawler & Deep Content Extraction Tool powered by Tavily API (/search & /extract).
    Crawls target URLs or deep-searches a topic to extract clean Markdown text ready for FAISS indexing.
    """

    name = "tavily_web_crawler"
    description = (
        "Crawls web pages and extracts full, clean textual content using Tavily API "
        "so the agent can deeply read articles and index them into FAISS vector memory."
    )

    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key

    async def execute(
        self, query_or_url: str, urls: Optional[List[str]] = None, max_pages: int = 3
    ) -> Dict[str, Any]:
        target = (query_or_url or "").strip()
        if not target and not urls:
            raise ToolExecutionError(
                self.name,
                "InvalidInputError",
                "tavily_web_crawler requires a query_or_url or list of urls.",
            )

        if self.api_key and self.api_key != "your_tavily_api_key_here":
            return await self._tavily_api_crawl(target, urls, max_pages)

        return await self._live_http_crawler_fallback(target, urls, max_pages)

    async def _tavily_api_crawl(
        self, query_or_url: str, urls: Optional[List[str]], max_pages: int
    ) -> Dict[str, Any]:
        async with httpx.AsyncClient(timeout=15.0) as client:
            if urls or query_or_url.startswith(("http://", "https://")):
                target_urls = urls if urls else [query_or_url]
                try:
                    resp = await client.post(
                        "https://api.tavily.com/extract",
                        json={"api_key": self.api_key, "urls": target_urls[:max_pages]},
                    )
                    if resp.status_code == 200:
                        data = resp.json()
                        pages = []
                        for item in data.get("results", []):
                            raw_text = item.get("raw_content", "")
                            if raw_text:
                                pages.append(
                                    {
                                        "title": item.get("url", "Extracted Page"),
                                        "url": item.get("url", ""),
                                        "content": raw_text[:3000],
                                        "source_tool": self.name,
                                    }
                                )
                        if pages:
                            return {
                                "provider": "tavily_extract_api",
                                "pages_crawled": len(pages),
                                "pages": pages,
                            }
                except Exception:
                    pass

            try:
                resp = await client.post(
                    "https://api.tavily.com/search",
                    json={
                        "api_key": self.api_key,
                        "query": query_or_url,
                        "search_depth": "advanced",
                        "include_answer": True,
                        "include_raw_content": True,
                        "max_results": max_pages,
                    },
                )
                if resp.status_code != 200:
                    raise ToolExecutionError(
                        self.name,
                        f"HTTP{resp.status_code}Error",
                        f"Tavily API returned status {resp.status_code}: {resp.text[:200]}",
                    )
                data = resp.json()
                pages = []
                for r in data.get("results", []):
                    body = r.get("raw_content") or r.get("content") or ""
                    if body:
                        pages.append(
                            {
                                "title": r.get("title", "Crawled Document"),
                                "url": r.get("url", ""),
                                "content": body[:3000],
                                "source_tool": self.name,
                            }
                        )
                if not pages:
                    raise ToolExecutionError(
                        self.name,
                        "EmptyCrawlError",
                        f"Tavily returned no crawlable content for '{query_or_url}'.",
                    )
                return {
                    "provider": "tavily_deep_crawl_api",
                    "synthesized_answer": data.get("answer", ""),
                    "pages_crawled": len(pages),
                    "pages": pages,
                }
            except ToolExecutionError:
                raise
            except Exception as exc:
                raise ToolExecutionError(
                    self.name, "NetworkError", f"Tavily API crawl error: {exc}"
                )

    async def _live_http_crawler_fallback(
        self, query_or_url: str, urls: Optional[List[str]], max_pages: int
    ) -> Dict[str, Any]:
        """Live keyless HTTP page crawler + Wikipedia Extract API + HN Algolia fallback."""
        pages: List[Dict[str, str]] = []
        headers = {"User-Agent": "AgenticAIInternSystem/1.0 (LiveCrawler)"}
        kw_4 = extract_core_keywords(query_or_url, max_words=4)
        kw_2 = extract_core_keywords(query_or_url, max_words=2)

        async with httpx.AsyncClient(timeout=10.0, headers=headers, follow_redirects=True) as client:
            candidate_urls = list(urls or [])
            if query_or_url.startswith(("http://", "https://")):
                candidate_urls.insert(0, query_or_url)

            # 1. Crawl direct URLs if provided
            for u in candidate_urls[:max_pages]:
                try:
                    resp = await client.get(u)
                    if resp.status_code == 200 and "text/html" in resp.headers.get("content-type", ""):
                        soup = BeautifulSoup(resp.text, "html.parser")
                        for tag in soup(["script", "style", "nav", "footer", "header"]):
                            tag.decompose()
                        title = soup.title.string.strip() if soup.title and soup.title.string else u
                        paragraphs = [
                            p.get_text(" ", strip=True)
                            for p in soup.find_all("p")
                            if len(p.get_text(strip=True)) > 40
                        ]
                        text = "\n".join(paragraphs)[:2800]
                        if text:
                            pages.append(
                                {
                                    "title": title,
                                    "url": u,
                                    "content": text,
                                    "source_tool": self.name,
                                }
                            )
                except Exception:
                    continue

            # 2. Pull full encyclopedic article extracts via MediaWiki API using extracted core keywords
            for q_variant in [kw_4, kw_2]:
                if len(pages) >= max_pages:
                    break
                try:
                    s_resp = await client.get(
                        "https://en.wikipedia.org/w/api.php",
                        params={
                            "action": "query",
                            "generator": "search",
                            "gsrsearch": q_variant,
                            "gsrlimit": max_pages,
                            "prop": "extracts|info",
                            "exintro": False,
                            "explaintext": True,
                            "exchars": 2200,
                            "inprop": "url",
                            "format": "json",
                        },
                    )
                    if s_resp.status_code == 200:
                        page_dict = s_resp.json().get("query", {}).get("pages", {})
                        for _, pdata in page_dict.items():
                            extract = pdata.get("extract", "").strip()
                            if extract and len(pages) < max_pages:
                                pages.append(
                                    {
                                        "title": pdata.get("title", "Extracted Article"),
                                        "url": pdata.get("fullurl", "https://en.wikipedia.org"),
                                        "content": extract,
                                        "source_tool": self.name,
                                    }
                                )
                except Exception:
                    pass

            # 3. Also pull HackerNews Algolia stories/texts if still below max_pages
            if len(pages) < max_pages:
                try:
                    hn_resp = await client.get(
                        "https://hn.algolia.com/api/v1/search",
                        params={"query": kw_2, "tags": "story", "hitsPerPage": max_pages},
                    )
                    if hn_resp.status_code == 200:
                        for h in hn_resp.json().get("hits", []):
                            title = h.get("title")
                            url = h.get("url") or f"https://news.ycombinator.com/item?id={h.get('objectID')}"
                            story_text = h.get("story_text") or (
                                f"Article: {title}. Published on {h.get('created_at', '')[:10]} by {h.get('author', 'researcher')}. "
                                f"Community engagement: {h.get('points', 0)} points and {h.get('num_comments', 0)} technical comments."
                            )
                            if title and len(pages) < max_pages:
                                pages.append(
                                    {
                                        "title": title,
                                        "url": url,
                                        "content": story_text,
                                        "source_tool": self.name,
                                    }
                                )
                except Exception:
                    pass

        if not pages:
            raise ToolExecutionError(
                self.name,
                "EmptyCrawlError",
                f"Live crawler could not extract readable content for '{query_or_url}'.",
            )

        return {
            "provider": "live_http_beautifulsoup_and_mediawiki_crawler",
            "effective_keywords": kw_4,
            "pages_crawled": len(pages),
            "pages": pages[:max_pages],
        }
