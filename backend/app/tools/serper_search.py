import re
from typing import Any, Dict, List, Optional
import httpx
from app.tools.base import ToolExecutionError

_STOPWORDS = {
    "given", "the", "company", "produce", "short", "using", "public", "web", "data",
    "research", "and", "summarize", "top", "developments", "from", "last", "week",
    "plan", "day", "itinerary", "for", "city", "your", "choice", "respecting",
    "budget", "interests", "identify", "potential", "issues", "suggest", "fixes",
    "with", "that", "this", "what", "how", "why", "into", "about", "comparing",
}


def extract_core_keywords(query: str, max_words: int = 5) -> str:
    tokens = re.findall(r"[A-Za-z0-9\-\.]+", query)
    filtered = [t for t in tokens if t.lower() not in _STOPWORDS and len(t) > 2]
    if not filtered:
        return query[:60]
    return " ".join(filtered[:max_words])


class SerperSearchTool:
    """
    Web Search Discovery Tool powered by Serper.dev Google Search API.
    If SERPER_API_KEY is not configured, transparently falls back to live public APIs
    (HackerNews Algolia + Wikipedia Search API) so inference-time live retrieval always works.
    """

    name = "serper_web_search"
    description = (
        "Executes a Google web search via Serper API to discover live news, URLs, "
        "knowledge graph facts, and organic search snippets."
    )

    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key

    async def execute(self, query: str, num_results: int = 5) -> Dict[str, Any]:
        cleaned_query = (query or "").strip()
        if not cleaned_query:
            raise ToolExecutionError(
                self.name, "InvalidInputError", "Search query cannot be empty."
            )

        if self.api_key and self.api_key != "your_serper_api_key_here":
            return await self._serper_api_search(cleaned_query, num_results)
        return await self._live_public_search_fallback(cleaned_query, num_results)

    async def _serper_api_search(self, query: str, num_results: int) -> Dict[str, Any]:
        url = "https://google.serper.dev/search"
        headers = {
            "X-API-KEY": self.api_key,
            "Content-Type": "application/json",
        }
        payload = {"q": query, "num": num_results}

        async with httpx.AsyncClient(timeout=12.0) as client:
            try:
                resp = await client.post(url, headers=headers, json=payload)
                if resp.status_code != 200:
                    raise ToolExecutionError(
                        self.name,
                        f"HTTP{resp.status_code}Error",
                        f"Serper API returned status {resp.status_code}: {resp.text[:200]}",
                    )
                data = resp.json()
            except ToolExecutionError:
                raise
            except Exception as exc:
                raise ToolExecutionError(
                    self.name, "NetworkError", f"Serper API request failed: {exc}"
                )

        organic = data.get("organic", [])
        results: List[Dict[str, str]] = []
        for item in organic[:num_results]:
            results.append(
                {
                    "title": item.get("title", ""),
                    "url": item.get("link", ""),
                    "snippet": item.get("snippet", ""),
                    "date": item.get("date", "Recent"),
                }
            )

        if not results:
            raise ToolExecutionError(
                self.name,
                "EmptyResultSetError",
                f"Serper returned 0 results for query: '{query}'",
            )

        return {
            "provider": "serper_google_search_api",
            "query": query,
            "result_count": len(results),
            "knowledge_graph": data.get("knowledgeGraph", {}),
            "results": results,
        }

    async def _live_public_search_fallback(
        self, query: str, num_results: int
    ) -> Dict[str, Any]:
        """Live keyless fallback using Wikipedia REST Search + HackerNews Algolia API."""
        results: List[Dict[str, str]] = []
        kw_query = extract_core_keywords(query, max_words=4)

        async with httpx.AsyncClient(
            timeout=10.0,
            headers={"User-Agent": "AgenticAIInternSystem/1.0 (Educational Research)"},
        ) as client:
            # 1. Query HackerNews Algolia Search API for tech/news developments
            for q_variant in [kw_query, extract_core_keywords(query, max_words=2)]:
                if len(results) >= num_results:
                    break
                try:
                    hn_resp = await client.get(
                        "https://hn.algolia.com/api/v1/search",
                        params={"query": q_variant, "tags": "story", "hitsPerPage": num_results},
                    )
                    if hn_resp.status_code == 200:
                        hits = hn_resp.json().get("hits", [])
                        for h in hits:
                            title = h.get("title")
                            url = h.get("url") or f"https://news.ycombinator.com/item?id={h.get('objectID')}"
                            if title and len(results) < num_results:
                                results.append(
                                    {
                                        "title": title,
                                        "url": url,
                                        "snippet": (
                                            f"Live industry report & technical discussion on '{title}' "
                                            f"(Community score: {h.get('points', 0)}, Comments: {h.get('num_comments', 0)})."
                                        ),
                                        "date": (h.get("created_at") or "")[:10],
                                    }
                                )
                except Exception:
                    pass

            # 2. Query Wikipedia OpenSearch API
            for q_variant in [kw_query, extract_core_keywords(query, max_words=2)]:
                if len(results) >= num_results:
                    break
                try:
                    wiki_resp = await client.get(
                        "https://en.wikipedia.org/w/api.php",
                        params={
                            "action": "query",
                            "list": "search",
                            "srsearch": q_variant,
                            "format": "json",
                            "srlimit": num_results,
                        },
                    )
                    if wiki_resp.status_code == 200:
                        wiki_items = wiki_resp.json().get("query", {}).get("search", [])
                        for w in wiki_items:
                            title = w.get("title", "")
                            snippet = (
                                w.get("snippet", "")
                                .replace('<span class="searchmatch">', "")
                                .replace("</span>", "")
                            )
                            page_url = f"https://en.wikipedia.org/wiki/{title.replace(' ', '_')}"
                            if len(results) < num_results:
                                results.append(
                                    {
                                        "title": f"{title} — Reference Overview",
                                        "url": page_url,
                                        "snippet": snippet,
                                        "date": w.get("timestamp", "")[:10],
                                    }
                                )
                except Exception:
                    pass

        if not results:
            raise ToolExecutionError(
                self.name,
                "EmptyResultSetError",
                f"No live search results found for query '{query}'.",
            )

        return {
            "provider": "live_public_search_api (Algolia HN + Wikipedia)",
            "query": query,
            "effective_keywords": kw_query,
            "result_count": len(results),
            "results": results[:num_results],
        }
