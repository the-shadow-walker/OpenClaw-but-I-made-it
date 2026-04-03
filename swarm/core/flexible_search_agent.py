"""
Flexible Search Agent - Supports Multiple Search Backends
Priority: SearXNG > DuckDuckGo > Google > Fallback

Key advantages of SearXNG:
- Self-hosted, no bot detection
- Aggregates multiple search engines
- Privacy-focused
- No rate limits
- JSON API
"""

import requests
import json
from typing import List, Dict, Optional
from dataclasses import dataclass
from urllib.parse import quote_plus, urlparse
from datetime import datetime
import re

# Valid named time filters
_NAMED_FILTERS = {"day", "week", "month", "year"}

# DuckDuckGo df param mapping
_DDG_DF = {"day": "d", "week": "w", "month": "m", "year": "y"}


@dataclass
class SearchResult:
    """Search result with source"""
    title: str
    url: str
    snippet: str
    content: Optional[str] = None
    source: str = "search"


class FlexibleSearchAgent:
    """
    Multi-backend search agent with SearXNG priority.
    
    Backends (in priority order):
    1. SearXNG (if configured) - BEST for avoiding bot detection
    2. DuckDuckGo HTML
    3. Google (last resort)
    
    Features:
    - Automatic fallback
    - r.jina.ai for content extraction
    - Source tracking
    """
    
    def __init__(
        self,
        searxng_url: Optional[str] = None,
        timeout: int = 30,
        max_results: int = 5,
        prefer_searxng: bool = True,
        date_filter: Optional[str] = None,
    ):
        """
        Initialize search agent.

        Args:
            searxng_url: URL of SearXNG instance (e.g., "http://localhost:8888")
            timeout: Request timeout in seconds
            max_results: Maximum results to return
            prefer_searxng: Try SearXNG first if available
            date_filter: Restrict results by recency.
                Named periods : "day" | "week" | "month" | "year"
                Specific floor: "YYYY-MM-DD"  (results on/after this date)
                None           : no filter (default, all time)
        """
        self.searxng_url = searxng_url
        self.timeout = timeout
        self.max_results = max_results
        self.prefer_searxng = prefer_searxng

        self.current_date: str = datetime.now().strftime("%Y-%m-%d")
        self.date_filter: Optional[str] = date_filter

        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Research Bot) AppleWebKit/537.36'
        })

        if self.date_filter:
            print(f"🗓️  Date filter: {self.date_filter}  (today: {self.current_date})")

        # Test SearXNG connectivity
        if self.searxng_url:
            self._test_searxng()

    def set_date_filter(self, date_filter: Optional[str]) -> None:
        """
        Change the date filter at runtime.

        Args:
            date_filter: "day" | "week" | "month" | "year" | "YYYY-MM-DD" | None
        """
        self.date_filter = date_filter
        if date_filter:
            print(f"🗓️  Date filter updated: {self.date_filter}")
        else:
            print("🗓️  Date filter cleared (all time)")
    
    def _test_searxng(self):
        """Test if SearXNG is accessible"""
        try:
            response = requests.get(
                f"{self.searxng_url}/search",
                params={'q': 'test', 'format': 'json'},
                timeout=5
            )
            if response.status_code == 200:
                print(f"✓ SearXNG connected: {self.searxng_url}")
                return True
        except:
            pass
        
        print(f"⚠️ SearXNG not accessible at {self.searxng_url}")
        self.searxng_url = None
        return False
    
    def search(self, query: str) -> List[SearchResult]:
        """
        Search with automatic backend selection and fallback.
        
        Args:
            query: Search query
            
        Returns:
            List of SearchResult objects
        """
        print(f"🔍 Searching: {query}")
        
        # Try backends in priority order
        backends = []
        
        if self.prefer_searxng and self.searxng_url:
            backends.append(('SearXNG', self._search_searxng))
        
        backends.extend([
            ('DuckDuckGo', self._search_duckduckgo),
            ('Google', self._search_google)
        ])
        
        for backend_name, backend_func in backends:
            try:
                print(f"   Trying {backend_name}...")
                results = backend_func(query)
                
                if results:
                    print(f"   ✓ Found {len(results)} results via {backend_name}")
                    return results[:self.max_results]
                else:
                    print(f"   ⚠️ {backend_name} returned 0 results")
            
            except requests.exceptions.ConnectionError:
                print(f"   ✗ {backend_name} connection failed (no network?)")
            except Exception as e:
                print(f"   ✗ {backend_name} error: {e}")
        
        print(f"   ⚠️ All backends failed")
        return []

    def search_stream(self, query: str):
        """
        Generator — yields status + result dicts as each backend responds.

        Yields dicts with a "type" key:
          {"type": "status", "backend": "searxng",    "msg": "Trying SearXNG…"}
          {"type": "result", "index": 0, "title": …,  "url": …, "snippet": …, "source": …}
          {"type": "error",  "backend": "duckduckgo", "msg": "0 results"}
          {"type": "done",   "total": N}

        Stops after the first backend that returns ≥1 result (same fallback
        logic as search()).  Results are yielded one-by-one immediately after
        the backend HTTP call returns, so the caller can forward them live via
        SSE without waiting for all results.
        """
        backends = []
        if self.prefer_searxng and self.searxng_url:
            backends.append(('searxng',    'SearXNG',    self._search_searxng))
        backends.extend([
            ('duckduckgo', 'DuckDuckGo', self._search_duckduckgo),
            ('google',     'Google',     self._search_google),
        ])

        idx = 0
        for key, name, func in backends:
            yield {"type": "status", "backend": key, "msg": f"Trying {name}…"}
            try:
                results = func(query)
            except requests.exceptions.ConnectionError:
                yield {"type": "error", "backend": key, "msg": "connection refused"}
                continue
            except Exception as e:
                yield {"type": "error", "backend": key, "msg": str(e)}
                continue

            if not results:
                yield {"type": "error", "backend": key, "msg": "0 results"}
                continue

            for r in results[:self.max_results]:
                yield {
                    "type":    "result",
                    "index":   idx,
                    "title":   r.title,
                    "url":     r.url,
                    "snippet": r.snippet,
                    "source":  r.source,
                }
                idx += 1
            break  # stop on first successful backend

        yield {"type": "done", "total": idx}

    def _search_searxng(self, query: str) -> List[SearchResult]:
        """
        Search using SearXNG instance.
        
        SearXNG advantages:
        - No bot detection (self-hosted)
        - Aggregates multiple engines
        - JSON API (easy parsing)
        - No rate limits
        """
        if not self.searxng_url:
            return []
        
        try:
            # SearXNG JSON API
            params: Dict = {
                'q': query,
                'format': 'json',
                'categories': 'general',
            }
            # SearXNG supports time_range: day / week / month / year
            if self.date_filter in _NAMED_FILTERS:
                params['time_range'] = self.date_filter
            elif self.date_filter:
                # Specific date floor — append as search operator; SearXNG will
                # pass it through to underlying engines.
                params['q'] = f"{query} after:{self.date_filter}"

            response = self.session.get(
                f"{self.searxng_url}/search",
                params=params,
                timeout=self.timeout
            )
            response.raise_for_status()
            
            data = response.json()
            results = []
            
            for item in data.get('results', [])[:self.max_results]:
                results.append(SearchResult(
                    title=item.get('title', 'No title'),
                    url=item.get('url', ''),
                    snippet=item.get('content', ''),
                    source='searxng'
                ))
            
            return results
            
        except Exception as e:
            print(f"      SearXNG error: {e}")
            return []
    
    def _search_duckduckgo(self, query: str) -> List[SearchResult]:
        """Search using DuckDuckGo HTML interface"""
        try:
            search_query = query
            if self.date_filter and self.date_filter not in _NAMED_FILTERS:
                # Specific date floor — append after: operator
                search_query = f"{query} after:{self.date_filter}"

            ddg_params = f"q={quote_plus(search_query)}"
            if self.date_filter in _DDG_DF:
                ddg_params += f"&df={_DDG_DF[self.date_filter]}"

            url = f"https://html.duckduckgo.com/html/?{ddg_params}"

            response = self.session.get(url, timeout=self.timeout)
            response.raise_for_status()
            
            results = []
            
            # Parse DuckDuckGo HTML results
            # Look for result blocks
            result_pattern = r'<a.*?class="result__a".*?href="(.*?)".*?>(.*?)</a>.*?<a.*?class="result__snippet".*?>(.*?)</a>'
            
            matches = re.findall(result_pattern, response.text, re.DOTALL)
            
            for url, title, snippet in matches[:self.max_results]:
                # Clean HTML tags
                title = re.sub(r'<[^>]+>', '', title).strip()
                snippet = re.sub(r'<[^>]+>', '', snippet).strip()
                url = url.strip()
                
                if url and title:
                    results.append(SearchResult(
                        title=title,
                        url=url,
                        snippet=snippet,
                        source='duckduckgo'
                    ))
            
            return results
            
        except Exception as e:
            print(f"      DuckDuckGo error: {e}")
            return []
    
    def _search_google(self, query: str) -> List[SearchResult]:
        """Search using Google (last resort, high bot detection)"""
        try:
            search_query = query
            if self.date_filter and self.date_filter not in _NAMED_FILTERS:
                search_query = f"{query} after:{self.date_filter}"
            url = f"https://www.google.com/search?q={quote_plus(search_query)}"
            
            response = self.session.get(url, timeout=self.timeout)
            response.raise_for_status()
            
            # Very basic Google parsing (often blocked)
            results = []
            
            # This is fragile and Google changes it often
            # Better to use SearXNG which handles this
            
            return results
            
        except Exception as e:
            print(f"      Google error: {e}")
            return []
    
    def fetch_content(self, url: str, use_jina: bool = True) -> Optional[str]:
        """
        Fetch full content from URL.
        
        Args:
            url: URL to fetch
            use_jina: Use r.jina.ai for clean extraction
            
        Returns:
            Clean text content or None
        """
        try:
            if use_jina:
                # Use Jina Reader for clean extraction
                jina_url = f"https://r.jina.ai/{url}"
                response = self.session.get(jina_url, timeout=self.timeout)
                response.raise_for_status()
                return response.text
            else:
                # Direct fetch
                response = self.session.get(url, timeout=self.timeout)
                response.raise_for_status()
                
                # Basic text extraction
                text = response.text
                # Remove HTML tags
                text = re.sub(r'<script[^>]*>.*?</script>', '', text, flags=re.DOTALL)
                text = re.sub(r'<style[^>]*>.*?</style>', '', text, flags=re.DOTALL)
                text = re.sub(r'<[^>]+>', ' ', text)
                text = re.sub(r'\s+', ' ', text)
                
                return text.strip()
        
        except Exception as e:
            print(f"      Fetch error for {url}: {e}")
            return None
    
    def search_and_fetch(
        self,
        query: str,
        num_sources: int = 3,
        fetch_content: bool = True
    ) -> List[SearchResult]:
        """
        Search and optionally fetch full content for results.
        
        Args:
            query: Search query
            num_sources: Number of sources to fetch content for
            fetch_content: Whether to fetch full content
            
        Returns:
            List of SearchResult objects with content
        """
        # Search
        results = self.search(query)
        
        if not results:
            return []
        
        # Fetch content for top results
        if fetch_content:
            for i, result in enumerate(results[:num_sources]):
                print(f"      Fetching content from: {result.title[:50]}...")
                content = self.fetch_content(result.url, use_jina=True)
                
                if content:
                    result.content = content[:5000]  # Limit to 5000 chars
                    print(f"         ✓ Got {len(result.content)} chars")
                else:
                    print(f"         ✗ Failed to fetch")
        
        return results


# Backward compatibility - alias to old name
JinaSearchAgent = FlexibleSearchAgent


# Quick test
if __name__ == "__main__":
    print("Testing Flexible Search Agent\n")
    
    # Test 1: Without SearXNG (will fallback to DuckDuckGo/Google)
    print("="*70)
    print("Test 1: Default (no SearXNG)")
    print("="*70)
    agent = FlexibleSearchAgent()
    results = agent.search("Python programming")
    print(f"\nResults: {len(results)}")
    for r in results[:3]:
        print(f"  - {r.title}")
        print(f"    {r.url}")
    
    # Test 2: With SearXNG (configure your instance)
    print("\n" + "="*70)
    print("Test 2: With SearXNG (if available)")
    print("="*70)
    
    # Example SearXNG URLs:
    # - http://localhost:8888 (local instance)
    # - https://searx.be (public instance)
    # - https://searx.tiekoetter.com (public instance)
    
    searxng_url = "http://localhost:8888"  # Change to your instance
    agent_searxng = FlexibleSearchAgent(searxng_url=searxng_url)
    
    results = agent_searxng.search("quantum computing")
    print(f"\nResults: {len(results)}")
    for r in results[:3]:
        print(f"  - {r.title}")
        print(f"    {r.url}")
    
    print("\n✓ Search agent ready!")
    print("\nTo use SearXNG:")
    print("1. Install: docker run -d -p 8888:8080 searxng/searxng")
    print("2. Configure: FlexibleSearchAgent(searxng_url='http://localhost:8888')")
