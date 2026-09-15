# =============================================================================
# research_engine.py
# Triple-Layer Evidence Gatherer:
#   Layer 1: Perplexity sonar-pro (broad semantic web search)
#   Layer 2: SerpAPI Google AI Mode (cross-verification via Google's AI)
#   Layer 3: Agent Reach / Jina Reader (deep page crawling)
# =============================================================================

import os
import sys
import logging
import requests
from typing import Dict, Any, List, Optional

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Agent Reach Integration (Jina Reader channel)
# ---------------------------------------------------------------------------
_AGENT_REACH_PATH = os.path.join(os.path.dirname(__file__), "Agent-Reach")
if os.path.isdir(_AGENT_REACH_PATH) and _AGENT_REACH_PATH not in sys.path:
    sys.path.insert(0, _AGENT_REACH_PATH)

try:
    from agent_reach.channels.web import WebChannel
    _WEB_CHANNEL = WebChannel()
    _AGENT_REACH_AVAILABLE = True
    logger.info("Agent Reach WebChannel (Jina Reader) loaded successfully.")
except ImportError:
    _WEB_CHANNEL = None
    _AGENT_REACH_AVAILABLE = False
    logger.warning(
        "Agent Reach not found. Falling back to direct Jina Reader HTTP calls."
    )

# ---------------------------------------------------------------------------
# SerpAPI Integration (Google AI Mode)
# ---------------------------------------------------------------------------
try:
    import serpapi
    _SERPAPI_AVAILABLE = True
except ImportError:
    _SERPAPI_AVAILABLE = False
    logger.warning("serpapi package not installed. Google AI Mode layer disabled.")


try:
    from tavily import TavilyClient
    _TAVILY_AVAILABLE = True
except ImportError:
    _TAVILY_AVAILABLE = False
    logger.warning("tavily-python package not installed. Path B will fail if missing.")


class HybridResearcher:
    """
    Triple-layer verification engine for executive due diligence.

    Layer 1 (Macro): Perplexity sonar-pro performs broad semantic web search,
                     returning a natural-language summary and citation URLs.
    Layer 2 (Cross): SerpAPI Google AI Mode provides an independent AI-grounded
                     verification pass using Google's own knowledge synthesis,
                     with additional source URLs.
    Layer 3 (Micro): Agent Reach / Jina Reader deep-crawls the top citation
                     URLs and returns the raw page text for granular checking.
    """

    PPLX_ENDPOINT = "https://api.perplexity.ai/chat/completions"
    JINA_PREFIX = "https://r.jina.ai/"
    MAX_DEEP_CHARS = 3000          # cap per-page text to control token cost
    MAX_CITATIONS_TO_CRAWL = 3     # deep-dive into top N urls

    def __init__(
        self,
        perplexity_api_key: Optional[str] = None,
        serpapi_api_key: Optional[str] = None,
        tavily_api_key: Optional[str] = None,
    ):
        self.pplx_key = perplexity_api_key or os.getenv("PERPLEXITY_API_KEY", "")
        self.serpapi_key = serpapi_api_key or os.getenv("SERPAPI_API_KEY", "")
        self.tavily_key = tavily_api_key or os.getenv("TAVILY_API_KEY", "")
        self.use_perplexity = os.getenv("USE_PERPLEXITY", "true").lower() == "true"

        if not self.pplx_key or not self.use_perplexity:
            logger.warning("Perplexity layer disabled (API key missing or USE_PERPLEXITY=false).")
        if not self.serpapi_key:
            logger.warning("SERPAPI_API_KEY not set. Layer 2 (Google AI Mode) will be skipped.")
        if not self.tavily_key:
            logger.warning("TAVILY_API_KEY not set. Path B (Independent Search) will fail.")

    # ------------------------------------------------------------------
    # Layer: Tavily (Independent Search Path B)
    # ------------------------------------------------------------------
    def run_tavily_search(
        self, claim: str, executive_name: str
    ) -> Dict[str, Any]:
        """
        Independent Path B search using Tavily.
        """
        if not self.tavily_key or not _TAVILY_AVAILABLE:
            return {"text": "Tavily layer disabled or missing key/package.", "citations": []}
            
        try:
            client = TavilyClient(api_key=self.tavily_key)
            response = client.search(
                query=f"{executive_name} {claim}",
                search_depth="advanced",
                include_answer=True,
                max_results=5,
            )
            
            # Extract citations from results
            citations = [res.get("url") for res in response.get("results", []) if res.get("url")]
            text_answer = response.get("answer", "No synthesized answer returned from Tavily.")
            
            return {"text": text_answer, "citations": citations}
        except Exception as exc:
            logger.error("Tavily request failed: %s", exc)
            return {"text": f"Search failed: {exc}", "citations": []}

    # ------------------------------------------------------------------
    # Layer 1: Perplexity sonar-pro
    # ------------------------------------------------------------------
    def run_perplexity_search(
        self, claim: str, executive_name: str
    ) -> Dict[str, Any]:
        """
        Broad web-grounded research via Perplexity sonar-pro.
        Returns: { "text": str, "citations": List[str] }
        """
        if not self.pplx_key or not self.use_perplexity:
            return {"text": "Perplexity layer disabled.", "citations": []}
            
        headers = {
            "Authorization": f"Bearer {self.pplx_key}",
            "Content-Type": "application/json",
        }

        system_prompt = (
            "You are a strict due-diligence investigator. "
            "Verify whether the following claim about the named executive "
            "is supported by primary sources (regulatory filings, press "
            "releases, credible news outlets, company registries). "
            "Return your factual findings and cite ALL source URLs. "
            "If you cannot find any evidence, state that explicitly."
        )

        user_content = (
            f"Executive: {executive_name}\n"
            f"Claim to verify: {claim}\n\n"
            "Provide evidence and source URLs."
        )

        payload = {
            "model": "sonar",
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_content},
            ],
            "temperature": 0.1,
        }

        try:
            resp = requests.post(
                self.PPLX_ENDPOINT,
                json=payload,
                headers=headers,
                timeout=45,
            )
            resp.raise_for_status()
            data = resp.json()
            content = data["choices"][0]["message"]["content"]
            citations = data.get("citations", [])
            return {"text": content, "citations": citations}
        except requests.exceptions.Timeout:
            logger.error("Perplexity request timed out for claim: %s", claim)
            return {"text": "Search timed out.", "citations": []}
        except requests.exceptions.RequestException as exc:
            logger.error("Perplexity request failed: %s", exc)
            return {"text": f"Search failed: {exc}", "citations": []}

    # ------------------------------------------------------------------
    # Layer 2: SerpAPI Google AI Mode
    # ------------------------------------------------------------------
    def run_google_ai_mode(
        self, claim: str, executive_name: str
    ) -> Dict[str, Any]:
        """
        Cross-verification via Google AI Mode (SerpAPI).
        Returns: {
            "ai_answer": str,        # Google AI synthesized response
            "citations": List[str],  # Source URLs from Google AI
            "organic_urls": List[str] # Top organic search result URLs
        }
        """
        if not _SERPAPI_AVAILABLE or not self.serpapi_key:
            logger.info("SerpAPI not available. Skipping Google AI Mode layer.")
            return {"ai_answer": "", "citations": [], "organic_urls": []}

        try:
            client = serpapi.Client(api_key=self.serpapi_key)
            query = f"Verify: {executive_name} {claim}"

            results = client.search({
                "engine": "google_ai_mode",
                "q": query,
                "hl": "en",
                "gl": "ae",               # UAE-focused for DIFC context
                "google_domain": "google.com",
            })

            ai_answer = ""
            ai_citations = []

            # google_ai_mode returns text_blocks and references at top level
            text_blocks = results.get("text_blocks", [])
            if text_blocks:
                ai_parts = []
                for block in text_blocks:
                    snippet = block.get("snippet", "")
                    if snippet:
                        ai_parts.append(snippet)
                ai_answer = " ".join(ai_parts)

            for ref in results.get("references", []):
                link = ref.get("link", "")
                if link:
                    ai_citations.append(link)

            # Fallback: use reconstructed_markdown if text_blocks empty
            if not ai_answer:
                ai_answer = results.get("reconstructed_markdown", "")

            organic_urls = []
            for result in results.get("organic_results", [])[:5]:
                link = result.get("link", "")
                if link:
                    organic_urls.append(link)

            logger.info(
                "Google AI Mode returned %d chars, %d citations, %d organic URLs.",
                len(ai_answer), len(ai_citations), len(organic_urls),
            )

            return {
                "ai_answer": ai_answer,
                "citations": ai_citations,
                "organic_urls": organic_urls,
            }

        except Exception as exc:
            logger.error("SerpAPI Google AI Mode failed: %s", exc)
            return {"ai_answer": "", "citations": [], "organic_urls": []}

    # ------------------------------------------------------------------
    # Layer 2B: SerpAPI for general person research (profile enrichment)
    # ------------------------------------------------------------------
    def research_person(self, executive_name: str, company: str = "") -> Dict[str, Any]:
        """
        Use Google AI Mode to gather general information about a person.
        Useful for initial profile enrichment beyond LinkedIn.
        Returns: {
            "ai_summary": str,
            "citations": List[str],
            "organic_urls": List[str]
        }
        """
        if not _SERPAPI_AVAILABLE or not self.serpapi_key:
            return {"ai_summary": "", "citations": [], "organic_urls": []}

        try:
            client = serpapi.Client(api_key=self.serpapi_key)

            query = f"Who is {executive_name} research in deep and extract deep details"
            if company:
                query += f" {company}"
            query += "? Background, career, achievements, and public record. -site:linkedin.com"

            results = client.search({
                "engine": "google_ai_mode",
                "q": query,
                "hl": "en",
                "gl": "ae",
                "google_domain": "google.com",
            })

            ai_summary = ""
            citations = []

            # google_ai_mode returns text_blocks and references at top level
            text_blocks = results.get("text_blocks", [])
            if text_blocks:
                parts = []
                for block in text_blocks:
                    snippet = block.get("snippet", "")
                    if snippet:
                        parts.append(snippet)
                ai_summary = " ".join(parts)

            for ref in results.get("references", []):
                link = ref.get("link", "")
                if link:
                    citations.append(link)

            # Fallback: use reconstructed_markdown if text_blocks empty
            if not ai_summary:
                ai_summary = results.get("reconstructed_markdown", "")
                
            # Clean summary: remove trailing AI questions like "Would you like to explore..."
            import re
            import html
            ai_summary = re.sub(r'Would you like to.*?(?:\?|$)', '', ai_summary, flags=re.IGNORECASE).strip()
            ai_summary = html.escape(ai_summary)
            
            # Append citations cleanly
            if citations:
                links_html = "<br><br><strong>Sources:</strong><ul style='margin-top:4px; padding-left:16px;'>"
                for ref in results.get("references", []):
                    title = ref.get("title", "Source")
                    link = ref.get("link", "")
                    if link:
                        links_html += f"<li><a href='{link}' target='_blank' style='color:var(--li-blue);text-decoration:none;'>{title}</a></li>"
                links_html += "</ul>"
                ai_summary += links_html

            organic_urls = []
            for result in results.get("organic_results", [])[:5]:
                link = result.get("link", "")
                if link:
                    organic_urls.append(link)

            logger.info(
                "Person research returned %d chars for %s.",
                len(ai_summary), executive_name,
            )

            return {
                "ai_summary": ai_summary,
                "citations": citations,
                "organic_urls": organic_urls,
            }

        except Exception as exc:
            logger.error("SerpAPI person research failed: %s", exc)
            return {"ai_summary": "", "citations": [], "organic_urls": []}

    # ------------------------------------------------------------------
    # Layer 3: Deep page crawl via Agent Reach / Jina Reader
    # ------------------------------------------------------------------
    def _read_via_agent_reach(self, url: str) -> str:
        """Use Agent Reach WebChannel (Jina Reader) if available."""
        if _AGENT_REACH_AVAILABLE and _WEB_CHANNEL:
            try:
                full_text = _WEB_CHANNEL.read(url)
                return full_text[: self.MAX_DEEP_CHARS]
            except Exception as exc:
                logger.warning(
                    "Agent Reach read failed for %s: %s. Falling back to direct HTTP.",
                    url, exc,
                )
        return ""

    def _read_via_jina_direct(self, url: str) -> str:
        """Fallback: direct Jina Reader HTTP call."""
        try:
            jina_url = f"{self.JINA_PREFIX}{url}"
            resp = requests.get(
                jina_url,
                headers={
                    "User-Agent": "Mozilla/5.0",
                    "Accept": "text/plain",
                },
                timeout=20,
            )
            if resp.status_code == 200:
                return resp.text[: self.MAX_DEEP_CHARS]
        except Exception as exc:
            logger.warning("Direct Jina Reader failed for %s: %s", url, exc)
        return ""

    def deep_crawl_url(self, url: str) -> str:
        """
        Deep-crawl a single URL. Tries Agent Reach first, falls back
        to direct Jina Reader HTTP.
        """
        text = self._read_via_agent_reach(url)
        if not text:
            text = self._read_via_jina_direct(url)
        return text

    def verify_claim(
        self, claim: str, executive_name: str
    ) -> Dict[str, Any]:
        """
        Dual-source corroboration pipeline.

        Path A: SerpAPI Google AI Mode (primary AI fact-checker)
        Path B: Tavily (independent search with different index)
        Path C: Perplexity (optional 3rd independent check)
        Deep Crawl: Agent Reach crawls ONLY Path B URLs that are NOT in Path A,
                    guaranteeing truly independent source inspection.

        Returns:
            {
                "summary": str,
                "citations": List[str],
                "google_ai_answer": str,
                "google_ai_citations": List[str],
                "perplexity_answer": str,
                "perplexity_citations": List[str],
                "tavily_answer": str,
                "tavily_citations": List[str],
                "deep_evidence": List[dict],
                "source_pools": {
                    "pool_a": List[str],   # Google AI Mode sources
                    "pool_b": List[str],   # Tavily + Agent Reach sources
                }
            }
        """
        # ---- Path A: Google AI Mode (primary fact-checker) ----
        google_result = self.run_google_ai_mode(claim, executive_name)
        pool_a_urls = list(google_result["citations"] + google_result["organic_urls"])

        # ---- Path B: Tavily (independent search engine) ----
        tavily_result = self.run_tavily_search(claim, executive_name)
        pool_b_urls_raw = list(tavily_result["citations"])

        # KEY DESIGN: Exclude Path A URLs from Path B to guarantee independence
        pool_a_domains = set()
        for u in pool_a_urls:
            try:
                from urllib.parse import urlparse
                pool_a_domains.add(urlparse(u).netloc.lower())
            except Exception:
                pass

        independent_urls = []
        for u in pool_b_urls_raw:
            try:
                from urllib.parse import urlparse
                domain = urlparse(u).netloc.lower()
                if domain not in pool_a_domains and "linkedin.com" not in domain:
                    independent_urls.append(u)
            except Exception:
                if u not in pool_a_urls and "linkedin.com" not in u.lower():
                    independent_urls.append(u)

        # ---- Deep Crawl: Only independent (Path B) URLs via Agent Reach ----
        deep_evidence: List[Dict[str, str]] = []
        urls_to_crawl = independent_urls[: self.MAX_CITATIONS_TO_CRAWL]

        for url in urls_to_crawl:
            page_text = self.deep_crawl_url(url)
            if page_text:
                deep_evidence.append({"url": url, "text": page_text})

        # ---- Path C: Perplexity (Optional) ----
        pplx_result = {"text": "", "citations": []}
        if self.use_perplexity:
            pplx_result = self.run_perplexity_search(claim, executive_name)

        # Merge all unique citation URLs for reference
        all_citations = list(pool_a_urls)
        for url in pool_b_urls_raw + pplx_result["citations"]:
            if url not in all_citations:
                all_citations.append(url)

        pool_b_final = list(set(independent_urls + pool_b_urls_raw))

        logger.info(
            "Dual-source verification: Pool A has %d sources, Pool B has %d sources (%d independent deep-crawled).",
            len(pool_a_urls), len(pool_b_final), len(deep_evidence),
        )

        return {
            "summary": tavily_result["text"] + "\n\n" + pplx_result["text"],
            "citations": all_citations,
            "google_ai_answer": google_result["ai_answer"],
            "google_ai_citations": google_result["citations"],
            "tavily_answer": tavily_result["text"],
            "tavily_citations": tavily_result["citations"],
            "perplexity_answer": pplx_result["text"],
            "perplexity_citations": pplx_result["citations"],
            "deep_evidence": deep_evidence,
            "source_pools": {
                "pool_a": pool_a_urls,
                "pool_b": pool_b_final,
            },
        }
