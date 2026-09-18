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
import re
from urllib.parse import urlparse
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

    PPLX_ENDPOINT = "https://api.aicredits.com/v1/chat/completions"
    JINA_PREFIX = "https://r.jina.ai/"
    MAX_DEEP_CHARS = 3000          # cap per-page text to control token cost
    MAX_CITATIONS_TO_CRAWL = 3     # deep-dive into top N urls
    PR_DOMAINS = {"prnewswire.com", "prweb.com", "globenewswire.com", "businesswire.com", "uniindia.com"}

    def __init__(
        self,
        perplexity_api_key: Optional[str] = None,
        serpapi_api_key: Optional[str] = None,
        tavily_api_key: Optional[str] = None,
    ):
        self.pplx_key = perplexity_api_key or os.getenv("ai_credit", "")
        self.serpapi_key = serpapi_api_key or os.getenv("SERPAPI_API_KEY", "")
        self.tavily_key = tavily_api_key or os.getenv("TAVILY_API_KEY", "")
        self.use_perplexity = os.getenv("USE_AICREDITS", "true").lower() == "true"

        if not self.pplx_key or not self.use_perplexity:
            logger.warning("Perplexity layer disabled (API key missing or USE_PERPLEXITY=false).")
        if not self.serpapi_key:
            logger.warning("SERPAPI_API_KEY not set. Layer 2 (Google AI Mode) will be skipped.")
        if not self.tavily_key:
            logger.warning("TAVILY_API_KEY not set. Path B (Independent Search) will fail.")

    def edgar_fulltext(self, query: str, forms: str = "10-Q,10-K,8-K,S-1") -> Optional[Dict[str, Any]]:
        try:
            r = requests.get(
                "https://efts.sec.gov/LATEST/search-index",
                params={"q": f'"{query}"', "forms": forms},
                headers={"User-Agent": "NoCap.ai Research contact@nocap.ai"},
                timeout=15
            )
            if r.status_code == 200:
                return r.json()
        except Exception as exc:
            logger.error("EDGAR request failed: %s", exc)
        return None

    def build_queries(self, claim: dict) -> list[str]:
        qs = [claim.get("text", "")]
        cparty = claim.get("counterparty")
        subj = claim.get("subject", "")
        val = claim.get("value")
        unit = claim.get("unit")
        
        if cparty:
            qs.append(f"{cparty} {subj} acquisition purchase price")
            qs.append(f"{cparty} SEC filing {subj}")
        if val:
            qs.append(f"{subj} {unit} actual amount reported")
        qs.append(f"{claim.get('text', '')} disputed OR corrected OR actually")
        return qs

    def shingles(self, text: str, n: int = 8) -> set[str]:
        words = re.findall(r"\w+", text.lower())
        return {" ".join(words[i:i+n]) for i in range(max(1, len(words)-n+1))}

    def same_origin(self, a: str, b: str, threshold: float = 0.25) -> bool:
        sa, sb = self.shingles(a), self.shingles(b)
        if not sa or not sb:
            return False
        return len(sa & sb) / max(1, min(len(sa), len(sb))) > threshold

    def origin_tag(self, cluster: list, subject_domains: set[str]) -> str:
        for c in cluster:
            url = c.get("url", "")
            host = urlparse(url).netloc.removeprefix("www.")
            if host in self.PR_DOMAINS or host in subject_domains:
                return "self_originating"
        return "independent"

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
            return {"text": "Tavily layer disabled or missing key/package.", "citations": [], "content_map": {}}
            
        try:
            client = TavilyClient(api_key=self.tavily_key)
            response = client.search(
                query=f"{executive_name} {claim}",
                search_depth="advanced",
                include_answer=True,
                include_raw_content=True,
                max_results=5,
            )
            
            citations = []
            content_map = {}
            for res in response.get("results", []):
                url = res.get("url")
                if url:
                    citations.append(url)
                    # Use raw_content if available, otherwise content
                    content = res.get("raw_content") or res.get("content") or ""
                    if content:
                        content_map[url] = content[:self.MAX_DEEP_CHARS]

            text_answer = response.get("answer", "No synthesized answer returned from Tavily.")
            
            return {"text": text_answer, "citations": citations, "content_map": content_map}
        except Exception as exc:
            logger.error("Tavily request failed: %s", exc)
            return {"text": f"Search failed: {exc}", "citations": [], "content_map": {}}

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
    # Layer 1B: Perplexity BATCH fact-check (fallback for weak claims)
    # ------------------------------------------------------------------
    def batch_fact_check(
        self, executive_name: str, claims: List[dict]
    ) -> Dict[str, Any]:
        """
        Send ALL weak (unverified/contradicted) claims to Perplexity in ONE
        call and ask for a per-claim verdict + source links.

        `claims` is a list of {"id": str, "text": str}. Returns:
          {
            "<claim_id>": {"verdict": str, "confidence": str,
                            "reasoning": str, "sources": [urls...]},
            ...,
            "_citations": [all urls returned],
          }
        Returns {} if Perplexity is disabled or the call fails.
        """
        # The batch fallback only needs a key — it is invoked explicitly by the
        # graph's perplexity_fallback node (which owns the enable/disable flag),
        # so it does NOT gate on self.use_perplexity (that flag controls the
        # per-claim inline Perplexity during evidence gathering).
        if not self.pplx_key:
            logger.info("[PerplexityBatch] disabled (no PERPLEXITY_API_KEY).")
            return {}
        if not claims:
            return {}

        headers = {
            "Authorization": f"Bearer {self.pplx_key}",
            "Content-Type": "application/json",
        }

        numbered = "\n".join(
            f'{i+1}. [id={c.get("id")}] {c.get("text","")}'
            for i, c in enumerate(claims)
        )

        system_prompt = (
            "You are a strict due-diligence fact-checker. For EACH numbered claim "
            "about the named executive, search the web and decide a verdict using "
            "primary/credible sources (regulatory filings, company registries, "
            "named news outlets). Verdicts: Verified, Partially Verified, "
            "Unverified, Contradicted. Be DECISIVE — do not default everything to "
            "Partially Verified.\n"
            "- Verified: TWO OR MORE independent credible sources (not copies of "
            "each other) confirm the claim, including its number/date. Use this "
            "confidently for well-documented facts.\n"
            "- Partially Verified: the substance holds but a specific detail is off, "
            "OR the only support is the subject's own site/press release/one origin. "
            "Publishable WITH attribution.\n"
            "- Contradicted: a credible source gives a DIFFERENT number, date, age, "
            "or fact. If the claimed figure/age is wrong (e.g. wrong age, wrong "
            "acquisition price), you MUST mark it Contradicted and state the correct "
            "value in the reasoning. Do NOT mark it Verified or Partially Verified.\n"
            "- Unverified: no credible evidence found.\n\n"
            "Return ONLY a JSON object of this exact shape:\n"
            '{"results":[{"id":"<claim id>","verdict":"...","confidence":"high|medium|low",'
            '"reasoning":"one or two sentences naming the sources","sources":["url1","url2"]}]}\n'
            "Include a real source URL for every non-Unverified verdict."
        )
        user_content = (
            f"Executive: {executive_name}\n\nClaims to fact-check:\n{numbered}\n\n"
            "Return the JSON object now."
        )

        payload = {
            "model": "sonar",
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_content},
            ],
            "temperature": 0.1,
        }

        logger.info("[PerplexityBatch] Fact-checking %d weak claims for %s in one call.",
                    len(claims), executive_name)
        try:
            resp = requests.post(self.PPLX_ENDPOINT, json=payload, headers=headers, timeout=90)
            resp.raise_for_status()
            data = resp.json()
            content = data["choices"][0]["message"]["content"] or ""
            citations = data.get("citations", []) or []
            logger.info("[PerplexityBatch] RETURNED %d chars, %d citations.",
                        len(content), len(citations))
            logger.info("[PerplexityBatch] ANSWER: %s", content[:1000])
            if citations:
                logger.info("[PerplexityBatch] CITATIONS: %s", " | ".join(citations[:12]))

            # Parse the per-claim JSON (best-effort, tolerant of fences/preamble).
            parsed = self._extract_json(content)
            out: Dict[str, Any] = {"_citations": citations}
            results = []
            if isinstance(parsed, dict):
                results = parsed.get("results") or []
            elif isinstance(parsed, list):
                results = parsed
            for r in results:
                if not isinstance(r, dict):
                    continue
                cid = str(r.get("id", "")).strip()
                if not cid:
                    continue
                out[cid] = {
                    "verdict": r.get("verdict", "Unverified"),
                    "confidence": r.get("confidence", "low"),
                    "reasoning": r.get("reasoning", ""),
                    "sources": r.get("sources", []) or [],
                }
            logger.info("[PerplexityBatch] Parsed verdicts for %d claims.",
                        len([k for k in out if k != "_citations"]))
            return out
        except Exception as exc:
            logger.error("[PerplexityBatch] failed: %s", exc)
            return {}

    @staticmethod
    def _extract_json(text: str) -> Any:
        """Tolerant JSON extraction from a model response (fences/preamble)."""
        import json as _json
        if not text:
            return None
        t = text.strip()
        if "```json" in t:
            t = t.split("```json", 1)[1].split("```", 1)[0].strip()
        elif "```" in t:
            parts = t.split("```")
            if len(parts) >= 3:
                t = parts[1].strip()
        try:
            return _json.loads(t)
        except Exception:
            pass
        s, e = t.find("{"), t.rfind("}")
        if s != -1 and e > s:
            try:
                return _json.loads(t[s:e + 1])
            except Exception:
                return None
        return None

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
            query = f"Verify: {executive_name} {claim}"
            client = serpapi.Client(api_key=self.serpapi_key)

            logger.info("[GoogleAIMode] QUERY: %s", query)

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
                "[GoogleAIMode] RETURNED %d chars, %d citations, %d organic URLs.",
                len(ai_answer), len(ai_citations), len(organic_urls),
            )
            # Deep logging: show exactly what Google AI Mode said and which
            # links it grounded the answer on.
            if ai_answer:
                logger.info("[GoogleAIMode] ANSWER: %s", ai_answer[:800])
            if ai_citations:
                logger.info("[GoogleAIMode] CITATIONS: %s", " | ".join(ai_citations[:8]))
            if organic_urls:
                logger.info("[GoogleAIMode] ORGANIC: %s", " | ".join(organic_urls[:8]))

            return {
                "ai_answer": ai_answer,
                "citations": ai_citations,
                "organic_urls": organic_urls,
            }

        except Exception as exc:
            logger.error("SerpAPI Google AI Mode failed for query %r: %s", query, exc)
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
            query = f"Provide a detailed and comprehensive biography of {executive_name}"
            if company:
                query += f", primarily known for {company}"
            query += ". Describe their full career history, major professional achievements, public reputation, controversies, and key business milestones in detail. -site:linkedin.com"

            client = serpapi.Client(api_key=self.serpapi_key)

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
        self, claim: dict, executive_name: str
    ) -> Dict[str, Any]:
        """
        Gather evidence using multiple query shapes.
        """
        queries = self.build_queries(claim)
        
        all_citations = []
        google_ai_answers = []
        tavily_answers = []
        tavily_content_cache = {}
        pool_a_urls = []
        pool_b_urls_raw = []

        for q in queries:
            g_res = self.run_google_ai_mode(q, executive_name)
            pool_a_urls.extend(g_res["citations"] + g_res["organic_urls"])
            if g_res["ai_answer"]:
                google_ai_answers.append(g_res["ai_answer"])
            
            t_res = self.run_tavily_search(q, executive_name)
            pool_b_urls_raw.extend(t_res["citations"])
            if t_res["text"]:
                tavily_answers.append(t_res["text"])
            if "content_map" in t_res:
                tavily_content_cache.update(t_res["content_map"])

        # De-duplicate URLs
        pool_a_urls = list(set(pool_a_urls))
        pool_b_urls_raw = list(set(pool_b_urls_raw))
        all_citations = list(set(pool_a_urls + pool_b_urls_raw))

        deep_evidence: List[Dict[str, str]] = []
        urls_to_crawl = all_citations[: self.MAX_CITATIONS_TO_CRAWL]

        for url in urls_to_crawl:
            if url in tavily_content_cache:
                deep_evidence.append({"url": url, "text": tavily_content_cache[url]})
            else:
                page_text = self.deep_crawl_url(url)
                if page_text:
                    deep_evidence.append({"url": url, "text": page_text})

        pplx_result = {"text": "", "citations": []}
        if self.use_perplexity:
            # use the primary confirm query for perplexity to save time
            pplx_result = self.run_perplexity_search(queries[0], executive_name)
            all_citations.extend(pplx_result["citations"])
            all_citations = list(set(all_citations))

        claim_txt = claim.get("text", "") if isinstance(claim, dict) else str(claim)
        logger.info(
            "[Evidence] claim=%r | %d queries | %d total URLs | %d deep-crawled",
            claim_txt[:80], len(queries), len(all_citations), len(deep_evidence)
        )
        if all_citations:
            logger.info("[Evidence] URLs: %s", " | ".join(all_citations[:10]))

        return {
            "summary": "\n".join(tavily_answers) + "\n\n" + pplx_result["text"],
            "citations": all_citations,
            "google_ai_answer": "\n".join(google_ai_answers),
            "google_ai_citations": pool_a_urls,
            "tavily_answer": "\n".join(tavily_answers),
            "tavily_citations": pool_b_urls_raw,
            "perplexity_answer": pplx_result["text"],
            "perplexity_citations": pplx_result["citations"],
            "deep_evidence": deep_evidence,
            "queries_run": queries,
            "source_pools": {
                "pool_a": pool_a_urls,
                "pool_b": pool_b_urls_raw,
            },
        }
