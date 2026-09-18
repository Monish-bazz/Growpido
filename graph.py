# =============================================================================
# graph.py
# LangGraph State Machine: Executive Reputation Diagnostic Pipeline
# Cyclic validation graph with HITL interrupt
# =============================================================================

import os
from typing import TypedDict, List, Annotated, Optional, Literal, Any, Union
from operator import add as _list_add
from enum import Enum
from pydantic import BaseModel, Field
import uuid
import re
import logging
import json
import time
import requests
import concurrent.futures

from dotenv import load_dotenv
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_core.runnables import RunnableConfig
from apify_client import ApifyClient
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver

from research_engine import HybridResearcher

load_dotenv()
logger = logging.getLogger(__name__)

# =============================================================================
# 1. Models & State Definition
# =============================================================================

class ClaimType(str, Enum):
    TRANSACTION = "transaction"       # M&A, funding, valuations
    RANKING = "ranking"           # "third largest", "top five"
    ROLE = "role"              # founder vs co-founder, titles, dates
    HEADCOUNT = "headcount"
    BIOGRAPHICAL = "biographical"      # birth, education
    SELF_MILESTONE = "self_milestone"    # "$1M at 18"
    AWARD = "award"

class Claim(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid.uuid4())[:8])
    text: str                  # atomic
    type: ClaimType
    subject: str
    counterparty: Optional[str] = None   # the other side of a transaction
    value: Optional[float] = None
    unit: Optional[str] = None           # USD, people, years
    as_of: Optional[str] = None          # date or age the value attaches to
    source_in_profile: str     # "about" | "experience" | "headline"

class VerdictType(str, Enum):
    VERIFIED = "Verified"
    PARTIALLY_VERIFIED = "Partially Verified"
    UNVERIFIED = "Unverified"
    CONTRADICTED = "Contradicted"

class Verdict(BaseModel):
    claim_id: str
    verdict: VerdictType
    confidence: str
    reasoning: str
    supporting_span: Optional[str] = None
    best_source_url: Optional[str] = None


class GraphState(TypedDict):
    executive_name: str
    linkedin_url: str
    raw_profile: dict
    profile_text: str                                   # flattened text for LLM
    google_person_context: str                          # SerpAPI person research
    claims: List[dict]                                  # atomic factual claims (dicts from Claim model)
    internal_contradictions: Annotated[List[dict], _list_add]
    verification_ledger: List[dict]  # verified / partial
    refusal_log: List[dict]          # unverified / contradicted
    evidence_cache: dict                                # gathered evidence per claim
    footprint_metrics: dict                             # footprint numbers
    gaps_analysis: List[dict]                            # positioning gaps
    approved: bool                                       # binary approval gate
    human_feedback: str                                 # optional HITL notes
    final_diagnostic: str                               # rendered markdown report


# =============================================================================
# 2. Shared Resources & LLM Provider
# =============================================================================

class OpenRouterRateLimitError(RuntimeError):
    """Raised when OpenRouter returns 401/402/403/429 (rate limit, credits, auth).

    Signals the fallback layer to switch to the secondary provider (NIM)
    immediately and stop trying OpenRouter for the rest of the run.
    """


class OpenRouterResponse:
    """Standardized response object matching LangChain invocation response."""
    def __init__(self, content: str, reasoning_details: Any = None, raw_data: Optional[dict] = None):
        self.content = content
        self.reasoning_details = reasoning_details
        self.raw_data = raw_data or {}

    def __str__(self) -> str:
        return self.content


class OpenRouterLLM:
    """
    OpenRouter API client compatible with LangGraph node invocations.
    Supports reasoning models such as nvidia/nemotron-3.5-lightning:free.
    """
    def __init__(
        self,
        model: Optional[str] = None,
        api_key: Optional[str] = None,
        temperature: float = 0.1,
        max_tokens: int = 2048,
        enable_reasoning: bool = False,
        timeout: int = 120,
        json_mode: bool = False,
    ):
        # Default to an instruction-following model that emits clean JSON.
        # (Reasoning models like nemotron-3.5-lightning dump chain-of-thought
        # into `content` and cannot be forced into strict JSON output.)
        self.model = model or os.getenv("OPENROUTER_MODEL", "z-ai/glm-5.2:free")
        self.api_key = api_key or os.getenv("OPENROUTER_API_KEY", "").strip()
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.enable_reasoning = enable_reasoning
        self.timeout = timeout
        # When True, ask OpenRouter to guarantee a valid JSON object response.
        self.json_mode = json_mode
        self.base_url = "https://openrouter.ai/api/v1/chat/completions"

    def invoke(self, messages: Any) -> OpenRouterResponse:
        formatted = []
        if isinstance(messages, str):
            formatted.append({"role": "user", "content": messages})
        elif isinstance(messages, list):
            for m in messages:
                if hasattr(m, "content"):
                    role = "user"
                    msg_type = getattr(m, "type", "").lower()
                    if msg_type in ("system", "systemmessage") or "system" in str(type(m)).lower():
                        role = "system"
                    elif msg_type in ("ai", "assistant", "aimessage") or "ai" in str(type(m)).lower():
                        role = "assistant"
                    formatted.append({"role": role, "content": str(m.content)})
                elif isinstance(m, dict):
                    formatted.append(m)
                else:
                    formatted.append({"role": "user", "content": str(m)})

        payload = {
            "model": self.model,
            "messages": formatted,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
        }
        if self.enable_reasoning:
            payload["reasoning"] = {"enabled": True}
        if self.json_mode:
            # Force a valid JSON object response. Kills the "Here's a thinking
            # process:" preamble that broke parsing on reasoning models.
            payload["response_format"] = {"type": "json_object"}

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": "http://localhost:8000",
            "X-Title": "Executive Reputation Diagnostic Engine",
        }

        max_retries = 2
        last_error = None
        for attempt in range(max_retries + 1):
            try:
                resp = requests.post(
                    self.base_url,
                    headers=headers,
                    json=payload,
                    timeout=self.timeout,
                )
                if resp.status_code == 200:
                    data = resp.json()
                    choice = data.get("choices", [{}])[0]
                    msg = choice.get("message", {})
                    content = msg.get("content", "") or ""
                    # Strip any inline thinking tags if emitted in content
                    if "<think>" in content and "</think>" in content:
                        content = content.split("</think>")[-1].strip()
                    reasoning_details = msg.get("reasoning_details") or msg.get("reasoning")
                    return OpenRouterResponse(content=content, reasoning_details=reasoning_details, raw_data=data)
                elif resp.status_code in (401, 402, 403, 429):
                    # Rate limited / out of credits / auth problem: retrying the
                    # same request will not help. Fail fast so the caller can
                    # fall back to the secondary (NIM) provider immediately
                    # instead of burning ~5s of retries per call.
                    raise OpenRouterRateLimitError(
                        f"HTTP {resp.status_code}: {resp.text[:200]}"
                    )
                else:
                    last_error = f"HTTP {resp.status_code}: {resp.text}"
                    logger.warning("OpenRouter API attempt %d failed: %s", attempt + 1, last_error)
            except OpenRouterRateLimitError:
                raise
            except Exception as e:
                last_error = str(e)
                logger.warning("OpenRouter API attempt %d error: %s", attempt + 1, last_error)
            time.sleep(1.0)

        raise RuntimeError(f"OpenRouter API request failed after {max_retries + 1} attempts: {last_error}")


class MorphLLM:
    """MorphLLM client (OpenAI-compatible). JSON mode disables reasoning
    automatically, giving clean, parseable JSON — ideal for the extract/judge
    nodes. Same .invoke(messages) interface as OpenRouterLLM."""

    def __init__(
        self,
        model: Optional[str] = None,
        api_key: Optional[str] = None,
        temperature: float = 0.1,
        max_tokens: int = 4096,
        timeout: int = 120,
        json_mode: bool = False,
    ):
        self.model = model or os.getenv("MORPH_MODEL", "morph-glm53-744b")
        self.api_key = api_key or os.getenv("MORPH_API_KEY", "").strip()
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.timeout = timeout
        self.json_mode = json_mode
        self.base_url = "https://api.morphllm.com/v1/chat/completions"

    def invoke(self, messages: Any) -> OpenRouterResponse:
        formatted = []
        if isinstance(messages, str):
            formatted.append({"role": "user", "content": messages})
        elif isinstance(messages, list):
            for m in messages:
                if hasattr(m, "content"):
                    role = "user"
                    msg_type = getattr(m, "type", "").lower()
                    if msg_type in ("system", "systemmessage") or "system" in str(type(m)).lower():
                        role = "system"
                    elif msg_type in ("ai", "assistant", "aimessage") or "ai" in str(type(m)).lower():
                        role = "assistant"
                    formatted.append({"role": role, "content": str(m.content)})
                elif isinstance(m, dict):
                    formatted.append(m)
                else:
                    formatted.append({"role": "user", "content": str(m)})

        payload = {
            "model": self.model,
            "messages": formatted,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
        }
        if self.json_mode:
            payload["response_format"] = {"type": "json_object"}

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

        max_retries = 2
        last_error = None
        for attempt in range(max_retries + 1):
            try:
                resp = requests.post(self.base_url, headers=headers, json=payload, timeout=self.timeout)
                if resp.status_code == 200:
                    data = resp.json()
                    msg = data.get("choices", [{}])[0].get("message", {})
                    content = msg.get("content", "") or ""
                    if "<think>" in content and "</think>" in content:
                        content = content.split("</think>")[-1].strip()
                    return OpenRouterResponse(content=content, raw_data=data)
                elif resp.status_code in (401, 403):
                    # Auth problem — retrying won't help; treat like OpenRouter
                    # so the fallback chain moves on.
                    raise OpenRouterRateLimitError(f"Morph HTTP {resp.status_code}: {resp.text[:200]}")
                elif resp.status_code == 429:
                    # Morph "service_overloaded" is transient — back off and retry.
                    last_error = f"HTTP 429: {resp.text[:160]}"
                    logger.warning("Morph attempt %d overloaded (429), backing off.", attempt + 1)
                    time.sleep(2.0 * (attempt + 1))
                    continue
                else:
                    last_error = f"HTTP {resp.status_code}: {resp.text[:200]}"
                    logger.warning("Morph attempt %d failed: %s", attempt + 1, last_error)
            except OpenRouterRateLimitError:
                raise
            except Exception as e:
                last_error = str(e)
                logger.warning("Morph attempt %d error: %s", attempt + 1, last_error)
            time.sleep(1.0)

        raise RuntimeError(f"Morph API request failed after {max_retries + 1} attempts: {last_error}")


class AICreditsLLM:
    """AI Credits gateway client (OpenAI-compatible, https://api.aicredits.in).
    Same .invoke(messages) interface as the other providers. Use a
    non-reasoning model (e.g. google/gemini-2.5-flash-lite) so JSON mode
    returns actual content rather than spending the token budget on hidden
    reasoning."""

    def __init__(
        self,
        model: Optional[str] = None,
        api_key: Optional[str] = None,
        temperature: float = 0.1,
        max_tokens: int = 4096,
        timeout: int = 120,
        json_mode: bool = False,
    ):
        self.model = model or os.getenv("AICREDITS_MODEL", "google/gemini-2.5-flash-lite")
        self.api_key = api_key or os.getenv("ai_credit", "").strip()
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.timeout = timeout
        self.json_mode = json_mode
        self.base_url = "https://api.aicredits.in/v1/chat/completions"

    def invoke(self, messages: Any) -> OpenRouterResponse:
        formatted = []
        if isinstance(messages, str):
            formatted.append({"role": "user", "content": messages})
        elif isinstance(messages, list):
            for m in messages:
                if hasattr(m, "content"):
                    role = "user"
                    msg_type = getattr(m, "type", "").lower()
                    if msg_type in ("system", "systemmessage") or "system" in str(type(m)).lower():
                        role = "system"
                    elif msg_type in ("ai", "assistant", "aimessage") or "ai" in str(type(m)).lower():
                        role = "assistant"
                    formatted.append({"role": role, "content": str(m.content)})
                elif isinstance(m, dict):
                    formatted.append(m)
                else:
                    formatted.append({"role": "user", "content": str(m)})

        payload = {
            "model": self.model,
            "messages": formatted,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
        }
        if self.json_mode:
            payload["response_format"] = {"type": "json_object"}

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

        max_retries = 2
        last_error = None
        for attempt in range(max_retries + 1):
            try:
                resp = requests.post(self.base_url, headers=headers, json=payload, timeout=self.timeout)
                if resp.status_code == 200:
                    data = resp.json()
                    msg = data.get("choices", [{}])[0].get("message", {})
                    content = msg.get("content", "") or ""
                    if "<think>" in content and "</think>" in content:
                        content = content.split("</think>")[-1].strip()
                    return OpenRouterResponse(content=content, raw_data=data)
                elif resp.status_code in (401, 402, 403):
                    # Auth / quota / forbidden — retrying won't help; cascade.
                    raise OpenRouterRateLimitError(f"AICredits HTTP {resp.status_code}: {resp.text[:200]}")
                elif resp.status_code == 429:
                    last_error = f"HTTP 429: {resp.text[:160]}"
                    logger.warning("AICredits attempt %d rate-limited (429), backing off.", attempt + 1)
                    time.sleep(2.0 * (attempt + 1))
                    continue
                else:
                    last_error = f"HTTP {resp.status_code}: {resp.text[:200]}"
                    logger.warning("AICredits attempt %d failed: %s", attempt + 1, last_error)
            except OpenRouterRateLimitError:
                raise
            except Exception as e:
                last_error = str(e)
                logger.warning("AICredits attempt %d error: %s", attempt + 1, last_error)
            time.sleep(1.0)

        raise RuntimeError(f"AICredits API request failed after {max_retries + 1} attempts: {last_error}")


# Process-wide latch: once OpenRouter reports rate-limit/credits exhaustion,
# every subsequent LLM call in this run skips OpenRouter and goes straight to
# NIM. This avoids re-hitting the daily-limit wall (and its retry delay) on
# every one of the ~20 claims. Reset only on process restart.
_OPENROUTER_DISABLED = False


# Live per-claim judging progress, keyed by thread_id. LangGraph only commits
# state when a node returns, so the judge node can't surface incremental
# verdicts through graph state. This side-channel lets the /status endpoint
# report claims as they finish being judged, one by one.
#   LIVE_JUDGE_PROGRESS[thread_id] = {claim_id: {"verdict":..., "claim_text":...}}
LIVE_JUDGE_PROGRESS: dict = {}


def get_live_judge_progress(thread_id: str) -> dict:
    """Return the live per-claim judging results for a thread (may be partial)."""
    return LIVE_JUDGE_PROGRESS.get(thread_id, {})


def reset_live_judge_progress(thread_id: str) -> None:
    LIVE_JUDGE_PROGRESS.pop(thread_id, None)


class FallbackLLM:
    """Primary (OpenRouter) with automatic, sticky fallback to secondary (NIM).

    Behaviour:
      - Normal errors: log, fall back to secondary for this call only.
      - Rate-limit / credit / auth errors (OpenRouterRateLimitError): fall back
        AND latch OpenRouter off for the rest of the process so we don't waste
        time hitting the same wall on every claim.
    """

    def __init__(self, primary, secondary):
        self.primary = primary
        self.secondary = secondary

    def invoke(self, messages: Any) -> Any:
        global _OPENROUTER_DISABLED

        primary_is_openrouter = isinstance(self.primary, OpenRouterLLM)

        # If OpenRouter is latched off and THIS primary is OpenRouter, skip it.
        if primary_is_openrouter and _OPENROUTER_DISABLED:
            return self.secondary.invoke(messages)

        try:
            return self.primary.invoke(messages)
        except OpenRouterRateLimitError as e:
            # Latch OpenRouter off for the rest of the run only when it was the
            # OpenRouter provider that hit the daily/credits wall.
            if primary_is_openrouter:
                logger.warning(
                    "OpenRouter rate-limited/out of credits (%s). "
                    "Disabling OpenRouter for the rest of this run.",
                    str(e)[:160],
                )
                _OPENROUTER_DISABLED = True
            else:
                logger.warning("Primary provider auth/limit error: %s. Falling back.", str(e)[:160])
        except Exception as e:
            logger.warning("Primary LLM failed: %s. Falling back for this call.", str(e)[:160])

        return self.secondary.invoke(messages)


def _iter_balanced_spans(text: str, open_ch: str, close_ch: str):
    """
    Yield every top-level balanced substring bounded by open_ch/close_ch,
    correctly ignoring braces/brackets that appear inside JSON string literals.
    Handles the common failure mode where a reasoning model emits a
    'thinking process' preamble that itself contains schema fragments like
    '{ "verdict": ... }' before the real JSON payload.
    """
    depth = 0
    start = -1
    in_str = False
    escape = False
    for i, ch in enumerate(text):
        if in_str:
            if escape:
                escape = False
            elif ch == "\\":
                escape = True
            elif ch == '"':
                in_str = False
            continue
        if ch == '"':
            in_str = True
            continue
        if ch == open_ch:
            if depth == 0:
                start = i
            depth += 1
        elif ch == close_ch and depth > 0:
            depth -= 1
            if depth == 0 and start != -1:
                yield text[start:i + 1]
                start = -1


def _parse_json_robust(raw: str, default: Any = None) -> Any:
    """Robustly extract and parse JSON from LLM output even if surrounded by commentary.

    Strategy (in order):
      1. Strip code fences and <think>...</think> reasoning blocks.
      2. Try a direct parse.
      3. Scan for every balanced [...] and {...} span (string-aware) and parse
         the LARGEST one that yields valid JSON. This survives reasoning-model
         preambles such as "Here's a thinking process: ... {schema} ..." that
         previously broke the naive first-brace/last-brace extraction.
    """
    if not raw:
        return default
    text = raw.strip()

    # If wrapped in code fences, prefer the fenced content
    if "```json" in text:
        text = text.split("```json", 1)[1].split("```", 1)[0].strip()
    elif "```" in text:
        parts = text.split("```")
        if len(parts) >= 3:
            text = parts[1].strip()

    # Strip thinking tags if present (some models emit <think>...</think>)
    if "</think>" in text:
        text = text.split("</think>")[-1].strip()

    def _looks_like_schema_template(obj: Any) -> bool:
        """Reject ONLY JSON the model literally copied from the prompt's schema
        example, where values are placeholder markers like '...' or option
        lists like 'high|medium|low'.

        IMPORTANT: empty strings are NOT placeholders — a real 'Unverified' or
        'Contradicted' verdict legitimately has empty supporting_span /
        best_source_url / source_name. Counting '' as a placeholder previously
        rejected valid answers and forced everything to Unverified."""
        if not isinstance(obj, dict):
            return False
        vals = [v for v in obj.values() if isinstance(v, str)]
        if not vals:
            return False
        template_markers = 0
        for v in vals:
            s = v.strip()
            if s in ("...", "…") or s.startswith("..."):
                template_markers += 1
            elif "|" in s and " " not in s:  # e.g. "high|medium|low", "a|b|c"
                template_markers += 1
        # Flag only if there are at least TWO genuine template markers. Empty
        # strings are ignored entirely.
        return template_markers >= 2

    # 1. Direct parse
    try:
        obj = json.loads(text)
        if not _looks_like_schema_template(obj):
            return obj
    except Exception:
        pass

    # 2. Collect all balanced candidate spans. Arrays are preferred over objects
    #    (claim extraction returns an array), then the largest valid span wins.
    candidates = list(_iter_balanced_spans(text, "[", "]")) + \
        list(_iter_balanced_spans(text, "{", "}"))
    # Try longest candidates first — the real payload is normally the biggest
    candidates.sort(key=len, reverse=True)
    for cand in candidates:
        try:
            obj = json.loads(cand)
        except Exception:
            continue
        if _looks_like_schema_template(obj):
            continue  # skip echoed schema examples
        return obj

    # 3. Repair truncated JSON. When a long field (e.g. "reasoning") hits the
    #    model's max_tokens, the response is cut off mid-string with no closing
    #    brace — no balanced span exists. Close the open string/brackets and
    #    retry so we still recover the verdict instead of defaulting to Unverified.
    repaired = _repair_truncated_json(text)
    if repaired is not None and not _looks_like_schema_template(repaired):
        return repaired

    raise json.JSONDecodeError(
        f"Could not parse JSON from output: {text[:120]}...", text, 0
    )


def _repair_truncated_json(text: str) -> Optional[Any]:
    """Best-effort recovery of a JSON object/array that was cut off mid-output.

    Walks the text tracking string/escape state and bracket depth, then appends
    the closing quote/brackets needed to make it parseable. Returns the parsed
    object or None if it still can't be salvaged."""
    start = text.find("{")
    alt = text.find("[")
    if alt != -1 and (start == -1 or alt < start):
        start = alt
    if start == -1:
        return None

    buf = text[start:]
    in_str = False
    escape = False
    stack = []
    for ch in buf:
        if in_str:
            if escape:
                escape = False
            elif ch == "\\":
                escape = True
            elif ch == '"':
                in_str = False
            continue
        if ch == '"':
            in_str = True
        elif ch in "{[":
            stack.append("}" if ch == "{" else "]")
        elif ch in "}]":
            if stack:
                stack.pop()

    repaired = buf
    if in_str:
        repaired += '"'          # close the dangling string
    # Drop a trailing comma or dangling ':' that would break parsing.
    repaired = repaired.rstrip()
    if repaired.endswith(","):
        repaired = repaired[:-1]
    if repaired.endswith(":"):
        repaired += ' ""'        # give the dangling key an empty value
    while stack:                 # close any open braces/brackets
        repaired += stack.pop()

    try:
        return json.loads(repaired)
    except Exception:
        return None


# Set LLM_DEBUG=true in .env to log full raw LLM outputs + provider/model for
# every judge/extract call. Verbose — use only when diagnosing parse issues.
_LLM_DEBUG = os.getenv("LLM_DEBUG", "false").strip().lower() in ("1", "true", "yes")


def _diagnose_json_failure(raw: str) -> str:
    """Produce a short, precise reason WHY a raw string failed to parse as JSON.
    Used in warning logs so the exact problem is visible without dumping the
    whole payload."""
    if not raw:
        return "empty output (model returned nothing — likely token cap spent on hidden reasoning)"
    s = raw.strip()
    reasons = []
    low = s.lower()
    if low.startswith("here's a thinking") or low.startswith("here is a thinking") or "<think>" in low:
        reasons.append("reasoning-preamble (model emitted chain-of-thought, not JSON)")
    if not (s.startswith("{") or s.startswith("[") or s.startswith("```")):
        reasons.append(f"does-not-start-with-JSON (first char {s[:1]!r})")
    # Try a direct load to capture the exact JSONDecodeError position.
    try:
        json.loads(s)
        reasons.append("json.loads actually SUCCEEDS on raw (bug is in guard/scan logic)")
    except json.JSONDecodeError as je:
        reasons.append(f"JSONDecodeError: {je.msg} at line {je.lineno} col {je.colno} (char {je.pos})")
    except Exception as e:  # noqa
        reasons.append(f"{type(e).__name__}: {str(e)[:60]}")
    # Detect truncation: unbalanced braces or dangling open string.
    opens = s.count("{") + s.count("[")
    closes = s.count("}") + s.count("]")
    if opens > closes:
        reasons.append(f"likely-truncated (unbalanced: {opens} open vs {closes} close)")
    return "; ".join(reasons) or "unknown"


def _get_nim_llm(temperature: float = 0.1, json_mode: bool = True, model: Optional[str] = None) -> Any:
    """Return an LLM instance configured for the pipeline (OpenRouter primary, NIM fallback).

    Args:
        temperature: sampling temperature.
        json_mode: when True (default) ask OpenRouter to guarantee a JSON object.
        model: explicit OpenRouter model override. Falls back to OPENROUTER_MODEL
               env var, then the GLM default.
    """
    aicredits_key = os.getenv("ai_credit", "").strip()
    morph_key = os.getenv("MORPH_API_KEY", "").strip()

    # Provider order: AI Credits (primary) -> Morph (secondary).
    # NIM and OpenRouter have been dropped from the chain per configuration.
    provider = os.getenv("LLM_PROVIDER", "aicredits").strip().lower()

    aicredits_llm = None
    if aicredits_key and not aicredits_key.startswith("<"):
        aicredits_llm = AICreditsLLM(
            model=model or os.getenv("AICREDITS_MODEL", "google/gemini-2.5-flash-lite"),
            api_key=aicredits_key,
            temperature=temperature,
            max_tokens=8096,
            timeout=120,
            json_mode=json_mode,
        )

    morph_llm = None
    if morph_key and not morph_key.startswith("<"):
        morph_llm = MorphLLM(
            model=os.getenv("MORPH_MODEL", "morph-glm53-744b"),
            api_key=morph_key,
            temperature=temperature,
            max_tokens=8096,
            timeout=120,
            json_mode=json_mode,
        )

    # Build the ordered chain (primary first).
    if provider == "morph":
        chain = [morph_llm, aicredits_llm]
    else:  # default: aicredits first
        chain = [aicredits_llm, morph_llm]

    chain = [llm for llm in chain if llm is not None]
    if not chain:
        raise ValueError(
            "No LLM provider configured. Set ai_credit or MORPH_API_KEY in .env."
        )
    if len(chain) == 1:
        return chain[0]

    # Nest FallbackLLM right-to-left so failures cascade primary -> secondary.
    llm = chain[-1]
    for provider_llm in reversed(chain[:-1]):
        llm = FallbackLLM(provider_llm, llm)
    return llm


_get_llm = _get_nim_llm


_researcher = HybridResearcher()


# =============================================================================
# 3. Node Functions
# =============================================================================

# ---- Node 1: Profile Ingestion via Apify ---------------------------------

def ingest_profile(state: GraphState) -> dict:
    """
    Call the Apify LinkedIn Profile Scraper and extract structured data.
    Actor ID: 4aIBkCdEVP6xBbX62
    """
    url = state["linkedin_url"]
    token = os.getenv("APIFY_API_TOKEN", "")

    if not token:
        logger.error("APIFY_API_TOKEN is not set.")
        return {
            "raw_profile": {},
            "profile_text": "ERROR: Apify API token not configured.",
            "executive_name": "Unknown",
            "google_person_context": "",
        }

    try:
        client = ApifyClient(token)
        # harvestapi/linkedin-profile-scraper uses 'queries' field
        run_input = {
            "profileScraperMode": "Profile details no email ($4 per 1k)",
            "queries": [url],
        }

        logger.info("Starting Apify actor run for: %s", url)
        # Bound the run. Without a wait limit the client waits indefinitely
        # (the ~24h "86,400,000 ms" timeout you saw), so a blocked/hung
        # LinkedIn scrape would stall the whole pipeline. We cap BOTH:
        #   - run_timeout:    server-side max actor runtime
        #   - wait_duration:  how long the client blocks waiting for it
        # Override via APIFY_TIMEOUT_SECS.
        from datetime import timedelta
        apify_timeout = int(os.getenv("APIFY_TIMEOUT_SECS", "180"))
        run = client.actor("harvestapi/linkedin-profile-scraper").call(
            run_input=run_input,
            run_timeout=timedelta(seconds=apify_timeout),
            wait_duration=timedelta(seconds=apify_timeout + 15),
        )

        # A run that timed out or was aborted still returns a Run object; guard
        # against a missing/failed run before touching the dataset.
        run_status = getattr(run, "status", None) if run else None
        if not run or run_status not in ("SUCCEEDED", "READY", None):
            logger.error("Apify run did not succeed (status=%s) for %s", run_status, url)
            return {
                "raw_profile": {},
                "profile_text": (
                    f"Apify run did not complete (status: {run_status}). "
                    "This is usually a timeout, LinkedIn block, or insufficient Apify credits."
                ),
                "executive_name": "Unknown",
                "google_person_context": "",
            }

        # Collect results — run is a Run object, access dataset id via attribute
        items = list(
            client.dataset(run.default_dataset_id).iterate_items()
        )

        if not items:
            logger.warning("Apify returned no items for URL: %s", url)
            return {
                "raw_profile": {},
                "profile_text": "No profile data retrieved from Apify.",
                "executive_name": "Unknown",
                "google_person_context": "",
            }

        profile = items[0]
        
        if "error" in profile and "status" in profile:
            logger.error("Apify scraper returned an error: %s", profile.get("error"))
            return {
                "raw_profile": profile,
                "profile_text": f"Apify Scraping Error: {profile.get('error')}. LinkedIn may have blocked the request.",
                "executive_name": "Unknown",
                "google_person_context": "",
            }

        # harvestapi actor returns 'name' directly, or firstName/lastName
        exec_name = (
            profile.get("name", "")
            or f"{profile.get('firstName', '')} {profile.get('lastName', '')}".strip()
            or profile.get("fullName", "Unknown Executive")
        )

        # Flatten into readable text for LLM processing
        sections = []
        sections.append(f"Name: {exec_name}")

        headline = profile.get("headline", "")
        if headline:
            sections.append(f"Headline: {headline}")

        # 'about' is the summary field in harvestapi output
        summary = profile.get("about", profile.get("summary", ""))
        if summary:
            sections.append(f"Summary: {summary}")

        # Experience — harvestapi uses 'positions' or 'experience' list
        experience_list = profile.get("positions", profile.get("experience", []))
        for exp in experience_list:
            title = exp.get("position") or exp.get("title", "")
            company = exp.get("companyName", exp.get("company", ""))
            
            # Extract date range
            start_val = exp.get("startDate")
            end_val = exp.get("endDate")
            start_txt = start_val.get("text", "") if isinstance(start_val, dict) else str(start_val or "")
            end_txt = end_val.get("text", "") if isinstance(end_val, dict) else str(end_val or "")
            dates_txt = f"{start_txt} - {end_txt}".strip(" -")
            duration = exp.get("duration", "")
            date_range = dates_txt or duration or exp.get("dateRange", exp.get("timePeriod", ""))
            
            desc = exp.get("description") or ""
            entry = f"Role: {title} at {company}" if title else f"Role at {company}"
            if date_range:
                entry += f" ({date_range})"
            if desc:
                entry += f"\n  Description: {desc[:300]}"
            sections.append(entry)

        # Education — harvestapi uses 'educations' list
        education_list = profile.get("educations", profile.get("education", []))
        for edu in education_list:
            school = edu.get("schoolName", edu.get("school", ""))
            degree = edu.get("degreeName", edu.get("degree", ""))
            field = edu.get("fieldOfStudy", edu.get("field", ""))
            entry = f"Education: {degree} in {field} from {school}"
            sections.append(entry)

        # Certifications
        for cert in profile.get("certifications", []):
            sections.append(f"Certification: {cert.get('name', '')}")

        profile_text = "\n".join(sections)

        # Enrich with Google AI Mode person research (SerpAPI)
        google_context = ""
        try:
            company = ""
            if profile.get("experience"):
                company = profile["experience"][0].get("companyName", "")
            person_data = _researcher.research_person(exec_name, company)
            google_context = person_data.get("ai_summary", "")
            if google_context:
                logger.info("Google AI Mode enrichment: %d chars for %s", len(google_context), exec_name)
        except Exception as exc:
            logger.warning("Google person research failed (non-fatal): %s", exc)

        logger.info("Profile ingested for: %s", exec_name)
        return {
            "raw_profile": profile,
            "profile_text": profile_text,
            "executive_name": exec_name,
            "google_person_context": google_context,
        }

    except Exception as exc:
        logger.error("Apify ingestion failed: %s", exc)
        return {
            "raw_profile": {},
            "profile_text": f"Ingestion error: {exc}",
            "executive_name": "Unknown",
            "google_person_context": "",
        }


# ---- Node 2: Claim Extraction via NVIDIA NIM -----------------------------

def extract_claims(state: GraphState) -> dict:
    """
    Use NVIDIA NIM to decompose the profile into
    atomic, verifiable factual claims. Enforces one subject, one predicate, one number per claim.
    """
    # Extraction returns an array. OpenRouter json_object mode requires an
    # object, so we ask for {"claims": [...]} and unwrap it below.
    llm = _get_nim_llm(
        temperature=0.0,
        json_mode=True,
        model=os.getenv("EXTRACT_MODEL") or None,
    )

    system_prompt = (
        "You are a meticulous due-diligence analyst. "
        "Your task is to extract specific, verifiable factual claims from "
        "the provided LinkedIn profile text AND the Web Intelligence context.\n\n"
        "CRITICAL RULES:\n"
        "1. ATOMICITY (but do NOT over-split): One checkable fact per claim. A fact may contain the numbers/dates that belong to it. Do NOT split a single fact across multiple attesting bodies — e.g. 'Recognized by Forbes and Hurun as youngest Indian billionaire' is ONE claim, not two. Do NOT create separate claims for each source that reports the same fact.\n"
        "2. TYPING: Each claim must have a specific type from: transaction, ranking, role, headcount, biographical, self_milestone, award.\n"
        "3. COUNTERPARTY: If the claim is a 'transaction', identify the other side of the transaction (e.g., acquirer name).\n"
        "4. DEDUPLICATION: Do NOT extract the same fact twice. If a claim is repeated in the profile (e.g., 'Founder of X' in both About and Experience), extract it ONLY ONCE.\n"
        "5. HIGH IMPACT ONLY: Ignore trivial statements, generic fluff, and minor timeline details. Only extract the MAJOR financial, milestone, transaction, and leadership claims. Extract BETWEEN 12 AND 15 claims total — pick the highest-risk, most substantial facts and merge closely related ones.\n\n"
        "Return ONLY a valid JSON object with a single key \"claims\" whose value is an array. "
        "Each array element must match this exact schema:\n"
        "{\n"
        '  "claims": [\n'
        "    {\n"
        '      "text": "The atomic claim text",\n'
        '      "type": "transaction|ranking|role|headcount|biographical|self_milestone|award",\n'
        '      "subject": "The person or entity the claim is about",\n'
        '      "counterparty": "The other side of the transaction, or null",\n'
        '      "value": 1000000.0,\n'
        '      "unit": "USD",\n'
        '      "as_of": "2020",\n'
        '      "source_in_profile": "about|experience|headline"\n'
        "    }\n"
        "  ]\n"
        "}\n\n"
        "Extract BETWEEN 12 AND 15 claims (never more than 15). Focus ONLY on the highest-risk, most substantial facts, and merge facts that a single source would attest together.\n\n"
        "CRITICAL OUTPUT REQUIREMENT: Output ONLY the JSON object. "
        "Do NOT include any thinking process, reasoning steps, conversational intro, or markdown fences."
    )

    google_ctx = state.get("google_person_context", "")
    content = f"### LinkedIn Profile:\n{state['profile_text']}\n"
    if google_ctx:
        content += f"\n### Web Intelligence Context (SerpAPI):\n{google_ctx}\n"

    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=content),
    ]

    try:
        response = llm.invoke(messages)
        raw = response.content.strip() if response.content else ""

        if _LLM_DEBUG:
            pm = (getattr(response, "raw_data", {}) or {}).get("model", "?")
            logger.info("[extract] provider_model=%s raw_len=%d FULL RAW:\n%s", pm, len(raw), raw)

        if not raw:
            logger.error("Claim extraction: model returned EMPTY output. "
                         "Likely token cap spent on hidden reasoning, or model unsupported.")
            raise json.JSONDecodeError("Empty response", "", 0)

        parsed = _parse_json_robust(raw, default=[])
        # Accept either {"claims": [...]} (json_object mode) or a bare [...] array.
        if isinstance(parsed, dict):
            raw_claims = parsed.get("claims")
            if raw_claims is None:
                # Some models return the array under a different key or as the
                # object itself; fall back sensibly.
                list_vals = [v for v in parsed.values() if isinstance(v, list)]
                raw_claims = list_vals[0] if list_vals else [parsed]
        else:
            raw_claims = parsed
        if not isinstance(raw_claims, list):
            raw_claims = [raw_claims]

        # Validate with Pydantic
        claims = []
        for rc in raw_claims:
            try:
                # Add ID if missing
                if "id" not in rc:
                    rc["id"] = str(uuid.uuid4())[:8]
                c = Claim(**rc)
                claims.append(c.model_dump())
            except Exception as e:
                logger.warning(f"Failed to validate claim {rc}: {e}")

        logger.info("Extracted %d valid atomic claims from profile.", len(claims))
        return {"claims": claims, "internal_contradictions": [], "footprint_metrics": {}}

    except (json.JSONDecodeError, IndexError) as exc:
        _raw = locals().get('raw', '') or ''
        logger.error("Claim extraction parse FAILED: %s | diag=%s",
                     exc, _diagnose_json_failure(_raw))
        logger.error("Claim extraction FULL RAW (len=%d):\n%s", len(_raw), _raw)
        profile_text = state.get("profile_text", "")
        fallback_claims = []
        for line in profile_text.split("\n"):
            line = line.strip()
            if line and len(line) > 20 and any(kw in line.lower() for kw in [
                "role:", "education:", "certification:", "founded", "ceo", "chairman",
                "managed", "raised", "invested", "built", "led", "$", "%"
            ]):
                fallback_claims.append(Claim(
                    text=line,
                    type=ClaimType.ROLE,
                    subject="Executive",
                    source_in_profile="about"
                ).model_dump())
        if not fallback_claims:
            fallback_claims = [Claim(
                text=profile_text[:200],
                type=ClaimType.ROLE,
                subject="Executive",
                source_in_profile="about"
            ).model_dump()]
        logger.info("Fallback extracted %d claims from profile text.", len(fallback_claims))
        return {"claims": fallback_claims, "internal_contradictions": [], "footprint_metrics": {}}


def _extract_company_names(state: GraphState) -> List[str]:
    """Derive the list of company names for THIS profile dynamically from the
    scraped experience/positions. No hardcoded companies — works for any
    subject in production."""
    names = set()
    profile = state.get("raw_profile", {}) or {}
    for exp in (profile.get("positions") or profile.get("experience") or []):
        if isinstance(exp, dict):
            comp = (exp.get("companyName") or exp.get("company") or "").strip().lower()
            if comp and len(comp) > 1:
                names.add(comp)
    # Also pull counterparties named in the extracted claims.
    for c in state.get("claims", []) or []:
        cp = (c.get("counterparty") or "").strip().lower()
        if cp and len(cp) > 1:
            names.add(cp)
    return sorted(names, key=len, reverse=True)  # longest first for matching


def _company_of(text_lower: str, companies: List[str]) -> Optional[str]:
    """Return the first known company name (from the profile-derived list)
    mentioned in the text, else None. Ensures role/date contradictions only
    compare claims about the SAME company."""
    for name in companies:
        if name and name in text_lower:
            return name
    return None


# Map common domains to friendly publication names for the ledger UI.
# Generic, subject-agnostic publication/reference domains only. (No
# subject-specific company domains — those are derived per-profile at runtime.)
_DOMAIN_NAMES = {
    "cnbc.com": "CNBC", "forbes.com": "Forbes", "forbesindia.com": "Forbes India",
    "techcrunch.com": "TechCrunch", "reuters.com": "Reuters", "wired.com": "WIRED",
    "wikipedia.org": "Wikipedia", "bloomberg.com": "Bloomberg", "ft.com": "Financial Times",
    "wsj.com": "Wall Street Journal", "nytimes.com": "New York Times",
    "business-standard.com": "Business Standard", "timesofindia.indiatimes.com": "Times of India",
    "economictimes.indiatimes.com": "Economic Times",
    "sec.gov": "SEC filing", "crunchbase.com": "Crunchbase", "linkedin.com": "LinkedIn",
    "gulfbusiness.com": "Gulf Business", "arabianbusiness.com": "Arabian Business",
    "thenationalnews.com": "The National", "gulfnews.com": "Gulf News",
    "yahoo.com": "Yahoo", "qz.com": "Quartz", "tracxn.com": "Tracxn",
}


def _significant_numbers(text: str) -> List[float]:
    """Extract significant numeric magnitudes from text, expanding M/B/K/thousand
    /million/billion suffixes so figures can be compared. Handles comma-grouped
    thousands ("1,600" -> 1600, NOT [1, 600]). Ignores bare years.
    """
    out: List[float] = []
    if not text:
        return out
    # Digits may include comma or space thousands separators and an optional
    # decimal part. e.g. "1,600", "1 600", "109.8"
    pattern = re.compile(
        r'\$?\s*(\d{1,3}(?:[,\s]\d{3})+(?:\.\d+)?|\d+(?:\.\d+)?)\s*'
        r'(billion|bn|b|million|mn|m|thousand|k)?',
        re.IGNORECASE,
    )
    for m in pattern.finditer(text):
        raw_num = m.group(1).replace(",", "").replace(" ", "")
        try:
            num = float(raw_num)
        except ValueError:
            continue
        suffix = (m.group(2) or "").lower()
        mult = {
            "billion": 1e9, "bn": 1e9, "b": 1e9,
            "million": 1e6, "mn": 1e6, "m": 1e6,
            "thousand": 1e3, "k": 1e3,
        }.get(suffix, 1)
        val = num * mult
        # Skip bare years (e.g. 1998, 2016) — those are dates, not amounts.
        if mult == 1 and 1900 <= num <= 2100 and num.is_integer():
            continue
        out.append(val)
    return out


def _domain_to_name(url: str) -> str:
    """Turn a URL into a readable source label (cnbc.com -> CNBC)."""
    try:
        import urllib.parse
        host = urllib.parse.urlparse(url).netloc.lower().removeprefix("www.")
        for dom, label in _DOMAIN_NAMES.items():
            if host.endswith(dom):
                return label
        # Otherwise use the second-level domain, title-cased.
        parts = host.split(".")
        if len(parts) >= 2:
            return parts[-2].capitalize()
        return host or "Source"
    except Exception:
        return "Source"


# ---- Node 2.5: Internal Consistency --------------------------------------

def internal_consistency(state: GraphState) -> dict:
    """
    Phase 2: Internal consistency
    Diffs the profile against itself with no network calls.
    Groups claims by entity name, compares dates, roles, and numeric values.
    """
    claims = state.get("claims", [])
    contradictions = []

    # Build the set of name tokens for THIS profile's executive so we can group
    # all claims about the person under one bucket — works for any subject, not
    # a hardcoded name.
    exec_name = (state.get("executive_name", "") or "").lower().strip()
    exec_tokens = {t for t in re.split(r"\s+", exec_name) if len(t) > 2}

    # Company names for THIS profile (dynamic — no hardcoded list).
    companies = _extract_company_names(state)

    groups = {}
    for c in claims:
        subj = c.get("subject", "").lower().strip()
        # If the claim's subject overlaps the executive's name (or is a generic
        # "executive"/"the founder" etc.), bucket it as the person.
        subj_tokens = {t for t in re.split(r"\s+", subj) if len(t) > 2}
        if (exec_tokens and (subj_tokens & exec_tokens)) or \
                subj in ("executive", "the executive", "the founder", "he", "she", "they"):
            subj = "executive"
        if subj not in groups:
            groups[subj] = []
        groups[subj].append(c)
        
    for subj, subj_claims in groups.items():
        for i in range(len(subj_claims)):
            for j in range(i + 1, len(subj_claims)):
                c1 = subj_claims[i]
                c2 = subj_claims[j]
                
                # Check 1: Date ranges — only when both claims name the SAME
                # company (otherwise different companies' date ranges are not a
                # contradiction).
                dates1 = re.findall(r'\b(19\d{2}|20\d{2})\b', c1.get("text", ""))
                dates2 = re.findall(r'\b(19\d{2}|20\d{2})\b', c2.get("text", ""))
                _c1 = _company_of(c1.get("text", "").lower(), companies)
                _c2 = _company_of(c2.get("text", "").lower(), companies)
                if len(dates1) >= 2 and len(dates2) >= 2 and _c1 and _c1 == _c2:
                    if dates1[:2] != dates2[:2] and c1.get("source_in_profile") != c2.get("source_in_profile"):
                        contradictions.append({
                            "type": "date_mismatch",
                            "subject": subj,
                            "claims": [c1.get("id"), c2.get("id")],
                            "description": f"Date mismatch for {_c1}: {'-'.join(dates1[:2])} vs {'-'.join(dates2[:2])}"
                        })
                
                # Check 2: Role — founder vs co-founder.
                # This is only a real contradiction when both claims are about
                # the SAME company. Comparing "founder of Ai.tech" against
                # "co-founder of Directi" is NOT a contradiction — that was the
                # bug that produced false CONTRADICTED verdicts. Also, even for
                # the same company, founder/co-founder is an attribution nuance
                # (his brother co-founded some of these), NOT a hard refusal,
                # so we DISABLE it as an auto-contradiction and let the judge
                # attribute it. Kept here (guarded off) for reference.
                text1_lower = c1.get("text", "").lower()
                text2_lower = c2.get("text", "").lower()
                comp1 = _company_of(text1_lower, companies)
                comp2 = _company_of(text2_lower, companies)
                role1 = "co-founder" if "co-found" in text1_lower else ("founder" if "found" in text1_lower else None)
                role2 = "co-founder" if "co-found" in text2_lower else ("founder" if "found" in text2_lower else None)
                if (False  # founder/co-founder is an attribution nuance, not a contradiction
                        and role1 and role2 and role1 != role2
                        and comp1 and comp2 and comp1 == comp2
                        and c1.get("source_in_profile") != c2.get("source_in_profile")):
                     contradictions.append({
                            "type": "role_mismatch",
                            "subject": subj,
                            "claims": [c1.get("id"), c2.get("id")],
                            "description": f"Role mismatch for {subj}: {role1} vs {role2}"
                     })
                     
                # Check 3: Numeric / As-of mismatch
                if c1.get("value") == c2.get("value") and c1.get("value") is not None:
                    if c1.get("as_of") and c2.get("as_of") and c1.get("as_of") != c2.get("as_of"):
                        contradictions.append({
                            "type": "numeric_mismatch",
                            "subject": subj,
                            "claims": [c1.get("id"), c2.get("id")],
                            "description": f"Age/Date mismatch for value {c1.get('value')}: {c1.get('as_of')} vs {c2.get('as_of')}"
                        })

    logger.info("Internal consistency found %d contradictions.", len(contradictions))
    return {"internal_contradictions": contradictions}

# ---- Node 2.6: Primary Source Routing ------------------------------------

def route_primary_source(state: GraphState) -> dict:
    """
    Phase 3: Primary-source routing
    Trigger targeted lookups (e.g., EDGAR) before general search.
    """
    claims = state.get("claims", [])
    exec_name = state.get("executive_name", "Unknown")
    evidence_cache = state.get("evidence_cache", {})
    
    for idx, claim in enumerate(claims):
        cid = claim.get("id", str(idx))
        if cid not in evidence_cache:
            evidence_cache[cid] = {"primary_evidence": []}
            
        ctype = claim.get("type")
        cparty = claim.get("counterparty")
        
        # TRANSACTION with counterparty -> EDGAR
        if ctype == ClaimType.TRANSACTION.value and cparty:
            logger.info(f"Routing claim {cid} to EDGAR for {cparty}")
            edgar_res = _researcher.edgar_fulltext(f"{cparty} {exec_name}")
            if edgar_res:
                evidence_cache[cid]["primary_evidence"].append({
                    "source": "edgar",
                    "data": edgar_res
                })
                
    return {"evidence_cache": evidence_cache}

# ---- Node 3: Batch Evidence Gathering ------------------------------------

def gather_evidence_batch(state: GraphState) -> dict:
    """
    Loop over all claims and gather evidence using HybridResearcher concurrently.
    """
    claims = state.get("claims", [])
    exec_name = state.get("executive_name", "Unknown")
    evidence_cache = {}

    logger.info("Gathering evidence for %d claims for %s", len(claims), exec_name)

    def _gather(claim):
        cid = claim.get("id", "")
        claim_text = claim.get("text", str(claim))
        logger.info("Researching claim %s: %s", cid, claim_text[:80])
        evidence = _researcher.verify_claim(claim, exec_name)
        return cid, evidence

    with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
        futures = [executor.submit(_gather, claim) for claim in claims]
        for future in concurrent.futures.as_completed(futures):
            try:
                cid, evidence = future.result()
                if cid:
                    if cid in evidence_cache:
                        evidence_cache[cid].update(evidence)
                    else:
                        evidence_cache[cid] = evidence
            except Exception as e:
                logger.error(f"Error gathering evidence for a claim: {e}")

    return {"evidence_cache": evidence_cache}

# ---- Node 4: Provenance Clustering ---------------------------------------

def cluster_provenance(state: GraphState) -> dict:
    """
    Phase 5: Provenance clustering
    Replaces pool subtraction. Groups retrieved documents into clusters based on text.
    Tags each cluster as self-originating or independent.
    """
    claims = state.get("claims", [])
    evidence_cache = state.get("evidence_cache", {})
    
    # Derive the subject's own company domains dynamically from the profile so
    # self-originating sources can be detected for ANY subject (no hardcoding).
    company_domains = set()
    for name in _extract_company_names(state):
        # Turn a company name into a likely domain guess, e.g. "media.net" ->
        # "media.net", "acme corp" -> "acmecorp.com".
        n = name.strip().lower()
        if "." in n and " " not in n:
            company_domains.add(n)                      # already looks like a domain
        else:
            slug = re.sub(r"[^a-z0-9]", "", n)
            if slug:
                company_domains.add(f"{slug}.com")
    # Include any explicit websites present in the raw profile.
    prof = state.get("raw_profile", {}) or {}
    for key in ("website", "websites", "companyWebsite"):
        val = prof.get(key)
        if isinstance(val, str) and "." in val:
            company_domains.add(val.replace("https://", "").replace("http://", "").split("/")[0].removeprefix("www."))
    
    for idx, claim in enumerate(claims):
        cid = claim.get("id", str(idx))
        if cid not in evidence_cache:
            continue
            
        evidence = evidence_cache[cid]
        deep_evidence = evidence.get("deep_evidence", [])
        
        clusters = []
        for doc in deep_evidence:
            text = doc.get("text", "")
            url = doc.get("url", "")
            if not text:
                continue
                
            matched_cluster = None
            for cl in clusters:
                if any(_researcher.same_origin(text, existing_doc["text"]) for existing_doc in cl["docs"]):
                    matched_cluster = cl
                    break
                    
            if matched_cluster:
                matched_cluster["docs"].append({"url": url, "text": text})
            else:
                clusters.append({
                    "docs": [{"url": url, "text": text}],
                    "origin": "unknown",
                    "tier": "T"
                })
                
        for cl in clusters:
            cl["origin"] = _researcher.origin_tag(cl["docs"], company_domains)
            cl["tier"] = "T"
            for doc in cl["docs"]:
                import urllib.parse
                host = urllib.parse.urlparse(doc["url"]).netloc.removeprefix("www.")
                if host in {"sec.gov", "efts.sec.gov", "icann.org", "moet.gov.ae", "difc.ae"}:
                    cl["tier"] = "P"
                    break
                if host in {"forbes.com", "forbesindia.com", "techcrunch.com", "cnbc.com", "wired.com", "domainnamewire.com", "business-standard.com", "reuters.com", "bloomberg.com", "medianama.com"}:
                    if cl["tier"] != "P":
                        cl["tier"] = "S"
                        
        evidence_cache[cid]["clusters"] = clusters
        
    return {"evidence_cache": evidence_cache}



# ---- Node 6: Per-Claim Judgement ---------------------------------------------

def judge_claims(state: GraphState, config: Optional[RunnableConfig] = None) -> dict:
    """
    Phase 6: Judge rewrite
    Processes claims concurrently. Computes deterministic verdicts if possible,
    else uses NIM to get a verdict with a required supporting span.
    """
    claims = state.get("claims", [])
    evidence_cache = state.get("evidence_cache", {})
    exec_name = state.get("executive_name", "Unknown")
    internal_contradictions = state.get("internal_contradictions", [])

    # thread_id for live progress reporting (passed by LangGraph via config).
    thread_id = ""
    if isinstance(config, dict):
        thread_id = (config.get("configurable") or {}).get("thread_id", "") or ""
    if thread_id:
        LIVE_JUDGE_PROGRESS[thread_id] = {}

    ledger = []
    refusals = []
    
    llm = _get_nim_llm(
        temperature=0.0,
        json_mode=True,
        model=os.getenv("JUDGE_MODEL") or None,
    )
    system_prompt = (
        "You are a strict due-diligence investigator for a Dubai-based advisory firm. "
        "You will be given a SINGLE claim and the evidence retrieved for it. Each piece of evidence has a URL and text snippet.\n"
        "Evaluate the evidence and assign EXACTLY ONE of THREE verdicts: "
        "Verified, Partially Verified, or Contradicted.\n"
        "SOURCE HIERARCHY: Official institutional registries, university directories, SEC filings, and primary corporate bios completely overrule third-party news blogs or aggregators. If a secondary source inverts facts (e.g. swapping degree institutions) compared to official sources, mark the claim as CONTRADICTED.\n"
        "RULES (use the full 3-label scale):\n"
        "1. Verified: The claim is CORRECT and confirmed. Two or more independent sources (not copies of each other) confirm it, at least one being a primary/named source.\n"
        "2. Partially Verified: The claim is NOT contradicted, but there isn't enough independent evidence to fully confirm it — e.g. some support exists but it is thin, traces only to the subject's own site/press release/a single origin, or a minor detail is uncertain. This means 'publishable WITH attribution'. Use this as the DEFAULT when evidence neither clearly confirms nor disproves the claim.\n"
        "3. Contradicted: The fact is WRONG. A credible source explicitly states something different (e.g. a filing/article gives a different number, age, date, or says the event did not happen). Signals: 'actually', 'corrected', 'inaccurate', 'not', 'rather than', 'false', 'no record'. State the correct value in the reasoning.\n"
        "IMPORTANT: Never invent a 4th label. If you find NO evidence at all, use Partially Verified (not contradicted — absence of evidence is not disproof). Only use Contradicted when a source actively disproves the claim.\n"
        "SUPPORTING SPAN: For Verified, quote an exact substring from the evidence. For Partially Verified, quote a span if one exists, otherwise leave it empty and explain in reasoning. For Contradicted, quote the contradicting span.\n"
        "REASONING RULE: In your reasoning, do NOT use generic terms like 'Cluster 1' or 'Cluster 2'. Instead, name the actual source or domain (e.g., 'Forbes', 'SEC filing', 'instagram.com'). Explain exactly what the source said.\n\n"
        "Return ONLY valid JSON matching this schema:\n"
        "{\n"
        '  "verdict": "Verified|Partially Verified|Contradicted",\n'
        '  "confidence": "high|medium|low",\n'
        '  "reasoning": "Explain why, naming the specific sources explicitly",\n'
        '  "supporting_span": "Exact substring quote from the evidence text supporting the claim (REQUIRED for Verified), or empty string",\n'
        '  "best_source_url": "URL of the best piece of evidence, or empty string",\n'
        '  "source_name": "The name of the publication or website (e.g., TechCrunch, LinkedIn), or empty string"\n'
        "}\n\n"
        "Keep 'reasoning' to at most 2 concise sentences (under 60 words) so the JSON is never truncated.\n"
        "CRITICAL OUTPUT REQUIREMENT: Output ONLY a valid JSON object starting directly with '{' and ending with '}'. "
        "Do NOT write any thinking process, reasoning steps, or conversational preamble."
    )

    def _judge(claim):
        cid = claim.get("id")
        claim_text = claim.get("text", "")
        ctype = claim.get("type", "")
        
        evidence = evidence_cache.get(cid, {})
        clusters = evidence.get("clusters", [])
        primary_ev = evidence.get("primary_evidence", [])
        
        # 1. Deterministic Checks
        deterministic_verdict = None
        deterministic_reasoning = ""
        
        # Check internal contradictions
        for ic in internal_contradictions:
            if cid in ic.get("claims", []):
                deterministic_verdict = VerdictType.CONTRADICTED.value
                deterministic_reasoning = f"Held back — internal profile contradiction: {ic.get('description')}"
                break
                
        if deterministic_verdict:
            res = {
                "verdict": deterministic_verdict,
                "confidence": "high",
                "reasoning": deterministic_reasoning,
                "supporting_span": None,
                "best_source_url": None
            }
        else:
            # 2. LLM Judgement
            evidence_text = ""

            # CRITICAL: include the Google AI Mode synthesis FIRST. This is
            # where corrections live (e.g. "age 34, not 33", "Skenzo was NOT
            # exited in 2008"). Previously the judge only saw crawled page text
            # and missed these, so contradicted claims slipped through.
            google_answer = evidence.get("google_ai_answer", "")
            if google_answer:
                evidence_text += (
                    "\n--- GOOGLE AI SYNTHESIS (weigh corrections heavily) ---\n"
                    f"{google_answer[:1500]}\n"
                )
            pplx_answer = evidence.get("perplexity_answer", "")
            if pplx_answer and pplx_answer.strip() not in ("", "Perplexity layer disabled."):
                evidence_text += f"\n--- PERPLEXITY SYNTHESIS ---\n{pplx_answer[:1000]}\n"

            for i, cl in enumerate(clusters):
                evidence_text += f"\n--- Cluster {i+1} (Tier {cl['tier']}, Origin: {cl['origin']}) ---\n"
                for doc in cl["docs"]:
                    evidence_text += f"URL: {doc['url']}\nText snippet: {doc['text'][:500]}...\n\n"
                    
            for pe in primary_ev:
                evidence_text += f"\n--- PRIMARY EVIDENCE ({pe['source']}) ---\n{pe['data']}\n"
                
            user_content = (
                f"Executive Name: {exec_name}\nClaim: {claim_text}\nType: {ctype}\n\n"
                f"EVIDENCE:\n{evidence_text}\n\n"
                "REMINDER: If any evidence gives a DIFFERENT number, age, date, or fact "
                "than the claim (words like 'actually', 'corrected', 'inaccurate', "
                "'not', 'rather than', 'false', 'no record' are strong signals), you MUST "
                "return verdict=Contradicted and state the correct value in reasoning. "
                "If evidence is merely thin or absent (but not disproving), use Partially "
                "Verified — never Contradicted for absence of evidence."
            )
            
            res = None
            last_raw = ""
            for attempt in range(2):
                try:
                    msgs = [
                        SystemMessage(content=system_prompt),
                        HumanMessage(content=user_content),
                    ]
                    if attempt > 0:
                        # Stricter re-prompt: the previous output could not be parsed.
                        msgs.append(HumanMessage(content=(
                            "Your previous response could not be parsed. "
                            "Respond again with ONLY the raw JSON object. "
                            "No preamble, no reasoning steps, no markdown fences. "
                            "Start with '{' and end with '}'."
                        )))
                    response = llm.invoke(msgs)
                    last_raw = (response.content or "").strip()
                    provider = getattr(response, "raw_data", {}) or {}
                    provider_model = provider.get("model", "?")

                    if _LLM_DEBUG:
                        logger.info(
                            "[judge %s attempt %d] provider_model=%s raw_len=%d FULL RAW:\n%s",
                            cid, attempt + 1, provider_model, len(last_raw), last_raw,
                        )

                    res = _parse_json_robust(last_raw, default={})
                    # The model sometimes returns valid JSON with the WRONG
                    # schema (e.g. {"description": "..."} with no verdict).
                    # Treat that as a parse failure so the retry / fallback
                    # can correct it rather than silently defaulting.
                    if not isinstance(res, dict) or not res.get("verdict"):
                        raise ValueError("parsed JSON missing 'verdict' field")
                    break
                except Exception as e:
                    # Deep diagnostics: show WHERE json failed and the full raw.
                    diag = _diagnose_json_failure(last_raw)
                    logger.warning(
                        "Judge parse failed for claim %s (attempt %d/2): %s | diag=%s",
                        cid, attempt + 1, e, diag,
                    )
                    if _LLM_DEBUG:
                        logger.warning("[judge %s attempt %d] FULL RAW that failed:\n%s",
                                       cid, attempt + 1, last_raw)

            if not isinstance(res, dict) or not res.get("verdict"):
                # Log the ENTIRE raw output (not just the head) so the exact
                # problem is always visible in the logs.
                logger.error(
                    "Failed to judge claim %s after retries. raw_len=%d FULL RAW:\n%s",
                    cid, len(last_raw), last_raw,
                )
                # Could not judge — treat as "not confirmed, not disproved" =
                # Partially Verified (absence of a parseable verdict is not proof
                # the claim is wrong).
                res = {
                    "verdict": VerdictType.PARTIALLY_VERIFIED.value,
                    "confidence": "low",
                    "reasoning": "Insufficient parseable evidence to confirm; treat as partially verified pending review.",
                    "supporting_span": None,
                    "best_source_url": None,
                }
            else:
                # Normalize any legacy/stray "Unverified" verdict to the new
                # 3-label scheme: absence of evidence = Partially Verified.
                if res.get("verdict") == "Unverified":
                    res["verdict"] = VerdictType.PARTIALLY_VERIFIED.value

                # STRICT EMPTY SPAN RULE: If Verified, MUST have a span.
                if res.get("verdict") == VerdictType.VERIFIED.value and not str(res.get("supporting_span") or "").strip():
                    res["verdict"] = VerdictType.PARTIALLY_VERIFIED.value
                    res["reasoning"] = "Downgraded to Partially Verified: No exact supporting span was quoted from the evidence. " + (res.get("reasoning", ""))
                
        final_verdict = res.get("verdict", VerdictType.PARTIALLY_VERIFIED.value)
        # Normalize any stray legacy label to the 3-label scheme.
        if final_verdict == "Unverified":
            final_verdict = VerdictType.PARTIALLY_VERIFIED.value
            
        # SAFEGUARD: Catch refusal statements hallucinated as 'Verified'
        if final_verdict in (VerdictType.VERIFIED.value, VerdictType.PARTIALLY_VERIFIED.value):
            span_reasoning = (str(res.get("supporting_span") or "") + " " + str(res.get("reasoning") or "")).lower()
            refusal_keywords = [
                "not involved", "incorrect", "false", "no record", 
                "contradicts", "not true", "does not support", "did not acquire"
            ]
            for kw in refusal_keywords:
                if kw in span_reasoning:
                    final_verdict = VerdictType.CONTRADICTED.value
                    res["reasoning"] = f"Auto-blocked: LLM output contained contradiction keyword '{kw}'. Original reasoning: {res.get('reasoning', '')}"
                    break
                    
        entry = {
            "index": claim.get("id"),
            "claim_text": claim_text,
            "category": ctype,
            "verdict": final_verdict,
            # True when the verdict is Contradicted — the "claim the system
            # refused to include" case.
            "contradicted": final_verdict == VerdictType.CONTRADICTED.value,
            "confidence": res.get("confidence", "low"),
            "reasoning": res.get("reasoning", ""),
            "supporting_span": res.get("supporting_span", ""),
            "primary_source_url": res.get("best_source_url", ""),
            "source_name": res.get("source_name", ""),
        }
        return entry

    total = len(claims)
    with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
        futures = [executor.submit(_judge, claim) for claim in claims]
        for future in concurrent.futures.as_completed(futures):
            try:
                entry = future.result()
                # Publish this claim's verdict live, one by one, so the UI can
                # stamp it the moment it's decided (not all at the end).
                if thread_id:
                    LIVE_JUDGE_PROGRESS.setdefault(thread_id, {})[entry["index"]] = {
                        "index": entry["index"],
                        "claim_text": entry["claim_text"],
                        "verdict": entry["verdict"],
                        "contradicted": entry.get("contradicted", False),
                        "confidence": entry["confidence"],
                        "source_name": entry["source_name"],
                    }
                if entry["verdict"] in (VerdictType.VERIFIED.value, VerdictType.PARTIALLY_VERIFIED.value):
                    ledger.append(entry)
                else:
                    refusals.append(entry)
            except Exception as e:
                logger.error(f"Error in judge thread: {e}")
                
    return {
        "verification_ledger": ledger,
        "refusal_log": refusals
    }


# ---- Node 6.5: Perplexity Fallback (second-opinion for weak claims) ------

def perplexity_fallback(state: GraphState) -> dict:
    """
    For every claim that came out Unverified or Contradicted, send them ALL to
    Perplexity in ONE batch call and ask for a per-claim verdict + source links.
    If Perplexity finds support, promote the claim into the ledger (with its
    Perplexity sources); otherwise keep it refused but attach whatever context
    Perplexity found.

    Controlled by USE_PERPLEXITY in .env. No-op if disabled.
    """
    # Enable the batch fallback with USE_PERPLEXITY_FALLBACK (preferred) or the
    # legacy USE_PERPLEXITY flag. Kept separate so you can run Perplexity ONLY
    # as a fallback for weak claims (cheaper) without it firing on every claim
    # during evidence gathering.
    fallback_on = (
        os.getenv("USE_PERPLEXITY_FALLBACK", "").strip().lower() in ("1", "true", "yes")
        or os.getenv("USE_PERPLEXITY", "false").strip().lower() in ("1", "true", "yes")
    )
    if not fallback_on:
        logger.info("[PerplexityFallback] skipped (USE_PERPLEXITY_FALLBACK not enabled).")
        return {}

    exec_name = state.get("executive_name", "Unknown")
    ledger = list(state.get("verification_ledger", []))
    refusals = list(state.get("refusal_log", []))

    # Re-check the claims that aren't already firmly Verified:
    #   - Partially Verified (in the ledger)  -> maybe upgrade to Verified, or
    #     flip to Contradicted if Perplexity finds it's wrong.
    #   - Contradicted (in the refusal log)   -> confirm / correct.
    # Firmly Verified claims are left untouched.
    all_entries = ledger + refusals
    weak = [
        {"id": e.get("index"), "text": e.get("claim_text", "")}
        for e in all_entries
        if e.get("verdict") != VerdictType.VERIFIED.value
    ]
    if not weak:
        logger.info("[PerplexityFallback] nothing to re-check (all Verified).")
        return {}

    results = _researcher.batch_fact_check(exec_name, weak)
    if not results:
        return {}

    new_ledger = []
    new_refusals = []
    upgraded = 0

    for entry in all_entries:
        cid = str(entry.get("index"))
        r = results.get(cid) or results.get(entry.get("index"))

        if r:
            pplx_verdict = r.get("verdict", entry.get("verdict"))
            sources = r.get("sources", []) or []
            if not sources:
                sources = results.get("_citations", [])[:1]

            # Map to the 3-label scheme.
            if pplx_verdict == VerdictType.CONTRADICTED.value:
                entry["verdict"] = VerdictType.CONTRADICTED.value
                entry["contradicted"] = True
            elif pplx_verdict == VerdictType.VERIFIED.value:
                entry["verdict"] = VerdictType.VERIFIED.value
                entry["contradicted"] = False
                upgraded += 1
            else:  # Partially Verified / Unverified / anything else
                entry["verdict"] = VerdictType.PARTIALLY_VERIFIED.value
                entry["contradicted"] = False

            entry["reasoning"] = r.get("reasoning") or entry.get("reasoning", "")
            entry["confidence"] = r.get("confidence", entry.get("confidence", "low"))
            if sources:
                url = sources[0]
                entry["primary_source_url"] = url
                entry["source_name"] = _domain_to_name(url)
                entry["all_sources"] = sources

        # Route by final verdict: Verified + Partially Verified -> ledger;
        # Contradicted -> held back.
        if entry.get("verdict") == VerdictType.CONTRADICTED.value:
            new_refusals.append(entry)
        else:
            new_ledger.append(entry)

    logger.info(
        "[PerplexityFallback] re-checked %d claims; %d confirmed as Verified.",
        len(weak), upgraded,
    )
    return {"verification_ledger": new_ledger, "refusal_log": new_refusals}


# ---- Node 7: Numeric Conflict Detection ----------------------------------

def detect_numeric_conflict(state: GraphState) -> dict:
    """
    Phase 7: Numeric conflict detection
    For HEADCOUNT and TRANSACTION claims, extract numbers from supporting span
    and compare against the claim's value. Overwrite to Contradicted if mismatch.
    """
    ledger = list(state.get("verification_ledger", []))
    refusals = list(state.get("refusal_log", []))
    
    new_ledger = []
    
    for entry in ledger:
        ctype = entry.get("category", "")
        if ctype in [ClaimType.TRANSACTION.value, ClaimType.HEADCOUNT.value]:
            span = entry.get("supporting_span", "")
            claim_text = entry.get("claim_text", "")
            if span:
                # Generic number-mismatch check (works for any subject): compare
                # the primary numeric magnitude in the claim against the numbers
                # in the supporting span. If the claim's key figure does not
                # appear in the span AND the span carries a clearly different
                # figure of the same kind, hold it back as a conflict.
                claim_nums = _significant_numbers(claim_text)
                span_nums = _significant_numbers(span)
                if claim_nums and span_nums:
                    key = claim_nums[0]
                    # Consider it a conflict if the claim's headline number is
                    # absent from the span but the span has a different number
                    # within the same order of magnitude ballpark.
                    if key not in span_nums and not any(
                        abs(key - s) / max(key, 1) < 0.05 for s in span_nums
                    ):
                        close_alt = [s for s in span_nums
                                     if 0.2 < (min(key, s) / max(key, s, 1)) < 0.95]
                        if close_alt:
                            entry["verdict"] = VerdictType.CONTRADICTED.value
                            entry["contradicted"] = True
                            entry["reasoning"] = (
                                f"Contradicted — numeric conflict: the claim states "
                                f"{key:g} but the cited source reports {close_alt[0]:g}. "
                                + (entry.get("reasoning", ""))
                            )
                            refusals.append(entry)
                            continue

        new_ledger.append(entry)
        
    return {"verification_ledger": new_ledger, "refusal_log": refusals}

# ---- Node 8: Triage Publishable ------------------------------------------

def triage_publishable(state: GraphState) -> dict:
    """
    Phase 8: Triage and output
    Maps claims to Publishable as written, Publishable with attribution, Blocked.
    """
    ledger = []
    refusals = list(state.get("refusal_log", []))
    
    # Bucket by verdict (aligned to the 3-label scheme):
    #   Verified          -> Publishable as written
    #   Partially Verified-> Publishable with attribution
    #   Contradicted      -> Blocked (held back)
    for entry in state.get("verification_ledger", []):
        if entry.get("verdict") == VerdictType.VERIFIED.value:
            entry["bucket"] = "Publishable as written"
        else:
            entry["bucket"] = "Publishable with attribution"
        ledger.append(entry)

    for entry in refusals:
        entry["bucket"] = "Blocked"
        
    return {"verification_ledger": ledger, "refusal_log": refusals}

# ---- Node 9: Measure Footprint -------------------------------------------

def measure_footprint(state: GraphState) -> dict:
    """
    Phase 9: Grounded gaps
    Computes footprint metrics deterministically.
    """
    evidence_cache = state.get("evidence_cache", {})
    metrics = {
        "independent_clusters_found": 0,
        "self_originating_clusters_found": 0,
        "tier_p_found": 0,
    }
    
    for cid, ev in evidence_cache.items():
        clusters = ev.get("clusters", [])
        for cl in clusters:
            if cl.get("origin") == "independent":
                metrics["independent_clusters_found"] += 1
            elif cl.get("origin") == "self_originating":
                metrics["self_originating_clusters_found"] += 1
            if cl.get("tier") == "P":
                metrics["tier_p_found"] += 1
                
    return {"footprint_metrics": metrics}

# ---- Node 10: Gaps Analysis ----------------------------------------------

def analyze_gaps(state: GraphState) -> dict:
    """
    Identify the top 3 positioning gaps, grounded in FootprintMetrics.
    """
    exec_name = state.get("executive_name", "Unknown")
    metrics = state.get("footprint_metrics", {})
    
    llm = _get_nim_llm(
        temperature=0.2,
        json_mode=True,
        model=os.getenv("JUDGE_MODEL") or None,
    )

    system_prompt = (
        "You are a senior positioning strategist. "
        "Identify the THREE most critical NARRATIVE GAPS in the executive's public record. "
        "Ground your analysis STRICTLY in the provided Footprint Metrics.\n\n"
        "Return ONLY a JSON object with a single key \"gaps\" whose value is an array of EXACTLY 3 items. Each item:\n"
        '{"gaps": [{"gap_title": "...", "why_it_matters": "One sentence explaining the risk", "severity": "Critical|High|Medium"}]}\n'
    )

    user_content = (
        f"Executive: {exec_name}\n\n"
        f"Footprint Metrics:\n"
        f"- Independent Clusters: {metrics.get('independent_clusters_found', 0)}\n"
        f"- Self-Originating Clusters (PR/Company): {metrics.get('self_originating_clusters_found', 0)}\n"
        f"- Tier P Sources (Primary/Gov): {metrics.get('tier_p_found', 0)}\n\n"
    )

    try:
        response = llm.invoke([
            SystemMessage(content=system_prompt),
            HumanMessage(content=user_content),
        ])
        raw = response.content.strip()
        parsed = _parse_json_robust(raw, default=[])
        if isinstance(parsed, dict):
            gaps = parsed.get("gaps")
            if gaps is None:
                list_vals = [v for v in parsed.values() if isinstance(v, list)]
                gaps = list_vals[0] if list_vals else [parsed]
        else:
            gaps = parsed
        if not isinstance(gaps, list):
            gaps = [gaps]
        gaps = gaps[:3]
        return {"gaps_analysis": gaps}

    except Exception as exc:
        logger.exception("Gaps analysis failed")
        return {"gaps_analysis": [
            {"gap_title": "Analysis unavailable", "why_it_matters": f"Gaps analysis failed: {type(exc).__name__}: {exc}", "severity": "Medium"}
        ]}


# ---- Node 11: Diagnostic Synthesizer -------------------------------------

def synthesize_report(state: GraphState) -> dict:
    """
    Generate the final one-page diagnostic markdown report.
    Enforces NoCap.ai house rules.
    """
    if not state.get("approved", False):
        logger.warning("Synthesis blocked: not approved by human reviewer.")
        return {"final_diagnostic": "ERROR: Report generation blocked. Human approval required."}

    exec_name = state.get("executive_name", "Unknown Executive")
    ledger = state.get("verification_ledger", [])
    refusals = state.get("refusal_log", [])
    gaps = state.get("gaps_analysis", [])

    # Separate by bucket
    publishable = [e for e in ledger if e.get("bucket") == "Publishable as written"]
    with_attr = [e for e in ledger if e.get("bucket") == "Publishable with attribution"]
    blocked = [e for e in refusals if e.get("bucket") == "Blocked"]

    report = f"# Executive Reputation Diagnostic: {exec_name}\n\n"
    
    report += "## 🚫 The Blocked List\n"
    if blocked:
        for b in blocked:
            report += f"- **{b['claim_text']}**\n  *Reason:* {b['reasoning']}\n"
    else:
        report += "No claims blocked.\n"
        
    report += "\n## ✅ Publishable As Written\n"
    if publishable:
        for p in publishable:
            src = p.get('primary_source_url', 'No link')
            report += f"- {p['claim_text']}\n  *[Source]({src})* - Span: \"{p.get('supporting_span', 'N/A')}\"\n"
    else:
        report += "None.\n"
        
    report += "\n## ⚠️ Publishable With Attribution\n"
    if with_attr:
        for a in with_attr:
            src = a.get('primary_source_url', 'No link')
            report += f"- {a['claim_text']}\n  *[Source]({src})* - Span: \"{a.get('supporting_span', 'N/A')}\"\n"
    else:
        report += "None.\n"
        
    report += "\n## Narrative Gaps\n"
    if gaps:
        for g in gaps:
            report += f"- **{g.get('gap_title', 'Gap')}** ({g.get('severity', 'Medium')}): {g.get('why_it_matters', '')}\n"
    else:
        report += "No gaps identified.\n"
        
    return {"final_diagnostic": report}



def _fallback_report(
    name: str, total: int, ledger: list, refusals: list
) -> str:
    """Manual report assembly if NIM synthesis fails."""
    lines = [
        f"## Executive Reputation Diagnostic: {name}\n",
        f"**Authority Score**: {len(ledger)}/{total} claims verified\n",
        "### Verified Footprint\n",
    ]
    for e in ledger:
        src = ""
        if e.get("primary_source_url"):
            src_name = e.get("source_name") or "Source"
            src = f" [Source: {src_name}]({e['primary_source_url']})"
        lines.append(f"- {e['claim_text']} ({e['verdict']}){src}")

    lines.append("\n### Warning: The Quarantine Zone\n")
    for e in refusals:
        lines.append(f"- **{e['verdict']}**: {e['claim_text']}")
        lines.append(f"  Reasoning: {e['reasoning']}")

    return "\n".join(lines)


# =============================================================================
# 4. Graph Construction
# =============================================================================

def build_graph() -> StateGraph:
    """Construct and compile the LangGraph StateGraph."""
    workflow = StateGraph(GraphState)

    # Add nodes
    workflow.add_node("ingest_profile", ingest_profile)
    workflow.add_node("extract_claims", extract_claims)
    workflow.add_node("internal_consistency", internal_consistency)
    workflow.add_node("route_primary_source", route_primary_source)
    workflow.add_node("gather_evidence_batch", gather_evidence_batch)
    workflow.add_node("cluster_provenance", cluster_provenance)
    workflow.add_node("judge_claims", judge_claims)
    workflow.add_node("perplexity_fallback", perplexity_fallback)
    workflow.add_node("detect_numeric_conflict", detect_numeric_conflict)
    workflow.add_node("triage_publishable", triage_publishable)
    workflow.add_node("measure_footprint", measure_footprint)
    workflow.add_node("analyze_gaps", analyze_gaps)
    workflow.add_node("synthesize_report", synthesize_report)

    # Linear edges
    workflow.add_edge("ingest_profile", "extract_claims")
    workflow.add_edge("extract_claims", "internal_consistency")
    workflow.add_edge("internal_consistency", "route_primary_source")
    workflow.add_edge("route_primary_source", "gather_evidence_batch")
    workflow.add_edge("gather_evidence_batch", "cluster_provenance")
    workflow.add_edge("cluster_provenance", "judge_claims")
    workflow.add_edge("judge_claims", "perplexity_fallback")
    workflow.add_edge("perplexity_fallback", "detect_numeric_conflict")
    workflow.add_edge("detect_numeric_conflict", "triage_publishable")
    workflow.add_edge("triage_publishable", "measure_footprint")
    workflow.add_edge("measure_footprint", "analyze_gaps")
    workflow.add_edge("analyze_gaps", "synthesize_report")

    # Terminal edge
    workflow.add_edge("synthesize_report", END)

    # Entry point
    workflow.set_entry_point("ingest_profile")

    return workflow


def compile_graph(checkpointer=None, interrupt_before_synthesis: bool = True):
    """
    Compile the graph with optional checkpointer and HITL interrupt.
    """
    workflow = build_graph()

    interrupt = ["synthesize_report"] if interrupt_before_synthesis else []

    if checkpointer is None:
        checkpointer = MemorySaver()

    compiled = workflow.compile(
        checkpointer=checkpointer,
        interrupt_before=interrupt,
    )
    return compiled, checkpointer


# =============================================================================
# 6. Standalone Execution (for testing without FastAPI)
# =============================================================================

if __name__ == "__main__":
    import sys

    logging.basicConfig(level=logging.INFO)

    if len(sys.argv) < 2:
        print("Usage: python graph.py <linkedin_url>")
        sys.exit(1)

    target_url = sys.argv[1]

    graph, checkpointer = compile_graph(interrupt_before_synthesis=False)

    initial_state: GraphState = {
        "executive_name": "",
        "linkedin_url": target_url,
        "raw_profile": {},
        "profile_text": "",
        "google_person_context": "",
        "claims": [],
        "verification_ledger": [],
        "refusal_log": [],
        "evidence_cache": {},
        "gaps_analysis": [],
        "approved": True,  # Auto-approve for CLI standalone runs
        "human_feedback": "",
        "final_diagnostic": "",
    }

    config = {"configurable": {"thread_id": "cli-run"}}
    result = graph.invoke(initial_state, config=config)
    print("\n" + "=" * 72)
    print(result.get("final_diagnostic", "No diagnostic generated."))
