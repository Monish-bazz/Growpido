# =============================================================================
# graph.py
# LangGraph State Machine: Executive Reputation Diagnostic Pipeline
# Cyclic validation graph with HITL interrupt
# =============================================================================

import os
import json
import logging
import requests
from typing import TypedDict, List, Annotated
from operator import add as _list_add

from dotenv import load_dotenv
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_nvidia_ai_endpoints import ChatNVIDIA
from apify_client import ApifyClient
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver

from research_engine import HybridResearcher

load_dotenv()
logger = logging.getLogger(__name__)

# =============================================================================
# 1. State Definition
# =============================================================================

class GraphState(TypedDict):
    executive_name: str
    linkedin_url: str
    raw_profile: dict
    profile_text: str                                   # flattened text for LLM
    google_person_context: str                          # SerpAPI person research
    claims: List[dict]                                  # atomic factual claims
    verification_ledger: Annotated[List[dict], _list_add]  # verified / partial
    refusal_log: Annotated[List[dict], _list_add]          # unverified / contradicted
    evidence_cache: dict                                # gathered evidence per claim
    gaps_analysis: List[dict]                            # positioning gaps
    approved: bool                                       # binary approval gate
    human_feedback: str                                 # optional HITL notes
    final_diagnostic: str                               # rendered markdown report


# =============================================================================
# 2. Shared Resources
# =============================================================================

def _get_nim_llm(temperature: float = 0.1) -> ChatNVIDIA:
    """Return a ChatNVIDIA instance pointed at the NIM endpoint."""
    model_name = os.getenv("NIM_MODEL", "meta/llama-3.2-11b-vision-instruct")
    return ChatNVIDIA(
        model=model_name,
        nvidia_api_key=os.getenv("NVIDIA_API_KEY", ""),
        temperature=temperature,
        max_completion_tokens=2048,
        timeout=180,
    )


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
        run = client.actor("harvestapi/linkedin-profile-scraper").call(run_input=run_input)

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
    Use NVIDIA NIM (Llama 3.1) to decompose the profile into
    atomic, verifiable factual claims.
    """
    llm = _get_nim_llm(temperature=0.0)

    system_prompt = (
        "You are a meticulous due-diligence analyst. "
        "Your task is to extract specific, verifiable factual claims from "
        "the provided LinkedIn profile text AND the Web Intelligence context. Focus on:\n"
        "- Revenue/funding amounts and growth metrics\n"
        "- Specific roles, titles, and companies with dates\n"
        "- Awards, rankings, or recognition claims\n"
        "- Educational credentials (degrees, institutions, years)\n"
        "- Quantitative achievements (team size, user counts, percentage growth)\n"
        "- Partnership or client claims involving named entities\n\n"
        "Return ONLY a valid JSON array. Each element must have:\n"
        '  {"claim_text": "...", "context": "...", "category": "..."}\n\n'
        "Categories: funding, metric, role, education, award, partnership, other\n\n"
        "Extract up to a MAXIMUM of 10 claims. However, prioritize ONLY the most RICH, substantial, and impressive facts "
        "(e.g., massive funding rounds, notable acquisitions, major leadership roles). "
        "We want HONEST facts. Do NOT invent claims that are not explicitly stated in the provided text."
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

        if not raw:
            logger.error("NIM returned empty response for claim extraction. Model may not support this prompt type.")
            raise json.JSONDecodeError("Empty response from NIM", "", 0)

        # Handle cases where the LLM wraps output in markdown code fences
        if "```json" in raw:
            raw = raw.split("```json")[1].split("```")[0].strip()
        elif "```" in raw:
            raw = raw.split("```")[1].split("```")[0].strip()

        claims = json.loads(raw)
        if not isinstance(claims, list):
            claims = [claims]

        logger.info("Extracted %d claims from profile.", len(claims))
        return {"claims": claims}

    except (json.JSONDecodeError, IndexError) as exc:
        logger.error("Failed to parse claims from NIM response: %s", exc)
        logger.warning("Raw NIM response was: %r", locals().get('raw', 'N/A'))
        # Fallback: extract claims directly from profile_text line by line
        profile_text = state.get("profile_text", "")
        fallback_claims = []
        for line in profile_text.split("\n"):
            line = line.strip()
            if line and len(line) > 20 and any(kw in line.lower() for kw in [
                "role:", "education:", "certification:", "founded", "ceo", "chairman",
                "managed", "raised", "invested", "built", "led", "$", "%"
            ]):
                fallback_claims.append({
                    "claim_text": line,
                    "context": "Extracted from profile text (NIM extraction failed)",
                    "category": "other",
                })
        if not fallback_claims:
            fallback_claims = [{
                "claim_text": profile_text[:500],
                "context": "Full profile snippet (NIM extraction failed)",
                "category": "other",
            }]
        logger.info("Fallback extracted %d claims from profile text.", len(fallback_claims))
        return {"claims": fallback_claims}


# ---- Node 3: Batch Evidence Gathering ------------------------------------

def gather_evidence_batch(state: GraphState) -> dict:
    """
    Loop over all claims and gather evidence using HybridResearcher.
    """
    claims = state.get("claims", [])
    exec_name = state.get("executive_name", "Unknown")
    evidence_cache = {}

    logger.info("Gathering evidence for %d claims for %s", len(claims), exec_name)

    for idx, claim in enumerate(claims):
        claim_text = claim.get("claim_text", str(claim))
        logger.info("Researching claim %d/%d: %s", idx + 1, len(claims), claim_text[:80])
        evidence = _researcher.verify_claim(claim_text, exec_name)
        evidence_cache[str(idx)] = evidence

    return {"evidence_cache": evidence_cache}


# ---- Node 4: Batch Judgement ---------------------------------------------

def batch_judge_claims(state: GraphState) -> dict:
    """
    Send all claims and gathered evidence to the Judge in a single batch call.
    Uses Perplexity if enabled, otherwise NVIDIA NIM.
    """
    claims = state.get("claims", [])
    evidence_cache = state.get("evidence_cache", {})
    exec_name = state.get("executive_name", "Unknown")
    
    if not claims:
        return {"verification_ledger": [], "refusal_log": []}
        
    logger.info("Batch judging %d claims for %s", len(claims), exec_name)
    
    # Format claims and evidence for the prompt
    claims_list_text = ""
    for idx, claim in enumerate(claims):
        claim_text = claim.get("claim_text", str(claim))
        evidence = evidence_cache.get(str(idx), {})
        
        ev_summary = evidence.get("summary", "No evidence retrieved.")
        google_ai = evidence.get("google_ai_answer", "")
        tavily_answer = evidence.get("tavily_answer", "")
        pplx_answer = evidence.get("perplexity_answer", "")
        deep_ev = evidence.get("deep_evidence", [])
        source_pools = evidence.get("source_pools", {})
        
        claims_list_text += f"\n--- Claim {idx + 1} ---\n{claim_text}\n"
        claims_list_text += f"\n[SOURCE POOL A — Google AI Mode]:\n{google_ai}\n"
        if source_pools.get("pool_a"):
            claims_list_text += f"Pool A URLs: {', '.join(source_pools['pool_a'][:3])}\n"
        claims_list_text += f"\n[SOURCE POOL B — Tavily + Perplexity (Optional) + Independent Deep-Crawl]:\n{tavily_answer}\n{pplx_answer}\n"
        if source_pools.get("pool_b"):
            claims_list_text += f"Pool B URLs: {', '.join(source_pools['pool_b'][:3])}\n"
        for de in deep_ev:
            claims_list_text += f"Independent deep-crawled page ({de['url']}):\n{de['text'][:1000]}\n"
        
    system_prompt = (
        "You are a strict due-diligence investigator for a Dubai-based advisory firm (DIFC). "
        "You will be given a list of claims about an executive and evidence from TWO INDEPENDENT source pools.\n"
        "Pool A: Google AI Mode (Google's knowledge synthesis)\n"
        "Pool B: Tavily + Independent deep-crawled pages (completely separate search engine and sources)\n\n"
        "For EACH claim, evaluate evidence from BOTH pools and assign a verdict based on CORROBORATION:\n"
        "Verdicts allowed: Verified, Partially Verified, Unverified, Contradicted.\n"
        "RULES:\n"
        "1. Verified: BOTH Pool A AND Pool B independently confirm the claim with specific evidence. This is true dual-source corroboration.\n"
        "2. Partially Verified: Only ONE pool confirms the claim, or both pools mention it but disagree on key specifics (dates, numbers, amounts).\n"
        "3. Unverified: Neither pool provides credible evidence for the claim.\n"
        "4. Contradicted: Either pool explicitly contradicts the claim (different numbers, revoked credentials, etc.).\n\n"
        "IMPORTANT: A claim confirmed by only ONE source pool is NEVER 'Verified', even if that one source is very detailed. "
        "Single-source confirmation is always 'Partially Verified' at best.\n\n"
        "Return the results in JSON Lines (JSONL) format, with exactly one JSON object per line. Do NOT output a JSON array. Do not include commas between lines. Do not include markdown formatting (like ```json).\n"
        "Each line MUST be this exact structure:\n"
        "{\n"
        '  "index": (integer, 1-indexed, matching the input list),\n'
        '  "verdict": "Verified|Partially Verified|Unverified|Contradicted",\n'
        '  "confidence": "high|medium|low",\n'
        '  "corroboration_count": (0, 1, or 2 — how many independent pools confirmed),\n'
        '  "reasoning": "2-3 sentences explaining which pools confirmed/denied and why",\n'
        '  "primary_source_url": "Best source URL or empty string",\n'
        '  "source_name": "Publication name or empty string"\n'
        "}"
    )
    
    user_content = (
        f"Executive Name: {exec_name}\n"
        f"Please verify these {len(claims)} claims based on the provided evidence:\n"
        f"{claims_list_text}"
    )
    
    raw_content = ""
    ledger = []
    refusals = []
    
    try:
        if _researcher.use_perplexity and _researcher.pplx_key:
            logger.info("Using Perplexity as batch judge.")
            headers = {
                "Authorization": f"Bearer {_researcher.pplx_key}",
                "Content-Type": "application/json",
            }
            payload = {
                "model": "sonar",
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_content},
                ],
                "temperature": 0.1,
            }
            resp = requests.post(
                _researcher.PPLX_ENDPOINT,
                json=payload,
                headers=headers,
                timeout=180,
            )
            resp.raise_for_status()
            raw_content = resp.json()["choices"][0]["message"]["content"].strip()
        else:
            logger.info("Using NVIDIA NIM as batch judge.")
            llm = _get_nim_llm(temperature=0.0)
            response = llm.invoke([
                SystemMessage(content=system_prompt),
                HumanMessage(content=user_content),
            ])
            raw_content = response.content.strip()
            
        if "```json" in raw_content:
            raw_content = raw_content.split("```json")[1].split("```")[0].strip()
        elif "```" in raw_content:
            raw_content = raw_content.split("```")[1].split("```")[0].strip()
            
        results_array = []
        for line in raw_content.strip().split('\n'):
            line = line.strip().strip(',')
            if not line or not line.startswith('{'):
                continue
            try:
                obj = json.loads(line)
                results_array.append(obj)
            except json.JSONDecodeError as exc:
                logger.warning("Skipping malformed JSON line: %s", exc)
        
        
        for i, claim in enumerate(claims):
            claim_text = claim.get("claim_text", str(claim))
            category = claim.get("category", "unknown")
            evidence = evidence_cache.get(str(i), {})
            
            matching_res = None
            for res in results_array:
                if res.get("index") == i + 1:
                    matching_res = res
                    break
                    
            if not matching_res:
                matching_res = {
                    "verdict": "Unverified",
                    "confidence": "low",
                    "reasoning": "Batch judge did not return a result for this claim.",
                    "primary_source_url": "",
                    "source_name": "",
                }
                
            entry = {
                "index": i,
                "claim_text": claim_text,
                "category": category,
                "verdict": matching_res.get("verdict", "Unverified"),
                "confidence": matching_res.get("confidence", "low"),
                "corroboration_count": matching_res.get("corroboration_count", 0),
                "reasoning": matching_res.get("reasoning", ""),
                "primary_source_url": matching_res.get("primary_source_url", ""),
                "source_name": matching_res.get("source_name", ""),
                "all_citations": evidence.get("citations", []),
                "pool_a_sources": evidence.get("source_pools", {}).get("pool_a", []),
                "pool_b_sources": evidence.get("source_pools", {}).get("pool_b", []),
            }
            
            if entry["verdict"] in ("Verified", "Partially Verified"):
                ledger.append(entry)
            else:
                refusals.append(entry)
                
    except Exception as exc:
        logger.error("Batch verification failed: %s", exc)
        for i, claim in enumerate(claims):
            entry = {
                "index": i,
                "claim_text": claim.get("claim_text", str(claim)),
                "category": claim.get("category", "unknown"),
                "verdict": "Unverified",
                "confidence": "low",
                "reasoning": f"Verification failed due to error: {exc}",
                "primary_source_url": "",
                "source_name": "",
                "all_citations": [],
            }
            refusals.append(entry)

    return {
        "verification_ledger": ledger,
        "refusal_log": refusals
    }


# ---- Node 5: Gaps Analysis -----------------------------------------------

def analyze_gaps(state: GraphState) -> dict:
    """
    Identify the top 3 positioning gaps in the executive's public narrative.
    This is NOT a repeat of what failed verification — it's an analysis of
    what's structurally MISSING from the public record.
    """
    exec_name = state.get("executive_name", "Unknown")
    ledger = state.get("verification_ledger", [])
    refusals = state.get("refusal_log", [])
    profile_text = state.get("profile_text", "")
    google_ctx = state.get("google_person_context", "")

    verified_claims = "\n".join(
        f"- {e['claim_text']} ({e['verdict']}, corroboration: {e.get('corroboration_count', '?')})"
        for e in ledger
    )
    failed_claims = "\n".join(
        f"- {e['claim_text']} ({e['verdict']}): {e['reasoning']}"
        for e in refusals
    )

    llm = _get_nim_llm(temperature=0.2)

    system_prompt = (
        "You are a senior positioning strategist at a Dubai-based reputation advisory firm (DIFC). "
        "Your job is to identify the THREE most critical NARRATIVE GAPS in an executive's public record. "
        "A narrative gap is NOT the same as a failed verification. It is a structural weakness in how "
        "the executive's story is presented to the public.\n\n"
        "Types of gaps to look for:\n"
        "1. No independent third-party validation beyond self-reported claims\n"
        "2. No recent press, podcast, or media presence outside LinkedIn\n"
        "3. Inconsistent narrative across different time periods or platforms\n"
        "4. Absence of verifiable specificity where competitors in the same space ARE specific\n"
        "5. Claims that exist in only one place (the person's own profile) with zero external echo\n"
        "6. Missing credentials or track record gaps that a sophisticated investor would notice\n\n"
        "Return EXACTLY 3 gaps as a JSON array. Each element:\n"
        '{"gap_title": "...", "why_it_matters": "One sentence explaining the risk for a reputation advisory client", "severity": "Critical|High|Medium"}\n'
        "Do NOT include markdown formatting. Return only the raw JSON array."
    )

    user_content = (
        f"Executive: {exec_name}\n\n"
        f"LinkedIn Profile Summary:\n{profile_text[:2000]}\n\n"
        f"Web Intelligence Context:\n{google_ctx[:1500]}\n\n"
        f"VERIFIED CLAIMS:\n{verified_claims}\n\n"
        f"FAILED/REFUSED CLAIMS:\n{failed_claims}\n"
    )

    try:
        response = llm.invoke([
            SystemMessage(content=system_prompt),
            HumanMessage(content=user_content),
        ])
        raw = response.content.strip()

        # Parse JSON
        if "```json" in raw:
            raw = raw.split("```json")[1].split("```")[0].strip()
        elif "```" in raw:
            raw = raw.split("```")[1].split("```")[0].strip()

        gaps = json.loads(raw)
        if not isinstance(gaps, list):
            gaps = [gaps]
        gaps = gaps[:3]  # enforce max 3

        logger.info("Gaps analysis identified %d gaps for %s.", len(gaps), exec_name)
        return {"gaps_analysis": gaps}

    except Exception as exc:
        logger.error("Gaps analysis failed: %s", exc)
        return {"gaps_analysis": [
            {"gap_title": "Analysis unavailable", "why_it_matters": f"Gaps analysis failed: {exc}", "severity": "Medium"}
        ]}


# ---- Node 6: Diagnostic Synthesizer --------------------------------------

def synthesize_report(state: GraphState) -> dict:
    """
    Generate the final one-page diagnostic markdown report.
    Enforces Growpido house rules. Only runs if approved == True.
    """
    if not state.get("approved", False):
        logger.warning("Synthesis blocked: not approved by human reviewer.")
        return {"final_diagnostic": "ERROR: Report generation blocked. Human approval required."}

    exec_name = state.get("executive_name", "Unknown Executive")
    ledger = state.get("verification_ledger", [])
    refusals = state.get("refusal_log", [])
    gaps = state.get("gaps_analysis", [])
    human_fb = state.get("human_feedback", "")

    total = len(ledger) + len(refusals)
    verified_count = sum(1 for e in ledger if e["verdict"] == "Verified")
    partial_count = sum(1 for e in ledger if e["verdict"] == "Partially Verified")
    unverified_count = sum(1 for e in refusals if e["verdict"] == "Unverified")
    contradicted_count = sum(1 for e in refusals if e["verdict"] == "Contradicted")

    # Build structured input for the synthesizer
    verified_bullets = []
    for entry in ledger:
        src = ""
        if entry.get("primary_source_url"):
            src_name = entry.get("source_name") or "Source"
            src = f" [Source: {src_name}]({entry['primary_source_url']})"
        corr = entry.get('corroboration_count', 0)
        corr_label = f" [Corroborated by {corr} independent source{'s' if corr != 1 else ''}]" if corr else ""
        verified_bullets.append(
            f"- {entry['claim_text']} (Verdict: {entry['verdict']}{corr_label}){src}"
        )

    quarantine_bullets = []
    for entry in refusals:
        quarantine_bullets.append(
            f"- **{entry['verdict']}**: {entry['claim_text']}\n"
            f"  Reasoning: {entry['reasoning']}"
        )

    verified_section = "\n".join(verified_bullets) if verified_bullets else "No claims fully verified."
    quarantine_section = "\n".join(quarantine_bullets) if quarantine_bullets else "All claims passed verification."

    llm = _get_nim_llm(temperature=0.2)

    system_prompt = (
        "You are a senior analyst at a Dubai-based (DIFC) reputation advisory firm. "
        "Generate a one-page Executive Reputation Diagnostic brief.\n\n"
        "STRICT HOUSE RULES:\n"
        "- NEVER use em dashes. Use parentheses or colons instead.\n"
        "- NEVER use hashtags.\n"
        "- NEVER use filler words: delve, leverage, testament, tapestry, "
        "multifaceted, synergy, paradigm, holistic.\n"
        "- Every verified claim MUST include its source link in the format: "
        "[Source: Publication](URL)\n"
        "- Write in a tone suitable for a busy fund manager: direct, scannable, "
        "zero fluff.\n"
        "- Use clean Markdown formatting.\n\n"
        "REPORT STRUCTURE:\n"
        "1. Executive Overview (2-3 sentences positioning the executive)\n"
        "2. Authority Score (X/{total} claims verified)\n"
        "3. Verified Footprint (bullet list with source citations)\n"
        "4. Narrative Gaps (the three biggest gaps between claimed and verified record)\n"
        "5. The Quarantine Zone (claims refused with exact reasoning)\n\n"
        "The Quarantine Zone section title must start with the warning emoji."
    )

    # Build gaps section
    gaps_bullets = []
    for g in gaps:
        gaps_bullets.append(
            f"- **{g.get('gap_title', 'Gap')}** ({g.get('severity', 'Medium')}): "
            f"{g.get('why_it_matters', '')}"
        )
    gaps_section = "\n".join(gaps_bullets) if gaps_bullets else "No significant positioning gaps identified."

    user_content = (
        f"Executive: {exec_name}\n\n"
        f"VERIFIED CLAIMS ({verified_count} verified, {partial_count} partial):\n"
        f"{verified_section}\n\n"
        f"NARRATIVE GAPS (pre-analyzed by positioning strategist):\n"
        f"{gaps_section}\n\n"
        f"QUARANTINED CLAIMS ({unverified_count} unverified, {contradicted_count} contradicted):\n"
        f"{quarantine_section}\n\n"
        f"Total claims analyzed: {total}\n"
    )

    if human_fb:
        user_content += f"\nHuman reviewer notes: {human_fb}\n"

    try:
        response = llm.invoke([
            SystemMessage(content=system_prompt),
            HumanMessage(content=user_content),
        ])
        report = response.content.strip()
    except Exception as exc:
        logger.error("Synthesis NIM call failed: %s", exc)
        # Fallback: manual assembly
        report = _fallback_report(exec_name, total, ledger, refusals)

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
    workflow.add_node("gather_evidence_batch", gather_evidence_batch)
    workflow.add_node("batch_judge_claims", batch_judge_claims)
    workflow.add_node("analyze_gaps", analyze_gaps)
    workflow.add_node("synthesize_report", synthesize_report)

    # Linear edges
    workflow.add_edge("ingest_profile", "extract_claims")
    workflow.add_edge("extract_claims", "gather_evidence_batch")
    workflow.add_edge("gather_evidence_batch", "batch_judge_claims")
    workflow.add_edge("batch_judge_claims", "analyze_gaps")
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
