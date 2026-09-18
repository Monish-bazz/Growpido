# =============================================================================
# app.py
# FastAPI Backend: Human-in-the-Loop Executive Diagnostic Pipeline
# =============================================================================

import os
import uuid
import logging
from typing import Optional
from contextlib import asynccontextmanager

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, FileResponse
from pydantic import BaseModel

from graph import (
    compile_graph,
    GraphState,
    get_live_judge_progress,
    reset_live_judge_progress,
)

load_dotenv()
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


# =============================================================================
# State Management
# =============================================================================

# In-memory store for active pipeline sessions
# Production: replace with Redis or a database
_sessions: dict = {}
_graph = None
_checkpointer = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Compile graph on startup."""
    global _graph, _checkpointer
    logger.info("Compiling LangGraph pipeline...")
    _graph, _checkpointer = compile_graph(interrupt_before_synthesis=True)
    logger.info("Pipeline ready. HITL interrupt enabled before synthesis.")
    yield
    logger.info("Shutting down.")


app = FastAPI(
    title="NoCap.ai Track B: Executive Reputation Diagnostic Engine",
    description=(
        "Fact-checking pipeline for executive LinkedIn profiles. "
        "Extracts claims, verifies against the open web, and generates "
        "a strict diagnostic report."
    ),
    version="1.0.0",
    lifespan=lifespan,
)

@app.get("/ping")
async def ping():
    """Keep-alive endpoint for Render / UptimeRobot."""
    return {"status": "alive", "message": "Server is awake"}



# =============================================================================
# Request/Response Models
# =============================================================================

class ProcessRequest(BaseModel):
    linkedin_url: str


class ApproveRequest(BaseModel):
    human_feedback: Optional[str] = ""
    edits: Optional[list] = None  # optional ledger overrides


class SessionStatus(BaseModel):
    thread_id: str
    status: str
    phase: Optional[str] = None
    evidence_ready: Optional[int] = None
    claims_count: Optional[int] = None
    live_verdicts: Optional[list] = None
    judged_count: Optional[int] = None
    executive_name: Optional[str] = None
    headline: Optional[str] = None
    profile_text: Optional[str] = None
    linkedin_url: Optional[str] = None
    total_claims: Optional[int] = None
    verified_count: Optional[int] = None
    refused_count: Optional[int] = None
    claims: Optional[list] = None
    verification_ledger: Optional[list] = None
    refusal_log: Optional[list] = None
    gaps_analysis: Optional[list] = None
    approved: Optional[bool] = None
    final_diagnostic: Optional[str] = None
    raw_profile: Optional[dict] = None
    google_person_context: Optional[str] = None
    evidence_cache: Optional[dict] = None


# =============================================================================
# API Endpoints
# =============================================================================

from fastapi import BackgroundTasks

def run_pipeline(initial_state, config, thread_id):
    try:
        logger.info("[%s] Starting pipeline...", thread_id)
        _graph.invoke(initial_state, config=config)
        
        snapshot = _graph.get_state(config)
        state = snapshot.values
        ledger = state.get("verification_ledger", [])
        refusals = state.get("refusal_log", [])
        _sessions[thread_id]["status"] = "pending_review"
        logger.info("[%s] Pipeline paused. %d verified, %d refused. Awaiting human review.", thread_id, len(ledger), len(refusals))
        
    except Exception as exc:
        logger.error("[%s] Pipeline error: %s", thread_id, exc)
        _sessions[thread_id]["status"] = "error"

@app.post("/process", response_model=SessionStatus)
async def start_process(req: ProcessRequest, background_tasks: BackgroundTasks):
    """
    Start the diagnostic pipeline in the background.
    """
    thread_id = str(uuid.uuid4())[:8]

    initial_state: GraphState = {
        "executive_name": "",
        "linkedin_url": req.linkedin_url,
        "raw_profile": {},
        "profile_text": "",
        "google_person_context": "",
        "claims": [],
        "verification_ledger": [],
        "refusal_log": [],
        "evidence_cache": {},
        "gaps_analysis": [],
        "approved": False,
        "human_feedback": "",
        "final_diagnostic": "",
    }

    config = {"configurable": {"thread_id": thread_id}}
    _sessions[thread_id] = {"config": config, "status": "running"}

    background_tasks.add_task(run_pipeline, initial_state, config, thread_id)
    return await get_status(thread_id)


@app.get("/status/{thread_id}", response_model=SessionStatus)
async def get_status(thread_id: str):
    """
    Check the current state of a pipeline session.
    """
    if thread_id not in _sessions:
        raise HTTPException(status_code=404, detail="Session not found.")

    config = _sessions[thread_id]["config"]
    snapshot = _graph.get_state(config)
    state = snapshot.values

    ledger = state.get("verification_ledger", [])
    refusals = state.get("refusal_log", [])
    profile_text = state.get("profile_text", "")
    claims = state.get("claims", [])
    evidence_cache = state.get("evidence_cache", {})
    headline = ""
    for line in profile_text.split("\n"):
        if line.startswith("Headline:"):
            headline = line.replace("Headline:", "").strip()
            break

    # Derive a human-readable phase so the UI can narrate long-running work
    # (evidence gathering and judging happen inside single graph nodes and are
    # otherwise invisible to the client until the whole run pauses for review).
    session_status = _sessions[thread_id]["status"]
    evidence_ready = len(evidence_cache)

    # Live per-claim verdicts streamed from the judge node (partial while judging).
    live_map = get_live_judge_progress(thread_id)
    live_verdicts = list(live_map.values()) if live_map else []
    judged_count = len(live_verdicts)

    if session_status in ("pending_review", "completed", "error"):
        phase = session_status
    elif ledger or refusals:
        phase = "categorizing"
    elif claims and judged_count > 0:
        phase = "judging"
    elif claims and evidence_ready >= len(claims) and len(claims) > 0:
        phase = "judging"
    elif claims and evidence_ready > 0:
        phase = "gathering_evidence"
    elif claims:
        phase = "extracting_claims"
    elif profile_text and state.get("google_person_context"):
        phase = "researching_context"
    elif profile_text:
        phase = "profile_ready"
    else:
        phase = "ingesting_profile"

    return SessionStatus(
        thread_id=thread_id,
        status=session_status,
        phase=phase,
        evidence_ready=evidence_ready,
        claims_count=len(claims),
        live_verdicts=live_verdicts,
        judged_count=judged_count,
        executive_name=state.get("executive_name"),
        headline=headline,
        profile_text=profile_text,
        linkedin_url=state.get("linkedin_url"),
        total_claims=len(ledger) + len(refusals),
        verified_count=len(ledger),
        refused_count=len(refusals),
        claims=state.get("claims", []),
        verification_ledger=ledger,
        refusal_log=refusals,
        gaps_analysis=state.get("gaps_analysis", []),
        approved=state.get("approved", False),
        final_diagnostic=state.get("final_diagnostic"),
        raw_profile=state.get("raw_profile"),
        google_person_context=state.get("google_person_context"),
        evidence_cache=state.get("evidence_cache")
    )


@app.post("/approve/{thread_id}", response_model=SessionStatus)
async def approve_and_synthesize(thread_id: str, req: ApproveRequest):
    """
    Human approves (optionally edits) the verification ledger.
    Resumes the graph to generate the final diagnostic report.
    """
    if thread_id not in _sessions:
        raise HTTPException(status_code=404, detail="Session not found.")

    if _sessions[thread_id]["status"] == "completed":
        raise HTTPException(
            status_code=400, detail="Session already completed."
        )

    config = _sessions[thread_id]["config"]

    # Apply human feedback and set approved = True
    update = {"approved": True}
    if req.human_feedback:
        update["human_feedback"] = req.human_feedback

    if req.edits:
        snapshot = _graph.get_state(config)
        state = snapshot.values
        ledger = state.get("verification_ledger", [])
        refusals = state.get("refusal_log", [])
        
        all_entries = ledger + refusals
        new_ledger = []
        new_refusals = []
        
        # apply edits
        for entry in all_entries:
            for edit in req.edits:
                if entry.get("index") == edit.get("claim_id") or str(entry.get("index")) == str(edit.get("claim_id")):
                    entry["verdict"] = edit.get("verdict", entry["verdict"])
                    entry["reasoning"] = edit.get("reasoning", "Human Override")
                    # Re-bucket based on edit
                    if entry["verdict"] == "Verified":
                        entry["bucket"] = "Publishable as written"
                    elif entry["verdict"] == "Partially Verified":
                        entry["bucket"] = "Publishable with attribution"
                    else:
                        entry["bucket"] = "Blocked"
            
            if entry.get("bucket") in ["Publishable as written", "Publishable with attribution"]:
                new_ledger.append(entry)
            else:
                new_refusals.append(entry)
                
        update["verification_ledger"] = new_ledger
        update["refusal_log"] = new_refusals

    try:
        logger.info("[%s] Human approved. Updating state.", thread_id)
        _graph.update_state(config, update)
        logger.info("[%s] Resuming synthesis.", thread_id)
        # Resume from the interrupt (runs synthesize_report -> END)
        _graph.invoke(None, config=config)
    except Exception as exc:
        logger.error("[%s] Synthesis failed: %s", thread_id, exc)
        raise HTTPException(
            status_code=500, detail=f"Synthesis failed: {str(exc)}"
        )

    # Read final state
    snapshot = _graph.get_state(config)
    state = snapshot.values

    _sessions[thread_id]["status"] = "completed"

    ledger = state.get("verification_ledger", [])
    refusals = state.get("refusal_log", [])

    logger.info("[%s] Diagnostic report generated.", thread_id)

    return SessionStatus(
        thread_id=thread_id,
        status="completed",
        executive_name=state.get("executive_name"),
        total_claims=len(ledger) + len(refusals),
        verified_count=len(ledger),
        refused_count=len(refusals),
        verification_ledger=ledger,
        refusal_log=refusals,
        gaps_analysis=state.get("gaps_analysis", []),
        approved=True,
        final_diagnostic=state.get("final_diagnostic", ""),
        raw_profile=state.get("raw_profile"),
        google_person_context=state.get("google_person_context"),
    )


class FeedbackRequest(BaseModel):
    human_feedback: str = ""


@app.post("/feedback/{thread_id}")
async def save_feedback(thread_id: str, req: FeedbackRequest):
    """
    Save reviewer feedback WITHOUT triggering synthesis.
    This only updates the human_feedback field in state.
    """
    if thread_id not in _sessions:
        raise HTTPException(status_code=404, detail="Session not found.")
    config = _sessions[thread_id]["config"]
    _graph.update_state(config, {"human_feedback": req.human_feedback})
    logger.info("[%s] Feedback saved (synthesis NOT triggered).", thread_id)
    return {"status": "feedback_saved", "thread_id": thread_id}


# =============================================================================
# Static Files & Frontend
# =============================================================================

# Serve the frontend
static_dir = os.path.join(os.path.dirname(__file__), "static")
if os.path.isdir(static_dir):
    app.mount("/static", StaticFiles(directory=static_dir), name="static")


@app.get("/", response_class=HTMLResponse)
async def serve_frontend():
    """Serve the main HTML interface."""
    index_path = os.path.join(static_dir, "index.html")
    if os.path.exists(index_path):
        return FileResponse(index_path)
    return HTMLResponse(
        "<h1>NoCap.ai Track B</h1><p>Place index.html in static/ folder.</p>"
    )


# =============================================================================
# Entry Point
# =============================================================================

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "app:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
    )
