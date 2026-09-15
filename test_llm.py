"""
test_llm.py
Diagnostic test script for NVIDIA NIM LLM used in the Executive Reputation Engine.
Tests API connectivity, latency, and structured JSON claim extraction capability.
"""

import os
import sys
import time
import json
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

def print_header(title: str):
    print("\n" + "=" * 60)
    print(f"  {title}")
    print("=" * 60)

def test_nim_llm(model_name: str = None):
    if not model_name:
        model_name = os.getenv("NIM_MODEL", "meta/llama-3.2-11b-vision-instruct")
    api_key = os.getenv("NVIDIA_API_KEY", "").strip()

    print_header("NVIDIA NIM LLM Diagnostic Test")
    print(f"[1] Target Model: {model_name}")
    print(f"[2] NVIDIA_API_KEY present: {'YES (starts with ' + api_key[:8] + '...)' if api_key else 'NO - MISSING!'}")

    if not api_key:
        print("\n[ERROR] NVIDIA_API_KEY is not set in .env! Please set it before testing.")
        return False

    try:
        from langchain_nvidia_ai_endpoints import ChatNVIDIA
        from langchain_core.messages import SystemMessage, HumanMessage
    except ImportError as e:
        print(f"\n[ERROR] Failed to import langchain_nvidia_ai_endpoints: {e}")
        return False

    # Initialize LLM
    print(f"\n[3] Initializing ChatNVIDIA (timeout=30s)...")
    llm = ChatNVIDIA(
        model=model_name,
        nvidia_api_key=api_key,
        temperature=0.1,
        max_completion_tokens=1024,
    )

    # Test 1: Simple Hello / Ping
    print("\n--- Test 1: Basic Generation & Ping ---")
    start_time = time.time()
    try:
        response = llm.invoke([
            HumanMessage(content="Hello! Respond in exactly one short sentence confirming you are working.")
        ])
        elapsed = time.time() - start_time
        reply_text = response.content.strip() if hasattr(response, "content") else str(response).strip()
        print(f"Status: SUCCESS (Response time: {elapsed:.2f}s)")
        print(f"Response:\n  \"{reply_text}\"")
    except Exception as e:
        elapsed = time.time() - start_time
        print(f"Status: FAILED after {elapsed:.2f}s")
        print(f"Error: {e}")
        return False

    # Test 2: Structured JSON Extraction Test (Simulating Claim Extraction in graph.py)
    print("\n--- Test 2: Structured JSON Claim Extraction ---")
    system_prompt = (
        "You are an expert fact-checking research analyst.\n"
        "Extract 1 verifiable factual claim from the text below.\n"
        "Return ONLY a valid JSON array of objects. No markdown ticks, no preamble, no thinking.\n"
        '[{"claim": "...", "category": "experience|metric|education", "target": "entity", "importance": "high"}]'
    )
    user_prompt = "Profile text: Fadi Ghandour founded Aramex in 1982 in Amman, Jordan."

    start_time = time.time()
    try:
        json_response = llm.invoke([
            SystemMessage(content=system_prompt),
            HumanMessage(content=user_prompt)
        ])
        elapsed = time.time() - start_time
        raw_output = json_response.content.strip() if hasattr(json_response, "content") else str(json_response).strip()
        print(f"Response time: {elapsed:.2f}s")
        print(f"Raw Output:\n{raw_output}\n")

        # Strip markdown fences if present
        clean_json = raw_output
        if clean_json.startswith("```"):
            lines = clean_json.split("\n")
            lines = [l for l in lines if not l.startswith("```")]
            clean_json = "\n".join(lines).strip()

        parsed = json.loads(clean_json)
        print(f"JSON Parsing: SUCCESS")
        print(f"Parsed object count: {len(parsed) if isinstance(parsed, list) else 1}")
        print(f"Parsed Content: {json.dumps(parsed, indent=2)}")
        print("\n" + "=" * 60)
        print("  ALL LLM TESTS PASSED SUCCESSFULLY!")
        print("=" * 60)
        return True
    except json.JSONDecodeError as jde:
        print(f"JSON Parsing FAILED: {jde}")
        print(f"Output was not clean JSON: {raw_output}")
        return False
    except Exception as e:
        elapsed = time.time() - start_time
        print(f"Extraction test FAILED after {elapsed:.2f}s: {e}")
        return False

if __name__ == "__main__":
    target_model = sys.argv[1] if len(sys.argv) > 1 else None
    success = test_nim_llm(target_model)
    sys.exit(0 if success else 1)
