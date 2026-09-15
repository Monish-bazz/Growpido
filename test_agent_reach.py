import os
import sys
import requests
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Try to import Agent Reach
AGENT_REACH_PATH = os.path.join(os.path.dirname(__file__), "Agent-Reach")
if os.path.isdir(AGENT_REACH_PATH) and AGENT_REACH_PATH not in sys.path:
    sys.path.insert(0, AGENT_REACH_PATH)

try:
    from agent_reach.channels.web import WebChannel
    WEB_CHANNEL = WebChannel()
    AGENT_REACH_AVAILABLE = True
    print("Success: Agent Reach WebChannel loaded.")
except ImportError as e:
    WEB_CHANNEL = None
    AGENT_REACH_AVAILABLE = False
    print(f"Failed to load Agent Reach WebChannel: {e}")

# Test URLs from the logs
URLS_TO_TEST = [
    "https://www.moet.gov.ae/en/w/divyank-turakhia",
    "https://gulfbusiness.com/top-100-indians-2025-page/div-turakhia/",
    "https://en.wikipedia.org/wiki/Divyank_Turakhia" 
    "https://finance.yahoo.com/news/media-net-acquired-chinese-consortium-120000347.html" # A known working URL
]

def test_jina_direct(url):
    print(f"\n--- Testing direct Jina Reader (Fallback) for: {url} ---")
    try:
        resp = requests.get(
            f"https://r.jina.ai/{url}",
            headers={"User-Agent": "Mozilla/5.0", "Accept": "text/plain"},
            timeout=20
        )
        print(f"Status Code: {resp.status_code}")
        if resp.status_code == 200:
            print(f"Success! Preview: {resp.text[:200].strip()}...")
        else:
            print(f"Failed. Error: {resp.text[:200].strip()}")
    except Exception as exc:
        print(f"Exception: {exc}")

def test_agent_reach(url):
    print(f"\n--- Testing Agent Reach WebChannel for: {url} ---")
    if not AGENT_REACH_AVAILABLE:
        print("Agent Reach not available, skipping.")
        return
    try:
        full_text = WEB_CHANNEL.read(url)
        print(f"Success! Length: {len(full_text)} chars. Preview: {full_text[:200].strip()}...")
    except Exception as exc:
        print(f"Failed. Exception: {exc}")

if __name__ == "__main__":
    print("Starting tests...\n")
    for u in URLS_TO_TEST:
        test_agent_reach(u)
        test_jina_direct(u)
