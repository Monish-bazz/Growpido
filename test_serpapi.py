import os
import serpapi
from dotenv import load_dotenv

load_dotenv()

SERPAPI_API_KEY = os.getenv("SERPAPI_API_KEY")

def test_serpapi_without_linkedin(executive_name, company=""):
    if not SERPAPI_API_KEY:
        print("Error: SERPAPI_API_KEY not set in .env")
        return

    client = serpapi.Client(api_key=SERPAPI_API_KEY)
    
    # Base query
    query = f"Who is {executive_name}"
    if company:
        query += f" {company}"
    query += "? Background, career, achievements, and public record."
    
    # Crucial: exclude linkedin to avoid duplicate/redundant info
    query += " -site:linkedin.com"

    print(f"Executing query: '{query}'\n")

    results = client.search({
        "engine": "google_ai_mode",
        "q": query,
        "hl": "en",
        "gl": "ae",
        "google_domain": "google.com",
    })

    # Extract AI Answer
    ai_answer = ""
    text_blocks = results.get("text_blocks", [])
    if text_blocks:
        parts = []
        for block in text_blocks:
            snippet = block.get("snippet", "")
            if snippet:
                parts.append(snippet)
        ai_answer = " ".join(parts)
    
    if not ai_answer:
        ai_answer = results.get("reconstructed_markdown", "")
        
    print("--- AI Answer ---")
    print(ai_answer if ai_answer else "No AI answer found.")
    print("\n--- References ---")
    for ref in results.get("references", []):
        print(f"- {ref.get('title')}: {ref.get('link')}")

if __name__ == "__main__":
    test_serpapi_without_linkedin("Divyank Turakhia")
