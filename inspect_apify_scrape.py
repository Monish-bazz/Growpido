"""
inspect_apify_scrape.py
Script to inspect and display everything Apify scraped for a LinkedIn profile.
Target URL: https://www.linkedin.com/in/divyankturakhia/
"""

import os
import sys
import json
from dotenv import load_dotenv
from apify_client import ApifyClient

load_dotenv()

DEFAULT_URL = "https://www.linkedin.com/in/divyankturakhia/"
OUTPUT_JSON = "divyankturakhia_scraped.json"

def print_banner(title: str):
    print("\n" + "=" * 70)
    print(f"  {title}")
    print("=" * 70)

def display_scraped_profile(data: dict):
    print_banner("1. BASIC PROFILE INFORMATION")
    first_name = data.get("firstName", "")
    last_name = data.get("lastName", "")
    full_name = f"{first_name} {last_name}".strip() or data.get("name", "N/A")
    
    print(f"Full Name:        {full_name}")
    print(f"Headline:         {data.get('headline', 'N/A')}")
    print(f"LinkedIn URL:     {data.get('linkedinUrl', DEFAULT_URL)}")
    print(f"Location:         {data.get('location', 'N/A')}")
    print(f"Followers:        {data.get('followerCount', 'N/A')}")
    print(f"Connections:      {data.get('connectionsCount', 'N/A')}")
    print(f"Verified Profile: {data.get('verified', False)}")
    pic = data.get("profilePicture")
    pic_url = pic.get("url") if isinstance(pic, dict) else (pic or data.get("photo", "N/A"))
    print(f"Profile Picture:  {pic_url}")

    print_banner("2. ABOUT / SUMMARY")
    about = data.get("about") or data.get("summary") or "No about/summary found."
    print(str(about).strip())

    print_banner(f"3. WORK EXPERIENCE ({len(data.get('experience', []))} Positions Scraped)")
    experiences = data.get("experience", data.get("positions", []))
    if not experiences:
        print("No experience entries found.")
    for idx, exp in enumerate(experiences, 1):
        pos = exp.get("position") or exp.get("title") or "N/A"
        company = exp.get("companyName") or exp.get("company") or "N/A"
        
        # Format dates
        start_dict = exp.get("startDate")
        end_dict = exp.get("endDate")
        start_txt = start_dict.get("text", "") if isinstance(start_dict, dict) else str(start_dict or "")
        end_txt = end_dict.get("text", "") if isinstance(end_dict, dict) else str(end_dict or "")
        date_str = f"{start_txt} - {end_txt}".strip(" -")
        duration = exp.get("duration", "")
        if duration and date_str:
            date_display = f"{date_str} ({duration})"
        else:
            date_display = date_str or duration or "Date not specified"

        raw_desc = exp.get("description") or ""
        desc = raw_desc.strip()
        print(f"\n  [{idx}] {pos} at {company}")
        print(f"      Duration: {date_display}")
        if desc:
            snippet = (desc[:160] + "...") if len(desc) > 160 else desc
            print(f"      Details:  {snippet}")

    print_banner("4. EDUCATION & CERTIFICATIONS")
    educations = data.get("education", data.get("educations", []))
    print(f"Educations ({len(educations)}):")
    if educations:
        for edu in educations:
            school = edu.get("schoolName", edu.get("school", "N/A"))
            degree = edu.get("degreeName", edu.get("degree", ""))
            field = edu.get("fieldOfStudy", edu.get("field", ""))
            print(f"  - {degree} {field} from {school}".strip())
    else:
        print("  (None listed)")

    certs = data.get("certifications", [])
    print(f"\nCertifications ({len(certs)}):")
    if certs:
        for cert in certs:
            print(f"  - {cert.get('name', 'N/A')}")
    else:
        print("  (None listed)")

    print_banner("5. ALL RAW JSON KEYS RETURNED BY APIFY")
    print(", ".join(data.keys()))
    print("\n" + "=" * 70)


def scrape_or_load(url: str = DEFAULT_URL, force_live: bool = False) -> dict:
    # If already downloaded or cached, load immediately unless --live flag is passed
    cached_files = ["divyank_scraped.json", OUTPUT_JSON]
    for cache_path in cached_files:
        if os.path.exists(cache_path) and not force_live:
            print(f"[*] Found existing scraped data in '{cache_path}'. Loading...")
            with open(cache_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            # Ensure saved to OUTPUT_JSON as well
            if cache_path != OUTPUT_JSON:
                with open(OUTPUT_JSON, "w", encoding="utf-8") as out:
                    json.dump(data, out, indent=2)
            return data

    # Live scrape via Apify
    token = os.getenv("APIFY_API_TOKEN", "").strip()
    if not token:
        print("[ERROR] APIFY_API_TOKEN not found in .env file.")
        sys.exit(1)

    print(f"[*] Connecting to Apify to scrape live: {url}")
    client = ApifyClient(token)
    run_input = {
        "profileScraperMode": "Profile details no email ($4 per 1k)",
        "queries": [url],
    }

    print("[*] Calling Apify actor 'harvestapi/linkedin-profile-scraper'...")
    run = client.actor("harvestapi/linkedin-profile-scraper").call(run_input=run_input)
    
    print(f"[*] Scrape complete! Fetching items from dataset {run.default_dataset_id}...")
    items = list(client.dataset(run.default_dataset_id).iterate_items())

    if not items:
        print("[!] No items returned by Apify for this URL.")
        return {}

    data = items[0]
    with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
    print(f"[+] Successfully saved full raw output to '{OUTPUT_JSON}'.")
    return data


if __name__ == "__main__":
    force_live = "--live" in sys.argv
    url = sys.argv[1] if len(sys.argv) > 1 and not sys.argv[1].startswith("--") else DEFAULT_URL
    
    data = scrape_or_load(url=url, force_live=force_live)
    if data:
        display_scraped_profile(data)
        print(f"[INFO] Full raw JSON saved to: {os.path.abspath(OUTPUT_JSON)}")
        print("Tip: Run with --live to trigger a fresh new Apify scrape call if needed:")
        print("     python inspect_apify_scrape.py --live")
