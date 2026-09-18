# Executive Reputation Diagnostic Engine

A fact-checking pipeline designed to verify the professional claims of executives by extracting their LinkedIn profile data and cross-referencing it against the open web.

## Features

- **Profile Ingestion**: Scrapes LinkedIn profiles via Apify.
- **Claim Extraction**: Uses LLMs to break down paragraphs into verifiable claims.
- **Multi-Layer Evidence Gathering**:
  - **Google AI Mode (SerpAPI)**: For cross-verification.
  - **Tavily**: For independent deep search.
  - **Perplexity / Jina Reader (Agent Reach)**: For broad semantic search and deep web crawling.
- **Human-in-the-Loop (HITL)**: Execution pauses before the final report synthesis, allowing a human reviewer to approve or edit the AI's verdicts.
- **Diagnostic Report**: Synthesizes findings into a final, structured Markdown report.

## Tech Stack

- **Backend**: Python 3, FastAPI, Uvicorn
- **Orchestration**: LangGraph, LangChain
- **APIs Used**: Apify, SerpAPI, Tavily, Perplexity, NVIDIA NIM

## Quick Start

### 1. Prerequisites
Ensure you have Python 3 installed. Install the dependencies:
```bash
pip install -r requirements.txt
pip install -e ./Agent-Reach
```

### 2. Environment Variables
Create a `.env` file based on `.env.example` and add your API keys:
```env
NVIDIA_API_KEY=your_key
APIFY_API_TOKEN=your_token
SERPAPI_API_KEY=your_key
TAVILY_API_KEY=your_key
PERPLEXITY_API_KEY=your_key
```

### 3. Run the Server
Start the FastAPI server:
```bash
python app.py
```
The server will run on `http://localhost:8000`.

### 4. Deployment (Render)
This application is designed to be deployed as a background Web Service on platforms like Render. It includes a `/ping` endpoint to prevent the server from sleeping on free tiers when combined with a cron service.
