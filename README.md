# ML Engineer Technical Test



## 📐 Architecture Overview

The system consists of three services:

```
┌─────────────────┐     HTTP      ┌─────────────────┐     MCP       ┌─────────────────┐
│    Web App      │ ────────────► │     Agent       │ ────────────► │   MCP Server    │
│   (Streamlit)   │               │    (FastAPI)    │               │   (FastMCP)     │
│   Port 8501     │ ◄──────────── │   Port 8001     │ ◄──────────── │   Port 8002     │
└─────────────────┘               └─────────────────┘               └─────────────────┘
       UI                           LangGraph Agent                   Data Tools
```

### 1. Web App (`/webapp`)
- Streamlit frontend
- Account selection dropdown
- Query input field
- Supports both streaming and non-streaming responses

### 2. Agent (`/agent`)
- FastAPI backend exposing the agent
- LangGraph-based agent with two nodes:
  - **Supervisor**: Deterministic orchestrator that fetches data via MCP tools
  - **Final Answer**: GPT-4o-mini generates the response based on retrieved context
- Uses a **fixed plan** to fetch all transcripts and emails

### 3. MCP Server (`/mcp_server`)
- FastMCP server using `streamable_http` transport
- Provides two tools:
  - `transcripts`: Retrieves call transcripts for an account
  - `emails`: Retrieves emails for an account

---


## 🚀 Quick Start

### Prerequisites

- Python 3.12+
- [uv](https://docs.astral.sh/uv/)
- uv pip install -e ".[agent,mcp_server,webapp]"

### 1. Set Environment Variables

```bash
# For OpenAI (default)
export OPENAI_API_KEY="your-openai-api-key"

# For Google Gemini (alternative)
export GOOGLE_API_KEY="your-google-api-key"
export LLM_PROVIDER="google"  # Set to "google" to use Gemini, defaults to "openai"
```

### 2. Run All Services

Open **three terminals**:

**Terminal 1 - MCP Server:**
```bash
python src/mcp_server/server.py
# Running on http://localhost:8002
```

**Terminal 2 - Agent API:**
```bash
python src/agent/api.py
# Running on http://localhost:8001
```

**Terminal 3 - Web App:**
```bash
streamlit run src/webapp/app.py
# Opens http://localhost:8501
```

### 3. Test the System

1. Open http://localhost:8501 in your browser
2. Select an account from the dropdown
3. Enter a question (e.g., "What are the main pain points discussed?")
4. Click "Ask Agent"


## Docker Setup (Optional)

Alternatively, you can run the entire system using Docker Compose.

1. Build and start the services:

```bash
docker-compose up --build
```
2. Access the web app at http://localhost:8003

---

## 📊 Dataset Format

Account data files are JSON with this structure:

```json
{
  "tenant_name": "string",
  "account_name": "string",
  "account_id": 123,
  "calls": [
    {
      "date": "2024-01-15",
      "call_name": "Discovery Call",
      "transcript": "...",
      "summary": "...",
      "crm_fields": [{"key": "value"}]
    }
  ],
  "emails": [
    {
      "date": "2024-01-16",
      "subject": "Follow-up",
      "content": "..."
    }
  ]
}
```

---

## 📁 Project Structure

```
ml-eng-hiring/
.
├── pyproject.toml              # Project metadata & dependencies
├── docker-compose.yml          # Multi-service orchestration
├── docker/                     # Dockerfiles for each service
│   ├── agent.Dockerfile
│   ├── mcp_server.Dockerfile
│   └── webapp.Dockerfile
├── scripts/                    # Evaluation & automation scripts
│   ├── aggregate_metrics.py
│   ├── fill_topics.py
│   ├── llm_as_judge.py
│   └── run_agent.py
└── src/
    ├── agent/                  # LLM agent implementation
    │   ├── api.py               # FastAPI server
    │   ├── main.py              # Agent entry point
    │   ├── graph.py             # LangGraph definition
    │   ├── config.py            # Configuration & state
    │   ├── llm_utils.py         # LLM helpers
    │   ├── agent_graph.png      # Agent graph visualization
    │   ├── nodes/               # Agent graph nodes
    │   │   ├── planner.py
    │   │   ├── plan_executer.py
    │   │   ├── mcp.py
    │   │   └── final_answer.py
    │   └── README.md
    ├── mcp_server/              # MCP data server
    │   ├── server.py            # MCP server implementation
    │   ├── data/
    │   │   ├── accounts/        # Account JSON files
    │   │   └── topics.json
    │   └── README.md
    └── webapp/                  # Streamlit UI
        ├── app.py               # Web application
        └── README.md
```
