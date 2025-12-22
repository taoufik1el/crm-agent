# Agent Service

FastAPI backend that exposes the LangGraph agent.

## Architecture


![Graph](agent_graph.png)

### Nodes

1. **mcp** (`nodes/mcp.py`)
   - deterministic node
   - Interacts with MCP server to fetch transcripts and emails
   - Tools:
     - `transcripts`: Fetches all transcripts for a given account
     - `emails`: Fetches all emails for a given account

2. **planner** (`nodes/planner.py`)
   - LLM-powered node
   - Creates a plan to filter irrelevant calls/emails based on user query


3. **plan_executor** (`nodes/plan_executor.py`)
   - deterministic node
   - Executes the plan created by the planner node by calling the appropriate tools
   - Aggregates results from multiple tool calls
   - Builds context for final answer generation

4. **final_answer** (`nodes/final_answer.py`)
   - LLM-powered node
   - Generates the final answer based on the user query and aggregated context from plan_executor

## Files

```
agent/
├── api.py              # FastAPI server (run this)
├── main.py             # Agent entry point
├── graph.py            # LangGraph workflow definition
├── config.py           # Configuration, state, types
└── nodes/
    ├── __init__.py
    ├── mcp.py           # MCP interaction node
    ├── planner.py       # Planner node
    ├── plan_executor.py # Plan executor node
    └── final_answer.py  # Final answer generation node
```

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/health` | Health check |
| GET | `/api/accounts` | List available accounts |
| POST | `/api/query` | Query agent (non-streaming) |
| POST | `/api/query/stream` | Query agent (streaming SSE) |

### Request

```json
POST /api/query
{
    "account_id": 1,
    "user_query": "What are the main pain points discussed?"
}
```

### Response

```json
{
    "response": "Based on the transcripts and emails..."
}
```

## Setup

```bash
uv pip install -e ".[agent]"
```

## Environment Variables

```bash
export OPENAI_API_KEY="your-key"
export GOOGLE_API_KEY="your-key
export MCP_SERVER_URL="http://localhost:8002/mcp"  # optional, this is default
```

## Running

```bash
python src/agent/api.py
```

API available at http://localhost:8001
