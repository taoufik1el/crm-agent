"""Agent API Server.

FastAPI backend that exposes the agent via HTTP endpoints.
Supports both streaming and non-streaming responses.
"""

import json
import os
from collections.abc import AsyncGenerator
from pathlib import Path
from typing import Any

import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from main import run_agent
from pydantic import BaseModel

app = FastAPI(title="Account Intelligence API", version="1.0.0")

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class QueryRequest(BaseModel):
    """Request model for agent queries."""

    account_id: int
    user_query: str


class QueryResponse(BaseModel):
    """Response model for agent queries."""

    response: str


def load_accounts_from_data() -> list[dict[str, str | int]]:
    """Load accounts from the MCP server data directory."""
    data_dir = Path(
        os.getenv("DATA_DIR", Path(__file__).parent.parent / "mcp_server" / "data")
    )
    accounts: list[dict[str, str | int]] = []

    if not data_dir.exists():
        return accounts

    for file_path in sorted(data_dir.glob("account_*.json")):
        try:
            # Extract account_id from filename (e.g., account_1.json -> 1)
            account_id = int(file_path.stem.replace("account_", ""))
            with open(file_path) as f:
                data = json.load(f)
                accounts.append(
                    {
                        "id": account_id,
                        "name": data.get("account_name", f"Account {account_id}"),
                    }
                )
        except (OSError, json.JSONDecodeError, ValueError):
            continue

    return accounts


@app.get("/health")
async def health_check() -> dict[str, str]:
    """Health check endpoint."""
    return {"status": "healthy"}


@app.get("/api/accounts")
async def get_accounts() -> dict[str, list[dict[str, str | int]]]:
    """Get list of available accounts for the dropdown."""
    accounts = load_accounts_from_data()
    return {"accounts": accounts}


@app.post("/api/query", response_model=QueryResponse)
async def query_agent(request: QueryRequest) -> QueryResponse:
    """Query the agent with a user question about an account.

    Returns the agent's response (non-streaming).
    """
    try:
        response = run_agent(
            user_query=request.user_query, account_id=request.account_id
        )

        return QueryResponse(response=response)

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) from e


@app.post("/api/query/stream")
async def query_agent_stream(request: QueryRequest) -> StreamingResponse:
    """Query the agent with streaming response.

    Returns a Server-Sent Events stream.
    """

    async def generate() -> AsyncGenerator[str, Any]:
        try:
            # Run the agent to get the full response
            response = run_agent(
                user_query=request.user_query, account_id=request.account_id
            )

            # Stream the response in chunks
            # In production, integrate with LLM's native streaming
            chunk_size = 10
            for i in range(0, len(response), chunk_size):
                chunk = response[i : i + chunk_size]
                yield f"data: {chunk}\n\n"

            yield "data: [DONE]\n\n"

        except Exception as e:
            yield f"data: [ERROR] {str(e)}\n\n"

    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "Connection": "keep-alive"},
    )


if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8001)
