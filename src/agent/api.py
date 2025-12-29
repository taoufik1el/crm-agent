"""Agent API Server.

FastAPI backend that exposes the agent via HTTP endpoints.
Supports both streaming and non-streaming responses.
"""

import json
import os
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from typing import Any

import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from agent.graph import create_agent_graph
from agent.main import run_agent, stream_agent
from agent.nodes.mcp import mcp_client

host = os.getenv("APP_HOST", "127.0.0.1")
port = int(os.getenv("APP_PORT", 8001))


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None]:  # type: ignore[type-arg]
    """Lifespan context manager to initialize agent graphs."""
    app.state.agent = create_agent_graph(streaming=False)
    app.state.streaming_agent = create_agent_graph(streaming=True)
    yield


app = FastAPI(title="Account Intelligence API", version="1.0.0", lifespan=lifespan)

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


@app.get("/health")
async def health_check() -> dict[str, str]:
    """Health check endpoint."""
    return {"status": "healthy"}


@app.get("/api/accounts")
async def get_accounts() -> dict[str, list[dict[str, str | int]]]:
    """Get list of available accounts for the dropdown."""
    accounts = mcp_client.call_tool("fetch_accounts", arguments={})
    return {"accounts": json.loads(accounts)}


@app.post("/api/query", response_model=QueryResponse)
async def query_agent(request: QueryRequest) -> QueryResponse:
    """Query the agent with a user question about an account.

    Returns the agent's response (non-streaming).
    """
    try:
        agent = app.state.agent
        response = run_agent(
            agent=agent, user_query=request.user_query, account_id=request.account_id
        )

        return QueryResponse(response=response)

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) from e


@app.post("/api/query/stream")
async def query_agent_stream(request: QueryRequest) -> StreamingResponse:
    """Query the agent with streaming response.

    Returns a Server-Sent Events stream.
    """
    agent = app.state.streaming_agent

    async def generate() -> AsyncGenerator[str, Any]:
        try:
            async for chunk in stream_agent(
                agent=agent,
                user_query=request.user_query,
                account_id=request.account_id,
            ):
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
    uvicorn.run(app, host=host, port=port)
    uvicorn.run(
        app,  # full module path
        host=host,
        port=port,
        log_level="info",
        reload=True,  # optional, dev mode
        loop="asyncio",
        http="h11",  # prevents HTTP/2 buffering issues
    )
