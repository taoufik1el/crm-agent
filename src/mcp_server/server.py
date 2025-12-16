"""MCP Server.

Provides tools for retrieving account data:
- transcripts: Get call transcripts for an account
- emails: Get emails for an account

Uses FastMCP with streamable_http transport.
"""

import json
import logging
import os
from pathlib import Path
from typing import Any

import uvicorn
from mcp.server.fastmcp import FastMCP
from starlette.applications import Starlette


# Suppress noisy ClosedResourceError logs from MCP's stateless HTTP transport
# These are expected in stateless mode and don't affect functionality
class ClosedResourceFilter(logging.Filter):
    """Filter to suppress ClosedResourceError logs."""

    def filter(self, record: Any) -> bool:
        """Filter out specific log messages."""
        # Filter out "Error in message router" with ClosedResourceError
        msg = record.getMessage()
        if "Error in message router" in msg:
            return False
        if "Terminating session" in msg:
            return False
        # Also check exception info
        if record.exc_info and record.exc_info[0]:
            exc_name = record.exc_info[0].__name__
            if exc_name == "ClosedResourceError":
                return False
        return True


logging.getLogger("mcp.server.streamable_http").addFilter(ClosedResourceFilter())


SERVICE_NAME = "account-data-mcp"

# Data directory
DATA_DIR = Path(os.getenv("DATA_DIR", Path(__file__).parent / "data"))

# Initialize MCP server
mcp = FastMCP(
    SERVICE_NAME,
    host="127.0.0.1",
    port=8002,
    streamable_http_path="/mcp",
    json_response=True,
    stateless_http=True,
)


def load_account_data(account_id: int) -> dict[str, Any] | None:
    """Load account data from JSON file.

    Account files are named account_<id>.json (e.g., account_1.json).
    """
    if not DATA_DIR.exists():
        return None

    # Try to load directly by account_id from filename
    file_path = DATA_DIR / f"account_{account_id}.json"
    if file_path.exists():
        try:
            with open(file_path) as f:
                data = json.load(f)
                data["account_id"] = account_id  # Add account_id to the data
                return data  # type: ignore[no-any-return]
        except (OSError, json.JSONDecodeError):
            return None

    return None


def list_all_accounts() -> list[dict[str, str | int]]:
    """List all available accounts from data files.

    Returns a list of {id, name} dicts for each account.
    """
    accounts: list[dict[str, str | int]] = []
    if not DATA_DIR.exists():
        return accounts

    for file_path in sorted(DATA_DIR.glob("account_*.json")):
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


# ----- Tools -----


@mcp.tool(
    name="transcripts",
    description="Retrieve all call transcripts for an account. Returns raw transcript data as JSON.",
)
async def get_transcripts(account_id: int) -> dict[str, Any]:
    """Get call transcripts for an account."""
    account_data = load_account_data(account_id)

    if account_data is None:
        return {
            "found": False,
            "transcripts": None,
            "error": f"No data found for account_id: {account_id}",
        }

    calls = account_data.get("calls", [])

    if not calls:
        return {
            "found": False,
            "transcripts": None,
            "error": "No transcripts found for this account",
        }

    # Return raw transcripts without summaries
    transcripts: list[dict[str, Any]] = []
    for call in calls:
        transcripts.append(
            {
                "date": call.get("date"),
                "call_name": call.get("call_name"),
                "transcript": call.get("transcript"),
            }
        )

    return {
        "found": True,
        "account_name": account_data.get("account_name"),
        "transcripts": transcripts,
    }


@mcp.tool(
    name="emails",
    description="Retrieve all emails for an account. Returns raw email data as JSON.",
)
async def get_emails(account_id: int) -> dict[str, Any]:
    """Get emails for an account."""
    account_data = load_account_data(account_id)

    if account_data is None:
        return {
            "found": False,
            "emails": None,
            "error": f"No data found for account_id: {account_id}",
        }

    emails = account_data.get("emails", [])

    if not emails:
        return {
            "found": False,
            "emails": None,
            "error": "No emails found for this account",
        }

    # Return raw emails
    raw_emails: list[dict[str, Any]] = []
    for email in emails:
        raw_emails.append(
            {
                "date": email.get("date"),
                "subject": email.get("subject"),
                "content": email.get("content"),
            }
        )

    return {
        "found": True,
        "account_name": account_data.get("account_name"),
        "emails": raw_emails,
    }


# ----- App -----


def create_app() -> Starlette:
    """Create the Starlette app with MCP routes."""
    return mcp.streamable_http_app()


app = create_app()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    port = int(os.getenv("MCP_SERVER_PORT", 8002))
    logging.info(f"Starting MCP Server on port {port}")
    logging.info(f"Data directory: {DATA_DIR}")
    logging.info(f"MCP endpoint: http://localhost:{port}/mcp")

    uvicorn.run(app, host="127.0.0.1", port=port)
