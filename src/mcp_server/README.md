# MCP Server

Model Context Protocol server that provides data retrieval tools.

## Tools

| Tool | Description                        | Input | Output                   |
|------|------------------------------------|--|--------------------------|
| `fetch_accounts` | Get all account ids                |  | List of account ids JSON |
| `calls_emails` | Get calls and emails of an account | `account_id: int` | Raw email JSON           |

## Architecture

Uses **FastMCP** with `streamable_http` transport:

```python
mcp = FastMCP(
    "account-data-mcp",
    streamable_http_path="/mcp",
    json_response=True,
    stateless_http=True,
)
```

## Files

```
mcp_server/
├── server.py           # MCP server implementation
├── requirements.txt
├── README.md
└── data/               # Account JSON files go here
    └── accounts/
        └── *.json
    └── topics.json      # topics of the calls/emails
```

## Data Format

Place account JSON files in the `data/accounts` directory:

```json
{
  "tenant_name": "modjo",
  "account_name": "Acme Corp",
  "account_id": 1,
  "calls": [
    {
      "date": "2024-01-15",
      "call_name": "Discovery Call",
      "transcript": "Sales Rep: Hi John...",
      "summary": "..."
    }
  ],
  "emails": [
    {
      "date": "2024-01-16",
      "subject": "Follow-up",
      "content": "Hi John,..."
    }
  ]
}
```

## Tool Responses

### `fetch_accounts`

```json
["account_1", "account_2"]
```

### `calls_emails`

```json
{
  "found": true,
  "account_name": "Acme Corp",
  "emails": [
    {
      "date": "2024-01-16",
      "subject": "Follow-up",
      "content": "..."
    }
  ],
    "calls": [
        {
        "date": "2024-01-15",
        "call_name": "Discovery Call",
        "transcript": "Sales Rep: Hi John...",
        "summary": "..."
        }
    ]
}
```

## Setup

```bash
uv pip install -e ".[mcp_server]"
```

## Environment Variables

```bash
export DATA_DIR="/path/to/data"      # optional, defaults to ./data
export MCP_SERVER_PORT=8002          # optional, defaults to 8002
```

## Running

```bash
python src/mcp_server/server.py
```

Server available at http://localhost:8002

- MCP endpoint: `POST /mcp`
- Health check: `GET /health` (if enabled)
