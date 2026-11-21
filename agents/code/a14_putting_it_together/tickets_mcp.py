"""A minimal MCP server over stdio (Python standard library only): one tool, `get_ticket`, backed by a small dict.

MCP messages are JSON-RPC 2.0, one JSON object per line on stdin/stdout. A harness starts this program, asks it
for its tools (tools/list) and forwards the model's calls to it (tools/call).
    python3 tickets_mcp.py      # then type: {"jsonrpc":"2.0","id":1,"method":"tools/list"}
"""
import json
import sys

TICKETS = {
    "T-42": "Customer reports invoice totals are too high by exactly the discount amount. Seen on invoice 2026-0042 "
            "(loyalty discount 10%). Expected: total = subtotal - discount + VAT.",
    "T-43": "Feature request: show the VAT ID on the PDF invoice.",
}
TOOLS = [{"name": "get_ticket", "description": "Get the text of a support ticket by its ID, e.g. T-42.",
          "inputSchema": {"type": "object", "properties": {"id": {"type": "string"}}, "required": ["id"]}}]


def handle(msg):
    method, params = msg.get("method"), msg.get("params") or {}
    if method == "initialize":
        return {"protocolVersion": params.get("protocolVersion", "2025-06-18"), "capabilities": {"tools": {}},
                "serverInfo": {"name": "tickets", "version": "1.0"}}
    if method == "tools/list":
        return {"tools": TOOLS}
    if method == "tools/call":
        tid = (params.get("arguments") or {}).get("id", "")
        text = TICKETS.get(tid.strip().upper(), f"no ticket with ID {tid!r}")
        return {"content": [{"type": "text", "text": text}], "isError": tid.strip().upper() not in TICKETS}
    if method == "ping":
        return {}
    raise KeyError(method)


for line in sys.stdin:
    if not line.strip():
        continue
    msg = json.loads(line)
    if "id" not in msg:              # a notification (e.g. notifications/initialized): no reply
        continue
    try:
        reply = {"jsonrpc": "2.0", "id": msg["id"], "result": handle(msg)}
    except KeyError:
        reply = {"jsonrpc": "2.0", "id": msg["id"], "error": {"code": -32601, "message": "method not found"}}
    print(json.dumps(reply), flush=True)
