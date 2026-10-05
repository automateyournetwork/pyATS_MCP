#!/usr/bin/env python3
"""Call a pyATS tool as an MCP task, then poll without holding a request open.

python examples/task_client.py http://localhost:8080/mcp pyats_run_show_command \
    '{"device_name":"router-1","command":"show version"}'
"""
import argparse
import asyncio
import json

import httpx2

EXTENSION = "io.modelcontextprotocol/tasks"


async def request(client, url, method, params):
    params = {
        **params,
        "_meta": {
            "io.modelcontextprotocol/protocolVersion": "2026-07-28",
            "io.modelcontextprotocol/clientCapabilities": {"extensions": {EXTENSION: {}}},
        },
    }
    response = await client.post(
        url,
        json={
            "jsonrpc": "2.0",
            "id": 1,
            "method": method,
            "params": params,
        },
        headers={
            "Accept": "application/json, text/event-stream",
            "MCP-Protocol-Version": "2026-07-28",
            "Mcp-Method": method,
            "Mcp-Name": params.get("taskId", params.get("name", "")),
        },
    )
    body = response.json()
    if "error" in body:
        raise RuntimeError(body["error"])
    response.raise_for_status()
    return body["result"]


async def main(url, tool, arguments):
    async with httpx2.AsyncClient(timeout=30) as client:
        result = await request(client, url, "tools/call", {"name": tool, "arguments": arguments})
        if result.get("resultType") != "task":
            print(json.dumps(result, indent=2))
            return
        task_id = result["taskId"]
        print(f"Task accepted: {task_id}")
        # Other work can run here. Each poll is a separate, short HTTP request.
        while result["status"] in ("working", "input_required"):
            await asyncio.sleep(result.get("pollIntervalMs", 1000) / 1000)
            result = await request(client, url, "tasks/get", {"taskId": task_id})
        print(json.dumps(result, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("url")
    parser.add_argument("tool")
    parser.add_argument("arguments", type=json.loads)
    args = parser.parse_args()
    asyncio.run(main(args.url, args.tool, args.arguments))
