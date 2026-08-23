"""
benchmark/_run_http_post.py
============================
Run under the modernized venv (.venv, mcp>=2.0.0). Connects to a running
Streamable HTTP pyATS MCP server and times each scenario in scenarios.py.

Usage:
    .venv/bin/python benchmark/_run_http_post.py <url> <client_mode> <out.json>

    <client_mode> is "auto" (negotiates the newest mutually-supported
    protocol — SEP-2575 / 2026-07-28, handshake-free) or "legacy" (forces
    the pre-SEP-2575 initialize-handshake protocol against the same
    server, to isolate the protocol-version effect from the
    stateless_http transport flag).
"""
import asyncio
import json
import sys
import time

from mcp.client.client import Client

sys.path.insert(0, __file__.rsplit("/", 1)[0])
from scenarios import SCENARIOS, WARMUP_ITERATIONS, MEASURED_ITERATIONS  # noqa: E402


async def time_scenario(client, tool_name, kwargs, iterations):
    samples = []
    for _ in range(iterations):
        t0 = time.perf_counter()
        try:
            result = await client.call_tool(tool_name, kwargs)
            text = result.content[0].text if result.content else "{}"
            payload = json.loads(text)
            ok = payload.get("status") in ("completed", "success")
        except Exception as exc:
            ok = False
            payload = {"error": str(exc)}
        elapsed = time.perf_counter() - t0
        samples.append({"elapsed_s": elapsed, "ok": ok})
    return samples


async def main(url: str, mode: str, out_path: str):
    connect_t0 = time.perf_counter()
    async with Client(url, mode=mode) as client:
        connect_elapsed = time.perf_counter() - connect_t0

        available = {t.name for t in (await client.list_tools()).tools}

        results = {}
        for name, tool_name, kwargs in SCENARIOS:
            if tool_name not in available:
                results[name] = {"status": "not_available", "tool": tool_name}
                continue
            # warmup (discarded)
            await time_scenario(client, tool_name, kwargs, WARMUP_ITERATIONS)
            samples = await time_scenario(client, tool_name, kwargs, MEASURED_ITERATIONS)
            results[name] = {"status": "measured", "tool": tool_name, "samples": samples}

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump({
            "connect_elapsed_s": connect_elapsed,
            "url": url,
            "client_mode": mode,
            "results": results,
        }, f, indent=2)


if __name__ == "__main__":
    asyncio.run(main(sys.argv[1], sys.argv[2], sys.argv[3]))
