"""
benchmark/_run_stdio_pre.py
=============================
Run under the pre-modernization venv (.venv-pre, mcp==1.26.0). Spawns the
ORIGINAL (main-branch) pyats_mcp_server.py over STDIO and times each
scenario in scenarios.py — this is the "pre" baseline: legacy protocol,
full initialize/session handshake, one connection for the whole run
(STDIO has no separate connections to open per call).

Usage:
    .venv-pre/bin/python benchmark/_run_stdio_pre.py <server_py_path> <testbed_yaml_path> <out.json>
"""
import asyncio
import json
import os
import sys
import time

from mcp import ClientSession
from mcp.client.stdio import StdioServerParameters, stdio_client

sys.path.insert(0, __file__.rsplit("/", 1)[0])
from scenarios import SCENARIOS, WARMUP_ITERATIONS, MEASURED_ITERATIONS  # noqa: E402


async def time_scenario(session, tool_name, kwargs, iterations):
    samples = []
    for _ in range(iterations):
        t0 = time.perf_counter()
        try:
            result = await session.call_tool(tool_name, kwargs)
            text = result.content[0].text if result.content else "{}"
            payload = json.loads(text)
            ok = payload.get("status") in ("completed", "success")
        except Exception as exc:
            ok = False
            payload = {"error": str(exc)}
        elapsed = time.perf_counter() - t0
        samples.append({"elapsed_s": elapsed, "ok": ok})
    return samples


async def main(server_py: str, testbed_path: str, out_path: str):
    params = StdioServerParameters(
        command=sys.executable,
        args=["-u", server_py],
        env={**os.environ, "PYATS_TESTBED_PATH": testbed_path},
    )

    connect_t0 = time.perf_counter()
    async with stdio_client(params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            connect_elapsed = time.perf_counter() - connect_t0

            available = {t.name for t in (await session.list_tools()).tools}

            results = {}
            for name, tool_name, kwargs in SCENARIOS:
                if tool_name not in available:
                    results[name] = {"status": "not_available", "tool": tool_name}
                    continue
                await time_scenario(session, tool_name, kwargs, WARMUP_ITERATIONS)
                samples = await time_scenario(session, tool_name, kwargs, MEASURED_ITERATIONS)
                results[name] = {"status": "measured", "tool": tool_name, "samples": samples}

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump({
            "connect_elapsed_s": connect_elapsed,
            "transport": "stdio",
            "results": results,
        }, f, indent=2)


if __name__ == "__main__":
    asyncio.run(main(sys.argv[1], sys.argv[2], sys.argv[3]))
