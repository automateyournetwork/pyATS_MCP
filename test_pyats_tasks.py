"""Real SDK and Streamable HTTP tests; device work is simulated, never sent to SSH."""

import asyncio
import json
import threading
from contextlib import asynccontextmanager
from types import SimpleNamespace

import httpx2
import pytest
from mcp import MCPError
from mcp.server.mcpserver import MCPServer
from mcp_types import CallToolRequestParams

from pyats_tasks import EXTENSION_ID, PROTOCOL, PyatsTasks, TaskParams, TaskStore


@asynccontextmanager
async def serve(tmp_path, *, workers=2, max_tasks=1000, stateless=True):
    runtime = PyatsTasks(
        tmp_path / "tasks.db",
        lambda: ["r1", "r2"],
        workers=workers,
        max_tasks=max_tasks,
        retention=60,
    )
    server = MCPServer("test-pyats", extensions=[runtime], lifespan=runtime.lifespan)
    entered = []
    release = threading.Event()

    @server.tool()
    async def pyats_run_show_command(device_name: str, command: str) -> str:
        entered.append(device_name)
        if command == "block":
            await asyncio.to_thread(release.wait, 5)
        if command == "protocol-error":
            raise MCPError(code=-32602, message="invalid command")
        if command == "tool-error":
            raise ValueError("device unreachable")
        return json.dumps({"device": device_name, "output": command})

    app = server.streamable_http_app(json_response=True, stateless_http=stateless)
    async with app.router.lifespan_context(app):
        async with httpx2.AsyncClient(
            transport=httpx2.ASGITransport(app=app), base_url="http://localhost:8080"
        ) as client:
            try:
                yield runtime, client, entered, release
            finally:
                release.set()


async def rpc(client, method, params=None, supported=True):
    params = dict(params or {})
    params["_meta"] = {
        "io.modelcontextprotocol/protocolVersion": PROTOCOL,
        "io.modelcontextprotocol/clientCapabilities": {
            "extensions": {EXTENSION_ID: {}} if supported else {},
        },
    }
    response = await client.post(
        "/mcp",
        json={"jsonrpc": "2.0", "id": 1, "method": method, "params": params},
        headers={
            "Accept": "application/json, text/event-stream",
            "MCP-Protocol-Version": PROTOCOL,
            "Mcp-Method": method,
            "Mcp-Name": params.get("taskId", params.get("name", "")),
        },
    )
    assert response.status_code in (200, 400), response.text
    return response.json()


async def call(client, device="r1", command="show version", supported=True):
    return await rpc(
        client,
        "tools/call",
        {
            "name": "pyats_run_show_command",
            "arguments": {"device_name": device, "command": command},
        },
        supported,
    )


async def terminal(client, task_id):
    async def poll():
        while True:
            reply = await rpc(client, "tasks/get", {"taskId": task_id})
            result = reply["result"]
            if result["status"] != "working":
                return result
            await asyncio.sleep(0.01)

    return await asyncio.wait_for(poll(), 5)


@pytest.mark.parametrize("stateless", [False, True])
async def test_http_task_returns_before_blocking_work_and_polls(tmp_path, stateless):
    async with serve(tmp_path, stateless=stateless) as (runtime, client, entered, release):
        discovery = await rpc(client, "server/discover")
        assert EXTENSION_ID in discovery["result"]["capabilities"]["extensions"]
        response = await asyncio.wait_for(call(client, command="block"), 1)
        task = response["result"]
        assert task["resultType"] == "task"
        assert task["status"] == "working"
        assert not release.is_set()
        polled = await rpc(client, "tasks/get", {"taskId": task["taskId"]})
        assert polled["result"]["status"] == "working"
        # Independent devices can complete while r1 is blocked.
        other = (await call(client, device="r2"))["result"]
        assert (await terminal(client, other["taskId"]))["status"] == "completed"
        release.set()
        done = await terminal(client, task["taskId"])
        assert json.loads(done["result"]["content"][0]["text"])["output"] == "block"
        assert done["ttlMs"] > 0


async def test_regular_calls_and_capability_gates(tmp_path):
    async with serve(tmp_path) as (_, client, _, _):
        response = await call(client, supported=False)
        assert response["result"]["resultType"] == "complete"
        assert "taskId" not in response["result"]
        for method in ("tasks/get", "tasks/cancel", "tasks/update"):
            params = {"taskId": "unknown", "inputResponses": {}}
            assert (await rpc(client, method, params, False))["error"]["code"] == -32021
            assert (await rpc(client, method, params))["error"]["code"] == -32602


async def test_queue_cancel_and_running_cancel_are_honest(tmp_path):
    async with serve(tmp_path, workers=1) as (_, client, entered, release):
        first = (await call(client, command="block"))["result"]["taskId"]
        while not entered:
            await asyncio.sleep(0.01)
        second = (await call(client, device="r2"))["result"]["taskId"]
        assert (await rpc(client, "tasks/cancel", {"taskId": second}))["result"][
            "resultType"
        ] == "complete"
        assert (await terminal(client, second))["status"] == "cancelled"
        await rpc(client, "tasks/cancel", {"taskId": first})
        assert (await rpc(client, "tasks/get", {"taskId": first}))["result"]["status"] == "working"
        release.set()
        assert (await terminal(client, first))["status"] == "completed"
        assert entered == ["r1"]


async def test_device_serialization_includes_ordinary_calls(tmp_path):
    async with serve(tmp_path) as (_, client, entered, release):
        first = (await call(client, command="block"))["result"]["taskId"]
        while not entered:
            await asyncio.sleep(0.01)
        regular = asyncio.create_task(call(client, supported=False))
        await asyncio.sleep(0.05)
        assert not regular.done()
        assert entered == ["r1"]
        release.set()
        await regular
        await terminal(client, first)
        assert entered == ["r1", "r1"]


async def test_errors_capacity_and_update(tmp_path):
    async with serve(tmp_path, max_tasks=2) as (_, client, _, _):
        protocol = (await call(client, command="protocol-error"))["result"]["taskId"]
        failed = await terminal(client, protocol)
        assert failed["status"] == "failed"
        assert failed["error"]["code"] == -32602
        tool = (await call(client, command="tool-error"))["result"]["taskId"]
        done = await terminal(client, tool)
        assert done["status"] == "completed"
        assert done["result"]["isError"] is True
        ack = await rpc(client, "tasks/update", {"taskId": tool, "inputResponses": {"unused": {}}})
        assert ack["result"]["resultType"] == "complete"
        assert (await call(client))["error"]["code"] == -32000


async def test_results_survive_server_restart(tmp_path):
    async with serve(tmp_path) as (_, client, _, _):
        task_id = (await call(client))["result"]["taskId"]
        result = await terminal(client, task_id)
    async with serve(tmp_path) as (_, client, _, _):
        recovered = (await rpc(client, "tasks/get", {"taskId": task_id}))["result"]
        assert recovered["result"] == result["result"]


def test_store_recovery_expiry_ownership_and_single_process(tmp_path):
    path = tmp_path / "tasks.db"
    store = TaskStore(path)
    store.start()
    try:
        with pytest.raises(BlockingIOError):
            TaskStore(path).start()
        task = dict(taskId="secret", status="working", createdAt="2026-01-01T00:00:00Z", ttlMs=None)
        store.put(task, "owner")
        with pytest.raises(MCPError):
            store.get("secret", "other-owner")
    finally:
        store.close()
    store.start()
    try:
        recovered = store.get("secret", "owner")
        assert recovered["status"] == "failed"
        assert recovered["error"]["code"] == -32603
        store.put(recovered, "owner", expires=1)
        with pytest.raises(MCPError):
            store.get("secret", "owner")
        assert store.count() == 0
    finally:
        store.close()


async def test_request_cancellation_does_not_cancel_task(tmp_path):
    runtime = PyatsTasks(tmp_path / "tasks.db", lambda: ["r1"])
    ctx = SimpleNamespace(
        protocol_version=PROTOCOL,
        session=SimpleNamespace(client_capabilities=SimpleNamespace(extensions={EXTENSION_ID: {}})),
    )
    entered = asyncio.Event()
    release = asyncio.Event()

    async def work(ctx):
        entered.set()
        await release.wait()
        return {"content": [{"type": "text", "text": "done"}]}

    async with runtime.lifespan(None):
        result = await runtime.intercept_tool_call(
            CallToolRequestParams(name="pyats_run_show_command", arguments={"device_name": "r1"}),
            ctx,
            work,
        )
        await entered.wait()
        # The original request scope has ended; the job remains independently owned.
        assert result["taskId"] in runtime.jobs
        release.set()
        await asyncio.gather(*runtime.jobs.values())
        assert (await runtime.get(ctx, TaskParams(taskId=result["taskId"])))[
            "status"
        ] == "completed"


async def test_cancelled_ordinary_request_keeps_device_locked_until_io_finishes(tmp_path):
    runtime = PyatsTasks(tmp_path / "tasks.db", lambda: ["r1"])
    ctx = SimpleNamespace(
        protocol_version=PROTOCOL,
        session=SimpleNamespace(client_capabilities=SimpleNamespace(extensions={})),
    )
    entered = asyncio.Event()
    release = asyncio.Event()

    async def work(ctx):
        entered.set()
        await release.wait()
        return {"content": []}

    async with runtime.lifespan(None):
        request = asyncio.create_task(
            runtime.intercept_tool_call(
                CallToolRequestParams(
                    name="pyats_run_show_command", arguments={"device_name": "r1"}
                ),
                ctx,
                work,
            )
        )
        await entered.wait()
        request.cancel()
        with pytest.raises(asyncio.CancelledError):
            await request
        assert runtime.device_locks["r1"].locked()
        release.set()
        await asyncio.gather(*runtime.background)
        assert not runtime.device_locks["r1"].locked()


async def test_actual_pyats_server_registration(tmp_path, monkeypatch):
    # Existing test helpers stub only pyATS/Genie; MCP and HTTP remain real.
    import pyats_mcp_server as srv
    import test_pyats_mcp_server as existing

    monkeypatch.setattr(srv.task_runtime, "store", TaskStore(tmp_path / "tasks.db"))
    monkeypatch.setattr(
        srv,
        "_execute_show_command",
        lambda *args: {
            "status": "completed",
            "output": "mock SSH output",
            "parsed": False,
        },
    )
    monkeypatch.setattr(srv, "_load_testbed", lambda: existing._mock_testbed("r1"))
    app = srv.mcp.streamable_http_app(json_response=True, stateless_http=True)
    async with app.router.lifespan_context(app):
        async with httpx2.AsyncClient(
            transport=httpx2.ASGITransport(app=app), base_url="http://localhost:8080"
        ) as client:
            tools = (await rpc(client, "tools/list"))["result"]["tools"]
            names = {tool["name"] for tool in tools}
            from pyats_tasks import TASK_TOOLS

            assert TASK_TOOLS <= names
            task_id = (await call(client))["result"]["taskId"]
            done = await terminal(client, task_id)
            assert "mock SSH output" in done["result"]["content"][0]["text"]


async def test_legacy_handshake_client_still_gets_regular_results(tmp_path):
    async with serve(tmp_path, stateless=False) as (_, client, _, _):
        headers = {"Accept": "application/json, text/event-stream"}
        initialized = await client.post(
            "/mcp",
            headers=headers,
            json={
                "jsonrpc": "2.0",
                "id": 1,
                "method": "initialize",
                "params": {
                    "protocolVersion": "2025-06-18",
                    "capabilities": {},
                    "clientInfo": {"name": "legacy-test", "version": "1.0"},
                },
            },
        )
        assert initialized.status_code == 200
        headers["Mcp-Session-Id"] = initialized.headers["mcp-session-id"]
        headers["MCP-Protocol-Version"] = "2025-06-18"
        await client.post(
            "/mcp",
            headers=headers,
            json={
                "jsonrpc": "2.0",
                "method": "notifications/initialized",
            },
        )
        response = await client.post(
            "/mcp",
            headers=headers,
            json={
                "jsonrpc": "2.0",
                "id": 2,
                "method": "tools/call",
                "params": {
                    "name": "pyats_run_show_command",
                    "arguments": {"device_name": "r1", "command": "show version"},
                },
            },
        )
        assert response.status_code == 200, response.text
        result = response.json()["result"]
        assert "taskId" not in result
        assert "show version" in result["content"][0]["text"]


async def test_cancel_before_worker_first_turn_persists_terminal_status(tmp_path):
    runtime = PyatsTasks(tmp_path / "tasks.db", lambda: ["r1"])
    ctx = SimpleNamespace(
        protocol_version=PROTOCOL,
        session=SimpleNamespace(client_capabilities=SimpleNamespace(extensions={EXTENSION_ID: {}})),
    )
    calls = []

    async def work(ctx):
        calls.append(True)
        return {}

    async with runtime.lifespan(None):
        result = await runtime._submit(
            CallToolRequestParams(name="pyats_run_show_command", arguments={"device_name": "r1"}),
            ctx,
            work,
            "anonymous",
        )
        runtime._cancel(result["taskId"])
        await asyncio.gather(*runtime.jobs.values())
        assert not calls
        assert (await runtime.get(ctx, TaskParams(taskId=result["taskId"])))[
            "status"
        ] == "cancelled"
