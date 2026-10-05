"""Polling-only MCP Tasks extension (2026-07-28) for pyATS operations.

One server process owns a SQLite task store. Work survives HTTP disconnects;
results survive process restarts, but interrupted commands are never replayed.
"""

from __future__ import annotations

import asyncio
import fcntl
import hashlib
import json
import logging
import os
import secrets
import sqlite3
import time
from contextlib import AsyncExitStack, asynccontextmanager, contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

from mcp import MCPError
from mcp.server.auth.middleware.auth_context import get_access_token
from mcp.server.auth.provider import principal_components
from mcp.server.extension import Extension, MethodBinding
from mcp.server.mcpserver import require_client_extension
from mcp_types import RequestParams
from pydantic import Field

EXTENSION_ID = "io.modelcontextprotocol/tasks"
PROTOCOL = "2026-07-28"
logger = logging.getLogger(__name__)

# Inventory, local snapshot diffs and operation history remain immediate.
TASK_TOOLS = frozenset(
    {
        "pyats_run_show_command",
        "pyats_run_show_command_multi",
        "pyats_pcall_show_command",
        "pyats_show_running_config",
        "pyats_show_logging",
        "pyats_device_health",
        "pyats_learn_feature",
        "pyats_ping_from_network_device",
        "pyats_get_neighbors",
        "pyats_find_interface_by_ip",
        "pyats_run_linux_command",
        "pyats_configure_device",
        "pyats_configure_with_diff",
        "pyats_rollback_config",
        "pyats_configure_devices_multi",
        "pyats_pcall_configure_devices",
        "pyats_rest_request",
        "pyats_clean_device",
        "pyats_run_dynamic_test",
        "pyats_run_blitz",
        "pyats_run_robot",
        "pyats_xpresso_request",
    }
)


class TaskParams(RequestParams):
    task_id: str = Field(alias="taskId", min_length=1, max_length=256)


class UpdateParams(TaskParams):
    input_responses: dict[str, Any] = Field(alias="inputResponses")


def _now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _owner() -> str:
    token = get_access_token()
    if token is None:
        # This server is unauthenticated by default. The random task ID is a
        # bearer credential; never publish it or expose a task enumeration API.
        return "anonymous"
    return hashlib.sha256(json.dumps(principal_components(token)).encode()).hexdigest()


class TaskStore:
    def __init__(self, path: Path):
        self.path = path
        self.lock_file = None

    def start(self, retention=86400):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.lock_file = open(str(self.path) + ".lock", "a")
        try:
            fcntl.flock(self.lock_file, fcntl.LOCK_EX | fcntl.LOCK_NB)
            fd = os.open(self.path, os.O_CREAT | os.O_WRONLY, 0o600)
            os.close(fd)
            os.chmod(self.path, 0o600)
            with self.connect() as db:
                db.execute(
                    "CREATE TABLE IF NOT EXISTS tasks "
                    "(id TEXT PRIMARY KEY, owner TEXT, expires REAL, payload TEXT)"
                )
                rows = db.execute("SELECT id, payload FROM tasks").fetchall()
                for task_id, payload in rows:
                    task = json.loads(payload)
                    if task["status"] == "working":
                        task.update(
                            status="failed",
                            lastUpdatedAt=_now(),
                            error={
                                "code": -32603,
                                "message": "Server restarted during execution; "
                                "device outcome may be "
                                "unknown. Inspect the device before retrying.",
                            },
                        )
                        task["ttlMs"] = max(
                            1,
                            int(
                                (
                                    time.time()
                                    + retention
                                    - datetime.fromisoformat(
                                        task["createdAt"].replace("Z", "+00:00")
                                    ).timestamp()
                                )
                                * 1000
                            ),
                        )
                        db.execute(
                            "UPDATE tasks SET payload=?, expires=? WHERE id=?",
                            (json.dumps(task), time.time() + retention, task_id),
                        )
            os.chmod(self.path, 0o600)
        except BaseException:
            self.close()
            raise

    def connect(self):
        # sqlite's own context manager does not close a connection.
        @contextmanager
        def connection():
            db = sqlite3.connect(self.path)
            try:
                with db:
                    yield db
            finally:
                db.close()

        return connection()

    def put(self, task: dict, owner: str, expires=None):
        with self.connect() as db:
            db.execute(
                "INSERT OR REPLACE INTO tasks VALUES (?, ?, ?, ?)",
                (task["taskId"], owner, expires, json.dumps(task)),
            )

    def get(self, task_id: str, owner: str):
        with self.connect() as db:
            row = db.execute(
                "SELECT payload FROM tasks WHERE id=? AND owner=? "
                "AND (expires IS NULL OR expires>?)",
                (task_id, owner, time.time()),
            ).fetchone()
        if row is None:
            raise MCPError(code=-32602, message="Task not found or expired")
        return json.loads(row[0])

    def count(self):
        with self.connect() as db:
            db.execute("DELETE FROM tasks WHERE expires IS NOT NULL AND expires<=?", (time.time(),))
            return db.execute("SELECT COUNT(*) FROM tasks").fetchone()[0]

    def close(self):
        if self.lock_file is not None:
            self.lock_file.close()
            self.lock_file = None


class PyatsTasks(Extension):
    identifier = EXTENSION_ID

    def __init__(
        self, path: Path, device_names: Callable, *, workers=4, max_tasks=1000, retention=86400
    ):
        if min(workers, max_tasks, retention) < 1:
            raise ValueError("Task workers, max_tasks and retention must be positive")
        self.store = TaskStore(path)
        self.device_names = device_names
        self.retention = retention
        self.max_tasks = max_tasks
        self.slots = asyncio.Semaphore(workers)
        self.admission = asyncio.Lock()
        self.device_locks: dict[str, asyncio.Lock] = {}
        self.jobs: dict[str, asyncio.Task] = {}
        self.running: set[str] = set()
        self.waiting: set[str] = set()
        self.cancel_requested: set[str] = set()
        self.background: set[asyncio.Task] = set()
        self.closing = False
        self.started = False

    @asynccontextmanager
    async def lifespan(self, server):
        await asyncio.to_thread(self.store.start, self.retention)
        self.started = True
        self.closing = False
        try:
            yield {}
        finally:
            self.closing = True
            # Finish admission before draining workers. Never release an SSH
            # device lock while its blocking thread may still be using it.
            if self.background:
                await asyncio.gather(*self.background, return_exceptions=True)
            for task_id, job in list(self.jobs.items()):
                if task_id not in self.running:
                    self._cancel(task_id)
            if self.jobs:
                await asyncio.gather(*self.jobs.values(), return_exceptions=True)
            await asyncio.to_thread(self.store.close)
            self.started = False

    def methods(self):
        return [
            MethodBinding(name, model, handler, protocol_versions=frozenset({PROTOCOL}))
            for name, model, handler in [
                ("tasks/get", TaskParams, self.get),
                ("tasks/cancel", TaskParams, self.cancel),
                ("tasks/update", UpdateParams, self.update),
            ]
        ]

    def _spawn(self, coro):
        task = asyncio.create_task(coro)
        self.background.add(task)

        def finished(job):
            self.background.discard(job)
            if not job.cancelled() and job.exception() is not None:
                logger.error("Background request failed: %s", job.exception())

        task.add_done_callback(finished)
        return task

    async def intercept_tool_call(self, params, ctx, call_next):
        if params.name not in TASK_TOOLS:
            return await call_next(ctx)
        capabilities = ctx.session.client_capabilities
        supported = (
            ctx.protocol_version == PROTOCOL
            and capabilities is not None
            and EXTENSION_ID in (capabilities.extensions or {})
        )
        if not self.started or self.closing:
            raise MCPError(code=-32603, message="Task runtime is not accepting work")
        if not supported:
            # Legacy calls share device locks with task calls and retain their
            # original result shape. A disconnected client cannot release a
            # lock before its SSH thread actually finishes.
            return await asyncio.shield(self._spawn(self._invoke(params, ctx, call_next)))
        return await asyncio.shield(self._spawn(self._submit(params, ctx, call_next, _owner())))

    async def _submit(self, params, ctx, call_next, owner):
        async with self.admission:
            if await asyncio.to_thread(self.store.count) >= self.max_tasks:
                raise MCPError(
                    code=-32000, message="Task capacity reached; retry after tasks expire"
                )
            stamp = _now()
            task = dict(
                taskId=secrets.token_urlsafe(32),
                status="working",
                statusMessage="Accepted for execution",
                createdAt=stamp,
                lastUpdatedAt=stamp,
                ttlMs=None,
                pollIntervalMs=1000,
            )
            await asyncio.to_thread(self.store.put, task, owner)
            task_id = task["taskId"]
            job = asyncio.create_task(self._work(task, owner, params, ctx, call_next))
            self.jobs[task_id] = job

            def finished(job):
                self.jobs.pop(task_id, None)
                self.cancel_requested.discard(task_id)
                if not job.cancelled() and job.exception() is not None:
                    logger.error("Could not finalize task %s: %s", task_id, job.exception())

            job.add_done_callback(finished)
            return {"resultType": "task", **task}

    async def _invoke(self, params, ctx, call_next, task_id=None):
        async with self.slots:
            args = params.arguments or {}
            names = args.get("device_names") or (
                [args["device_name"]] if args.get("device_name") else []
            )
            if params.name in {"pyats_run_dynamic_test", "pyats_run_robot"} or (
                params.name == "pyats_find_interface_by_ip" and not names
            ):
                names = await asyncio.to_thread(self.device_names)
            # Invalid argument shapes are left to the SDK's tool validation.
            names = names if isinstance(names, list) else []
            async with AsyncExitStack() as stack:
                for name in sorted({n for n in names if isinstance(n, str)}):
                    lock = self.device_locks.setdefault(name, asyncio.Lock())
                    await stack.enter_async_context(lock)
                if task_id:
                    self.waiting.discard(task_id)
                    self.running.add(task_id)
                return await call_next(ctx)

    async def _work(self, task, owner, params, ctx, call_next):
        task_id = task["taskId"]
        self.waiting.add(task_id)
        try:
            if task_id in self.cancel_requested:
                raise asyncio.CancelledError
            result = await self._invoke(params, ctx, call_next, task_id)
            if hasattr(result, "model_dump"):
                result = result.model_dump(mode="json", by_alias=True, exclude_none=True)
            task.update(status="completed", statusMessage="Execution finished", result=result)
        except asyncio.CancelledError:
            if task_id in self.running:
                task.update(
                    status="failed",
                    error={
                        "code": -32603,
                        "message": "Execution interrupted; device outcome may be unknown",
                    },
                )
            else:
                task.update(status="cancelled", statusMessage="Cancelled before execution")
        except MCPError as exc:
            task.update(
                status="failed",
                error=exc.error.model_dump(mode="json", by_alias=True, exclude_none=True),
            )
        except Exception as exc:
            task.update(status="failed", error={"code": -32603, "message": str(exc)})
        finally:
            self.running.discard(task_id)
            self.waiting.discard(task_id)
            self.cancel_requested.discard(task_id)
        task["lastUpdatedAt"] = _now()
        expires = time.time() + self.retention
        created = datetime.fromisoformat(task["createdAt"].replace("Z", "+00:00")).timestamp()
        task["ttlMs"] = int((expires - created) * 1000)
        await asyncio.to_thread(self.store.put, task, owner, expires)

    async def get(self, ctx, params):
        require_client_extension(ctx, EXTENSION_ID)
        task = await asyncio.to_thread(self.store.get, params.task_id, _owner())
        return {"resultType": "complete", **task}

    async def cancel(self, ctx, params):
        require_client_extension(ctx, EXTENSION_ID)
        await asyncio.to_thread(self.store.get, params.task_id, _owner())
        job = self.jobs.get(params.task_id)
        # Cooperative cancellation: once a tool starts, preserve its result and
        # let it finish. SSH, configuration and subprocess side effects cannot
        # be safely interrupted by cancelling the awaiting coroutine.
        if job is not None and params.task_id not in self.running:
            self._cancel(params.task_id)
        return {"resultType": "complete"}

    def _cancel(self, task_id):
        # Also handles cancellation before the coroutine gets its first turn,
        # and avoids cancelling a second time while the final state is saved.
        self.cancel_requested.add(task_id)
        if task_id in self.waiting:
            self.waiting.discard(task_id)
            self.jobs[task_id].cancel()

    async def update(self, ctx, params):
        require_client_extension(ctx, EXTENSION_ID)
        await asyncio.to_thread(self.store.get, params.task_id, _owner())
        # These tools never request client input. The spec says to ignore
        # unknown/already-answered input response keys.
        return {"resultType": "complete"}
