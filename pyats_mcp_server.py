#!/usr/bin/env python3
"""
pyats_mcp_server.py
===================
MCP server that exposes Cisco pyATS / Genie functionality as structured tools
for use by AI agents (Claude, LangGraph, etc.) over Streamable HTTP JSON-RPC 2.0
transport, in either stateful or stateless mode (see PYATS_MCP_TRANSPORT_MODE).

Design principles
-----------------
- All blocking I/O (device connections, CLI execution) runs in a thread-pool
  executor so the asyncio event loop is never blocked.
- Every tool returns a JSON string so the calling agent always gets a
  predictable, parseable response — never a raw exception traceback.
- Every error response includes "tool", "device", "command", and a
  "suggestion" field to help an agent self-recover without human guidance.
- An in-memory operation log (_OP_LOG) lets an agent review what has already
  been tried before deciding its next step.
- Config changes are guarded by guardrails (dangerous-command detection) and
  optionally saved as rollback snapshots before being applied.
"""

from __future__ import annotations

import asyncio
import difflib
import json
import logging
import os
import re
import shutil
import string
import subprocess
import sys
import textwrap
import threading
import time
import zipfile
from difflib import SequenceMatcher
from functools import partial
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import requests
import yaml
from dotenv import load_dotenv
from genie.libs.parser.utils import get_parser
from genie.utils.diff import Diff
from mcp.server.mcpserver import MCPServer
from pyats.async_ import pcall
from pyats.topology import loader

# ---------------------------------------------------------------------------
# Logging — written to stderr so STDIO transport is not polluted
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(name)s %(levelname)s %(message)s",
    stream=sys.stderr,
)
logger = logging.getLogger("PyatsMCPServer")

# ---------------------------------------------------------------------------
# Environment configuration
# ---------------------------------------------------------------------------
load_dotenv()

TESTBED_PATH: str = os.getenv("PYATS_TESTBED_PATH", "")
if not TESTBED_PATH or not os.path.exists(TESTBED_PATH):
    logger.critical("PYATS_TESTBED_PATH not set or file not found: %s", TESTBED_PATH)
    sys.exit(1)

logger.info("Using testbed: %s", TESTBED_PATH)

# Artifact storage for dynamic test runs
ARTIFACTS_DIR: Path = Path(
    os.getenv("PYATS_MCP_ARTIFACTS_DIR", str(Path.home() / ".pyats-mcp" / "artifacts"))
).resolve()
ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

KEEP_ARTIFACTS: bool = os.getenv("PYATS_MCP_KEEP_ARTIFACTS", "1") == "1"

# XPresso REST API — optional; pyats_xpresso_request errors clearly if unset.
XPRESSO_URL: str = os.getenv("XPRESSO_URL", "").rstrip("/")
XPRESSO_API_TOKEN: str = os.getenv("XPRESSO_API_TOKEN", "")
XPRESSO_GROUP: str = os.getenv("XPRESSO_GROUP", "")

# Testbed re-load interval (seconds); avoids hammering disk on every call
def _parse_int_env(var: str, default: int) -> int:
    raw = os.getenv(var, str(default))
    try:
        return int(raw)
    except ValueError:
        logger.warning("Invalid value for %s=%r; using default %d", var, raw, default)
        return default

_TESTBED_CACHE_TTL: int = _parse_int_env("PYATS_MCP_TESTBED_CACHE_TTL", 30)
_testbed_cache: Dict[str, Any] = {"loaded_at": 0.0, "tb": None}

# Connection caching — keep connections alive between calls (0 = disabled)
_CONN_CACHE_TTL: int = _parse_int_env("PYATS_MCP_CONN_CACHE_TTL", 0)
_conn_cache: Dict[str, Dict[str, Any]] = {}

# Connection defaults — these are used only when the testbed does NOT define a value
# Set to empty string to disable the MCP default and use only testbed-defined values
_DEFAULT_CONNECTION_TIMEOUT: Optional[int] = (
    None if os.getenv("PYATS_MCP_CONNECTION_TIMEOUT", "") == ""
    else _parse_int_env("PYATS_MCP_CONNECTION_TIMEOUT", 120)
)
_DEFAULT_LEARN_HOSTNAME: bool = os.getenv("PYATS_MCP_LEARN_HOSTNAME", "1") == "1"
_DEFAULT_LOG_STDOUT: bool = os.getenv("PYATS_MCP_LOG_STDOUT", "0") == "1"
_DEFAULT_MIT: Optional[bool] = (
    None if os.getenv("PYATS_MCP_MIT", "") == ""
    else os.getenv("PYATS_MCP_MIT", "1") == "1"
)

# In-memory operation log — survives for the lifetime of the server process
_OP_LOG: List[Dict[str, Any]] = []
_OP_LOG_MAX: int = _parse_int_env("PYATS_MCP_OP_LOG_MAX", 500)

# Pre-configure snapshots for rollback  { device_name -> running-config string }
_config_snapshots: Dict[str, str] = {}

# Genie "learn" snapshots for diffing  { "device:feature:label" -> learned dict }
_learn_snapshots: Dict[str, Dict[str, Any]] = {}

# ---------------------------------------------------------------------------
# Concurrency
#
# Under the STDIO transport, tool calls from one client were the only source
# of concurrency (multi-device fan-out via asyncio.gather). Under Streamable
# HTTP — stateful or stateless — multiple clients can hit this process at
# once, so every mutation of the process-global caches above (_testbed_cache,
# _conn_cache, _config_snapshots, _learn_snapshots, _OP_LOG) must be
# serialized. Blocking device I/O (connect/disconnect/execute) must NEVER
# happen while holding this lock — only the dict reads/writes around it —
# otherwise concurrent device fan-out would be serialized server-wide,
# defeating the point of the thread pool.
# ---------------------------------------------------------------------------
_STATE_LOCK = threading.Lock()

# ---------------------------------------------------------------------------
# ANSI / non-printable stripping
# ---------------------------------------------------------------------------
_ANSI_RE = re.compile(r"\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])")


def clean_output(text: str) -> str:
    """Strip ANSI escape codes and non-printable characters from CLI output."""
    text = _ANSI_RE.sub("", text)
    return "".join(ch for ch in text if ch in string.printable)


# ---------------------------------------------------------------------------
# Operation log
# ---------------------------------------------------------------------------

def _log_op(
    tool: str,
    device: Optional[str],
    detail: str,
    status: str,
    error: Optional[str] = None,
) -> None:
    """Append one entry to the in-memory operation log, evicting oldest if full."""
    entry: Dict[str, Any] = {
        "ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "tool": tool,
        "device": device,
        "detail": detail,
        "status": status,
    }
    if error:
        entry["error"] = error
    with _STATE_LOCK:
        _OP_LOG.append(entry)
        if len(_OP_LOG) > _OP_LOG_MAX:
            _OP_LOG.pop(0)


def _err(
    tool: str,
    device: Optional[str],
    command: Optional[str],
    message: str,
    suggestion: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Build a rich error payload and log the failure.

    Every error includes the originating tool, device, and command so the
    calling agent has enough context to decide what to do next.
    """
    payload: Dict[str, Any] = {
        "status": "error",
        "tool": tool,
        "device": device,
        "command": command,
        "error": message,
    }
    if suggestion:
        payload["suggestion"] = suggestion
    _log_op(tool, device, command or "", "error", message)
    return payload


# ---------------------------------------------------------------------------
# Testbed helpers
# ---------------------------------------------------------------------------

def _load_testbed():
    """Return the cached testbed, reloading from disk when the TTL has expired."""
    with _STATE_LOCK:
        now = time.time()
        if _testbed_cache["tb"] is None or (now - _testbed_cache["loaded_at"]) > _TESTBED_CACHE_TTL:
            _testbed_cache["tb"] = loader.load(TESTBED_PATH)
            _testbed_cache["loaded_at"] = now
        return _testbed_cache["tb"]


def _get_testbed_connection_args(device) -> Dict[str, Any]:
    """
    Extract connection arguments from the testbed device definition.

    Checks both the default connection's 'arguments' block and the device-level
    'custom.connection_args' for any user-defined connection settings such as:
      - connection_timeout
      - init_exec_commands
      - init_config_commands
      - mit
      - etc.

    These testbed-defined values take precedence over MCP defaults.
    """
    args: Dict[str, Any] = {}

    # Check device-level custom.connection_args (less common but supported)
    if hasattr(device, "custom") and isinstance(device.custom, dict):
        custom_args = device.custom.get("connection_args", {})
        if isinstance(custom_args, dict):
            args.update(custom_args)

    # Check the default connection's arguments block (most common location)
    connections = getattr(device, "connections", {})
    if connections:
        # Try to find the default connection or first available
        default_conn = connections.get("defaults", {})
        if default_conn and hasattr(default_conn, "get"):
            conn_args = default_conn.get("arguments", {})
            if isinstance(conn_args, dict):
                args.update(conn_args)

        # Also check the primary connection (cli, ssh, etc.)
        for conn_name in ("cli", "ssh", "telnet", "netconf", "rest", "a", "default"):
            conn = connections.get(conn_name)
            if conn and hasattr(conn, "__dict__"):
                conn_dict = getattr(conn, "__dict__", {})
                conn_args = conn_dict.get("arguments", {})
                if isinstance(conn_args, dict):
                    args.update(conn_args)
                break

    return args


def _evict_expired_connections() -> None:
    """Disconnect and evict cache entries whose TTL has elapsed."""
    if _CONN_CACHE_TTL <= 0:
        return
    now = time.time()
    # Snapshot-and-pop under the lock so a concurrent _get_device() call for
    # the same device never observes a half-evicted entry; the actual
    # dev.disconnect() I/O then happens outside the lock.
    with _STATE_LOCK:
        expired = [
            k for k, v in _conn_cache.items()
            if (now - float(v.get("last_used", 0))) > _CONN_CACHE_TTL
        ]
        expired_entries = [(k, _conn_cache.pop(k, None)) for k in expired]
    for name, entry in expired_entries:
        dev = (entry or {}).get("device")
        try:
            if dev and getattr(dev, "is_connected", lambda: False)():
                logger.info("Connection cache TTL expired — disconnecting %s", name)
                dev.disconnect()
        except Exception as e:
            logger.warning("Error disconnecting expired connection for %s: %s", name, e)


def _get_device(device_name: str):
    """
    Return a connected pyATS device object for *device_name*.

    Raises ValueError if the device is not present in the testbed.
    Respects the connection cache when PYATS_MCP_CONN_CACHE_TTL > 0.

    Connection arguments are merged in this priority order (highest wins):
      1. Testbed-defined arguments (device's connection 'arguments' block)
      2. MCP environment variable overrides (PYATS_MCP_CONNECTION_TIMEOUT, etc.)
      3. MCP built-in defaults (learn_hostname=True, log_stdout=False)

    To fully respect testbed settings, set PYATS_MCP_CONNECTION_TIMEOUT=""
    and PYATS_MCP_MIT="" in your environment to disable MCP defaults for
    those parameters.
    """
    tb = _load_testbed()
    device = tb.devices.get(device_name)
    if not device:
        raise ValueError(
            f"Device '{device_name}' not found in testbed. "
            "Use pyats_list_devices or pyats_search_devices to find valid names."
        )

    if _CONN_CACHE_TTL > 0:
        _evict_expired_connections()
        with _STATE_LOCK:
            cached = _conn_cache.get(device_name, {}).get("device")
            if cached and getattr(cached, "is_connected", lambda: False)():
                _conn_cache[device_name]["last_used"] = time.time()
                return cached

    if not device.is_connected():
        logger.info("Connecting to %s …", device_name)

        # Start with MCP defaults (only for values that are set)
        connect_args: Dict[str, Any] = {
            "learn_hostname": _DEFAULT_LEARN_HOSTNAME,
            "log_stdout": _DEFAULT_LOG_STDOUT,
        }

        # Add optional MCP defaults only if they are configured
        if _DEFAULT_CONNECTION_TIMEOUT is not None:
            connect_args["connection_timeout"] = _DEFAULT_CONNECTION_TIMEOUT
        if _DEFAULT_MIT is not None:
            connect_args["mit"] = _DEFAULT_MIT

        # Merge in testbed-defined arguments (these take precedence)
        testbed_args = _get_testbed_connection_args(device)
        if testbed_args:
            logger.debug("Merging testbed connection args for %s: %s", device_name, testbed_args)
            connect_args.update(testbed_args)

        logger.debug("Final connection args for %s: %s", device_name, connect_args)
        device.connect(**connect_args)
        logger.info("Connected to %s", device_name)

    if _CONN_CACHE_TTL > 0:
        with _STATE_LOCK:
            _conn_cache[device_name] = {"device": device, "last_used": time.time()}

    return device


def _disconnect_device(device, force: bool = False) -> None:
    """
    Disconnect *device* unless connection caching is active (and force=False).

    When the connection cache is enabled, we just refresh the last-used
    timestamp instead of tearing down the session.
    """
    if not device:
        return
    if _CONN_CACHE_TTL > 0 and not force:
        with _STATE_LOCK:
            entry = _conn_cache.get(getattr(device, "name", ""))
            if entry is not None:
                entry["last_used"] = time.time()
        return
    if getattr(device, "is_connected", lambda: False)():
        try:
            logger.info("Disconnecting from %s", device.name)
            device.disconnect()
        except Exception as exc:
            logger.warning("Error disconnecting from %s: %s", device.name, exc)


# ---------------------------------------------------------------------------
# Show-command validation
# ---------------------------------------------------------------------------
_SHOW_BLOCK_CHARS: List[str] = ["|", ">", "<"]
_SHOW_BLOCK_WORDS: frozenset = frozenset(
    {"copy", "delete", "erase", "reload", "write", "configure", "conf"}
)


def validate_show_command(command: str) -> Optional[str]:
    """
    Return an error string if *command* is not a safe show command, else None.

    Blocks:
    - Commands that do not start with 'show'
    - Pipes and redirects (|, >, <)
    - Dangerous keywords (reload, erase, write, …)
    """
    cmd = (command or "").strip()
    if not cmd.lower().startswith("show"):
        return f"'{command}' is not a show command."
    if any(ch in cmd for ch in _SHOW_BLOCK_CHARS):
        return f"'{command}' contains a disallowed pipe or redirect character."
    for token in re.findall(r"[a-zA-Z0-9_-]+", cmd.lower()):
        if token in _SHOW_BLOCK_WORDS:
            return f"'{command}' contains disallowed term '{token}'."
    return None


# ---------------------------------------------------------------------------
# Config normalization + guardrails
# ---------------------------------------------------------------------------
_WRAPPER_LINES: frozenset = frozenset(
    {"configure terminal", "conf t", "config t", "configure t", "end"}
)


def _normalize_config_lines(config_commands: Union[str, List[Any], None]) -> List[str]:
    """
    Normalise a config payload into a clean list of CLI lines.

    Accepts a list of strings or a multiline string.  Strips wrapper commands
    (``configure terminal``, ``end``) that device.configure() adds itself, and
    splits semicolon-joined commands.  Indentation is preserved for submode
    commands.  ``exit`` is kept intentionally — it is needed to leave interface
    context blocks.
    """
    if config_commands is None:
        return []

    if isinstance(config_commands, list):
        raw_lines: List[str] = [str(x) for x in config_commands]
    else:
        raw_lines = textwrap.dedent(str(config_commands)).strip("\n").splitlines()

    out: List[str] = []
    for line in raw_lines:
        stripped = line.rstrip("\r\n")
        if not stripped.strip():
            continue
        # Expand semicolon-separated commands onto separate lines
        if ";" in stripped:
            for part in (p.strip() for p in stripped.split(";") if p.strip()):
                if part.lower() not in _WRAPPER_LINES:
                    out.append(part)
        elif stripped.strip().lower() not in _WRAPPER_LINES:
            out.append(stripped)

    return out


def _config_guardrails(config_lines: List[str]) -> Optional[str]:
    """
    Return an error string if any line contains a known-dangerous command.

    Catches: write erase, erase, reload, delete, format.
    """
    joined = "\n".join(config_lines).lower()
    dangerous = [
        (r"\bwrite\s+erase\b", "write erase"),
        (r"^\s*erase\b", "erase"),
        (r"\breload\b", "reload"),
        (r"\bdelete\b", "delete"),
        (r"\bformat\b", "format"),
    ]
    for pattern, label in dangerous:
        if re.search(pattern, joined, flags=re.MULTILINE):
            return f"Dangerous command detected: '{label}'. Operation aborted."
    return None


# ---------------------------------------------------------------------------
# Core blocking executors  (run in thread-pool via run_in_executor)
# ---------------------------------------------------------------------------

def _execute_show_command(device_name: str, command: str) -> Dict[str, Any]:
    """
    Execute *command* on *device_name*.

    Tries Genie structured parsing first; falls back to raw string output if
    no parser exists or parsing fails.
    """
    device = None
    try:
        device = _get_device(device_name)
        try:
            logger.info("Parsing '%s' on %s", command, device_name)
            return {
                "status": "completed", "device": device_name,
                "command": command, "output": device.parse(command), "parsed": True,
            }
        except Exception as exc:
            logger.warning("Parse failed for '%s' on %s (%s) — using raw output", command, device_name, exc)
            raw = device.execute(command)
            return {
                "status": "completed", "device": device_name, "command": command,
                "output": clean_output(raw) if isinstance(raw, str) else raw, "parsed": False,
            }
    except Exception as exc:
        logger.error("_execute_show_command failed: %s", exc, exc_info=True)
        return {"status": "error", "device": device_name, "command": command, "error": str(exc)}
    finally:
        _disconnect_device(device)


def _execute_show_raw(device_name: str, command: str) -> Dict[str, Any]:
    """
    Thin wrapper around _execute_show_command used by multi-device fan-out.

    Skips show-command validation so it can also be called by health checks
    using non-standard commands (ping, etc.).
    """
    return _execute_show_command(device_name, command)


def _execute_config(device_name: str, config_commands: Union[str, List[Any], None]) -> Dict[str, Any]:
    """
    Apply *config_commands* to *device_name* via device.configure().

    device.configure() handles entering/exiting config mode; callers must NOT
    include 'configure terminal' or 'end' in the payload.
    """
    device = None
    try:
        device = _get_device(device_name)
        lines = _normalize_config_lines(config_commands)
        if not lines:
            return {"status": "error", "device": device_name,
                    "error": "No configuration lines after normalisation."}
        guard = _config_guardrails(lines)
        if guard:
            return {"status": "error", "device": device_name, "error": guard}

        logger.info("Configuring %s: %s", device_name, lines)
        out = device.configure(lines)  # pass list — unicon handles submode correctly
        return {
            "status": "success", "device": device_name,
            "message": "Configuration applied successfully.",
            "commands_applied": lines,
            "output": clean_output(out) if isinstance(out, str) else out,
        }
    except Exception as exc:
        logger.error("_execute_config failed on %s: %s", device_name, exc, exc_info=True)
        return {"status": "error", "device": device_name, "error": str(exc)}
    finally:
        _disconnect_device(device)


def _execute_get_running_config_str(device_name: str) -> str:
    """
    Return the full running-config as a plain string.

    Used internally for diff generation and rollback snapshots.
    Raises on connection failure so callers can handle it explicitly.
    """
    device = None
    try:
        device = _get_device(device_name)
        device.enable()
        return clean_output(device.execute("show running-config"))
    finally:
        _disconnect_device(device)


def _execute_health(device_name: str) -> Dict[str, Any]:
    """
    Collect a structured health snapshot from *device_name*.

    Gathers version, interfaces, CPU, memory, BGP summary, and OSPF neighbors.
    Each section is attempted independently; missing keys mean the feature is
    not running or no parser exists for that OS.
    """
    device = None
    try:
        device = _get_device(device_name)
        snap: Dict[str, Any] = {"status": "completed", "device": device_name}

        def _collect(cmds: List[str], key: str) -> None:
            """Try each command in *cmds*; store parsed or raw result under *key*."""
            for cmd in cmds:
                try:
                    snap[key] = device.parse(cmd)
                    return
                except Exception as e:
                    logger.debug("Parse failed for %r on %s: %s", cmd, device_name, e)
                try:
                    snap[f"{key}_raw"] = clean_output(device.execute(cmd))
                    return
                except Exception as e:
                    logger.debug("Execute failed for %r on %s: %s", cmd, device_name, e)

        _collect(["show version", "show platform"], "version")
        _collect(["show interfaces summary", "show ip interface brief"], "interfaces")
        _collect(["show processes cpu"], "cpu")
        _collect(["show processes memory"], "memory")
        # Routing protocols are best-effort — not all devices run them
        _collect(["show ip bgp summary"], "bgp_summary")
        _collect(["show ip ospf neighbor"], "ospf_neighbors")

        return snap
    except Exception as exc:
        return {"status": "error", "device": device_name, "error": str(exc)}
    finally:
        _disconnect_device(device)


def _execute_learn_feature(device_name: str, feature: str) -> Dict[str, Any]:
    """
    Run Genie's device.learn(feature) and return the learned Ops data.

    Unlike pyats_run_show_command's per-command parsing, device.learn()
    gathers and normalizes everything Genie knows about one feature (e.g.
    'interface', 'ospf', 'bgp') across whatever show commands that feature's
    Ops model needs, vendor-neutrally.
    """
    device = None
    try:
        device = _get_device(device_name)
        ops = device.learn(feature)
        learned = getattr(ops, "info", None)
        if learned is None:
            learned = {k: v for k, v in vars(ops).items() if not k.startswith("_")}
        return {
            "status": "completed", "device": device_name, "feature": feature,
            "learned": learned,
        }
    except Exception as exc:
        return {"status": "error", "device": device_name, "feature": feature, "error": str(exc)}
    finally:
        _disconnect_device(device)


def _execute_get_neighbors(device_name: str) -> Dict[str, Any]:
    """
    Return a clean adjacency list from CDP or LLDP neighbor tables.

    Tries CDP first (Cisco-native), falls back to LLDP (vendor-neutral).
    Flattens the Genie-parsed structure into a simple list of dicts so the
    agent does not need to know Genie's internal schema.
    """
    device = None
    try:
        device = _get_device(device_name)
        cmds = [
            ("show cdp neighbors detail", "cdp"),
            ("show lldp neighbors detail", "lldp"),
            ("show cdp neighbors", "cdp"),
            ("show lldp neighbors", "lldp"),
        ]
        for cmd, proto in cmds:
            try:
                parsed = device.parse(cmd)
                neighbors = [
                    {
                        "neighbor": e.get("device_id") or e.get("system_name") or str(idx),
                        "local_interface": e.get("local_interface") or e.get("port_id", ""),
                        "remote_interface": e.get("port_id") or e.get("remote_port_id", ""),
                        "platform": e.get("platform", ""),
                        "ip": (
                            e.get("management_addresses") or e.get("interface_addresses") or {}
                        ).get("ipv4", ""),
                        "protocol": proto,
                    }
                    for idx, e in parsed.get("index", {}).items()
                ]
                return {
                    "status": "completed", "device": device_name,
                    "protocol": proto, "neighbors": neighbors, "raw_parsed": parsed,
                }
            except Exception:
                # Parser failed — try raw, return immediately with empty neighbors list
                try:
                    raw = clean_output(device.execute(cmd))
                    return {
                        "status": "completed", "device": device_name,
                        "protocol": proto, "neighbors": [], "raw_output": raw, "parsed": False,
                    }
                except Exception:
                    continue  # Try the next protocol

        return {"status": "error", "device": device_name,
                "error": "Neither CDP nor LLDP commands succeeded on this device."}
    except Exception as exc:
        return {"status": "error", "device": device_name, "error": str(exc)}
    finally:
        _disconnect_device(device)


def _execute_find_interface_by_ip(device_name: str, ip_address: str) -> Dict[str, Any]:
    """
    Search *device_name* for an interface whose address contains *ip_address*.

    Supports partial IP strings (e.g. '10.0.0' matches '10.0.0.1/24').
    Tries structured parsing first, falls back to raw text search.
    """
    device = None
    try:
        device = _get_device(device_name)
        matches: List[Dict[str, Any]] = []

        for cmd in ("show ip interface brief", "show interfaces"):
            try:
                parsed = device.parse(cmd)
                for intf, data in parsed.get("interface", {}).items():
                    addr = data.get("ip_address") or data.get("ipv4", {})
                    if isinstance(addr, dict):
                        for a in addr:
                            if ip_address in a:
                                matches.append({"device": device_name, "interface": intf, "address": a})
                    elif isinstance(addr, str) and ip_address in addr:
                        matches.append({"device": device_name, "interface": intf, "address": addr})
                if matches:
                    break
            except Exception:
                try:
                    raw = clean_output(device.execute(cmd))
                    for line in raw.splitlines():
                        if ip_address in line:
                            matches.append({"device": device_name, "line": line.strip()})
                    if matches:
                        break
                except Exception:
                    continue

        return {"status": "completed", "device": device_name,
                "ip_searched": ip_address, "matches": matches}
    except Exception as exc:
        return {"status": "error", "device": device_name, "error": str(exc)}
    finally:
        _disconnect_device(device)


# ---------------------------------------------------------------------------
# Async wrappers — offload blocking executors to the thread pool
# ---------------------------------------------------------------------------

async def _run_in_executor(fn, *args) -> Any:
    """Convenience wrapper: run *fn(*args)* in the default thread-pool executor."""
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(None, partial(fn, *args))


async def run_show_command_async(device_name: str, command: str) -> Dict[str, Any]:
    """Validate and execute a show command asynchronously."""
    err = validate_show_command(command)
    if err:
        return {"status": "error", "device": device_name, "command": command, "error": err}
    return await _run_in_executor(_execute_show_command, device_name, command)


async def apply_device_configuration_async(
    device_name: str, config_commands: Union[str, List[Any], None]
) -> Dict[str, Any]:
    """Apply *config_commands* to *device_name* asynchronously."""
    return await _run_in_executor(_execute_config, device_name, config_commands)


async def _apply_config_with_diff(
    device_name: str,
    config_commands: Union[str, List[Any], None],
    save_snapshot: bool = True,
) -> Dict[str, Any]:
    """
    Apply *config_commands* and return a unified diff of the running-config.

    Steps:
    1. Capture running-config BEFORE change (optionally save as rollback snapshot).
    2. Apply the config.
    3. Capture running-config AFTER change.
    4. Return unified diff alongside the normal apply result.
    """
    try:
        before = await _run_in_executor(_execute_get_running_config_str, device_name)
    except Exception as exc:
        return _err(
            "pyats_configure_with_diff", device_name, None,
            f"Failed to capture pre-config snapshot: {exc}",
            "Check connectivity with pyats_device_health first.",
        )

    if save_snapshot:
        with _STATE_LOCK:
            _config_snapshots[device_name] = before

    result = await apply_device_configuration_async(device_name, config_commands)
    if result.get("status") == "error":
        return result

    try:
        after = await _run_in_executor(_execute_get_running_config_str, device_name)
    except Exception as exc:
        result["diff_warning"] = f"Config applied but post-snapshot failed: {exc}"
        return result

    diff_lines = list(difflib.unified_diff(
        before.splitlines(keepends=True),
        after.splitlines(keepends=True),
        fromfile=f"{device_name}:before",
        tofile=f"{device_name}:after",
    ))
    result["diff"] = (
        "".join(diff_lines) if diff_lines else "(no diff — lines may already be present)"
    )
    result["snapshot_saved"] = save_snapshot
    return result


# ---------------------------------------------------------------------------
# Dynamic test execution (pyATS AEtest)
# ---------------------------------------------------------------------------
_BANNED_IMPORTS: frozenset = frozenset({
    "os", "sys", "subprocess", "shutil", "socket", "pathlib",
    "pickle", "yaml", "requests", "urllib", "http", "ssl",
})
_BANNED_PATTERNS: List[str] = [
    r"\b__import__\b", r"\beval\s*\(", r"\bexec\s*\(",
    r"\bcompile\s*\(", r"\bopen\s*\(", r"\bjson\.loads\s*\(",
]
_IMPORT_RE = re.compile(r"^\s*(import|from)\s+([a-zA-Z0-9_.]+)", re.MULTILINE)


def reject_unsafe_script(script: str) -> Optional[str]:
    """
    Return an error string if *script* contains unsafe imports or calls.

    Also enforces that the script defines TEST_DATA as a dict literal so all
    device data is embedded rather than fetched at runtime.
    """
    for match in _IMPORT_RE.finditer(script or ""):
        root = (match.group(2) or "").split(".")[0].lower()
        if root in _BANNED_IMPORTS:
            return f"Unsafe import blocked: '{root}'"
    for pattern in _BANNED_PATTERNS:
        if re.search(pattern, script or "", flags=re.IGNORECASE):
            return f"Unsafe pattern blocked: {pattern}"
    if "TEST_DATA" not in (script or ""):
        return "Script must define TEST_DATA as a Python dict literal (no json.loads)."
    return None


def _extract_overall_result(stdout: str) -> Optional[str]:
    """Parse the 'Result : PASSED/FAILED' line from pyATS job stdout."""
    match = re.search(r"Result\s+:\s+([A-Z]+)", stdout or "")
    return match.group(1) if match else None


_ARCHIVE_LINE_RE = re.compile(r"Archive\s*:\s*(\S+\.zip)")


def _extract_job_report(stdout: str) -> Dict[str, Any]:
    """
    Read the structured results.json out of the job's own archive zip.

    There is no working `--json-job <path>` flag in this pyATS version —
    it is silently accepted by the CLI's argument parser but never
    produces a report file (verified). pyATS does, however, always
    archive the run to a .zip under ~/.pyats/archive/ by default, and
    that archive contains a real results.json — this reads that instead.
    """
    match = _ARCHIVE_LINE_RE.search(stdout or "")
    if not match:
        return {"report": None, "archive_path": None}
    archive_path = match.group(1)
    try:
        with zipfile.ZipFile(archive_path) as zf:
            report = json.loads(zf.read("results.json"))
        return {"report": report, "archive_path": archive_path}
    except Exception as exc:
        logger.warning("Could not read results.json from archive %s: %s", archive_path, exc)
        return {"report": None, "archive_path": archive_path}


def _run_test_script(script_content: str, timeout_s: int = 300) -> Dict[str, Any]:
    """
    Write *script_content* to a temp directory, run it as a pyATS job, and
    return the structured result including stdout, stderr, and the JSON report.
    """
    ts = time.strftime("%Y%m%d_%H%M%S")
    run_dir = ARTIFACTS_DIR / f"run_{ts}_{os.getpid()}"
    run_dir.mkdir(parents=True, exist_ok=True)

    script_path = run_dir / "testscript.py"
    job_path = run_dir / "job.py"

    try:
        script_path.write_text(script_content, encoding="utf-8")
        job_path.write_text(
            f"from pyats.easypy import run\n"
            f"def main(runtime):\n"
            f"    run(testscript=r'{script_path}', runtime=runtime)\n",
            encoding="utf-8",
        )

        cmd = [shutil.which("pyats") or "pyats", "run", "job", str(job_path)]
        try:
            proc = subprocess.run(
                cmd, capture_output=True, text=True,
                env={**os.environ, "PYATS_TESTBED_PATH": TESTBED_PATH},
                timeout=timeout_s,
            )
        except subprocess.TimeoutExpired:
            return {"status": "error",
                    "error": f"pyATS job timed out after {timeout_s}s",
                    "artifacts_dir": str(run_dir)}

        report_info = _extract_job_report(proc.stdout)

        payload = {
            "status": "completed",
            "returncode": proc.returncode,
            "overall_result": _extract_overall_result(proc.stdout),
            "stdout": proc.stdout,
            "stderr": proc.stderr,
            "report": report_info["report"],
            "artifacts_dir": str(run_dir),
            "paths": {
                "script": str(script_path), "job": str(job_path),
                "archive": report_info["archive_path"],
            },
        }
        if not KEEP_ARTIFACTS:
            shutil.rmtree(run_dir, ignore_errors=True)
        return payload
    except Exception as exc:
        logger.error("_run_test_script failed: %s", exc, exc_info=True)
        return {"status": "error", "error": str(exc), "artifacts_dir": str(run_dir)}


# ---------------------------------------------------------------------------
# MCP server instance
# ---------------------------------------------------------------------------
mcp = MCPServer("pyATS Network Automation Server")


# ===========================================================================
# DEVICE DISCOVERY TOOLS
# ===========================================================================

@mcp.tool()
async def pyats_list_devices() -> str:
    """
    List every device in the testbed with its OS, type, platform, and connections.

    WHEN TO USE:
      Call this first when you do not know what devices are available.
      It gives you the exact names needed by all other tools.

    Returns:
        {
          "status": "completed",
          "devices": {
            "router-1": {"os": "iosxe", "type": "router", "platform": "...", "connections": [...]},
            ...
          }
        }
    """
    try:
        tb = _load_testbed()
        devices = {
            name: {
                "os": getattr(dev, "os", None),
                "type": getattr(dev, "type", None),
                "platform": getattr(dev, "platform", None),
                "connections": list(getattr(dev, "connections", {}).keys()),
            }
            for name, dev in tb.devices.items()
        }
        _log_op("pyats_list_devices", None, "list", "completed")
        return json.dumps({"status": "completed", "devices": devices}, indent=2)
    except Exception as exc:
        logger.error("pyats_list_devices failed: %s", exc, exc_info=True)
        return json.dumps(_err("pyats_list_devices", None, None, str(exc)), indent=2)


@mcp.tool()
async def pyats_search_devices(query: str, min_score: float = 0.4) -> str:
    """
    Find devices using a partial or approximate name — no exact match required.

    WHEN TO USE:
      Use instead of pyats_list_devices when you only know part of a device
      name (e.g. "core", "rtr-1", "border leaf").

    HOW SCORING WORKS:
      1. Exact substring match → score 1.0
      2. Token overlap (split on -_./space) → partial score
      3. Fuzzy SequenceMatcher ratio → catches typos / abbreviations
      The highest of the three scores is used.

    Args:
        query:     Partial or approximate device name.
        min_score: Minimum score threshold (0.0–1.0). Default 0.4.
                   Lower = more permissive; raise to 0.7+ for stricter matching.

    Returns:
        { "status": "completed", "query": "...", "matches": [ {name, score, os, ...}, ... ] }
        Results are sorted by score descending (best match first).
    """
    try:
        tb = _load_testbed()
        q = (query or "").strip().lower()
        if not q:
            return json.dumps(_err("pyats_search_devices", None, None,
                                   "query must not be empty."), indent=2)

        q_tokens = set(re.split(r"[\s\-_./]+", q))
        results = []

        for name, dev in tb.devices.items():
            name_lower = name.lower()
            if q in name_lower:
                score = 1.0
            else:
                name_tokens = set(re.split(r"[\s\-_./]+", name_lower))
                token_score = len(q_tokens & name_tokens) / max(len(q_tokens), 1)
                fuzzy_score = SequenceMatcher(None, q, name_lower).ratio()
                score = max(token_score, fuzzy_score)

            if score >= min_score:
                results.append({
                    "name": name, "score": round(score, 3),
                    "os": getattr(dev, "os", None),
                    "type": getattr(dev, "type", None),
                    "platform": getattr(dev, "platform", None),
                    "connections": list(getattr(dev, "connections", {}).keys()),
                })

        results.sort(key=lambda x: x["score"], reverse=True)
        _log_op("pyats_search_devices", None, query, "completed")
        return json.dumps({"status": "completed", "query": query, "matches": results}, indent=2)
    except Exception as exc:
        logger.error("pyats_search_devices failed: %s", exc, exc_info=True)
        return json.dumps(_err("pyats_search_devices", None, query, str(exc)), indent=2)


# ===========================================================================
# SHOW / READ TOOLS
# ===========================================================================

@mcp.tool()
async def pyats_run_show_command(
    device_name: str,
    command: str,
    timeout: int = 60,
    retries: int = 1,
) -> str:
    """
    Execute a show command on one device and return parsed or raw output.

    WHEN TO USE:
      Use for any 'show' command not covered by a dedicated tool.
      For 'show running-config' use pyats_show_running_config.
      For 'show logging' use pyats_show_logging.

    RETRY BEHAVIOUR:
      Each attempt runs independently with its own timeout.  On failure the
      tool waits 2 seconds before retrying.  As soon as one attempt succeeds
      it returns immediately.  The final response includes "attempts_made"
      when all attempts were exhausted.

    Args:
        device_name: Exact device name (use pyats_search_devices if unsure).
        command:     Must start with 'show'. No pipes, redirects, or
                     dangerous keywords (reload, delete, etc.).
        timeout:     Seconds to wait per attempt (default 60).
        retries:     Total attempts on failure/timeout (default 1 = no retry).

    Returns:
        { "status": "completed", "device": "...", "command": "...",
          "output": <dict if parsed, string if raw>, "parsed": true|false }
    """
    retries = max(retries, 1)
    last: Dict[str, Any] = {}

    for attempt in range(1, retries + 1):
        try:
            last = await asyncio.wait_for(
                run_show_command_async(device_name, command), timeout=timeout
            )
            if last.get("status") != "error":
                if attempt > 1:
                    last["attempt"] = attempt
                _log_op("pyats_run_show_command", device_name, command, "completed")
                return json.dumps(last, indent=2)
            logger.warning("Attempt %d/%d failed for '%s' on %s: %s",
                           attempt, retries, command, device_name, last.get("error"))
        except asyncio.TimeoutError:
            logger.warning("Attempt %d/%d timed out (%ds) for '%s' on %s",
                           attempt, retries, timeout, command, device_name)
            last = {"status": "error", "error": f"Timed out after {timeout}s."}
        except Exception as exc:
            logger.error("Attempt %d/%d error: %s", attempt, retries, exc, exc_info=True)
            last = {"status": "error", "error": str(exc)}

        if attempt < retries:
            await asyncio.sleep(2)

    last["attempts_made"] = retries
    _log_op("pyats_run_show_command", device_name, command, "error", last.get("error"))
    return json.dumps(last, indent=2)


@mcp.tool()
async def pyats_run_show_command_multi(device_names: List[str], command: str) -> str:
    """
    Run ONE show command across MULTIPLE devices in parallel.

    WHEN TO USE:
      Use instead of looping over pyats_run_show_command.  One call replaces N
      calls and runs all devices concurrently, saving significant time on large
      fleets.  A failure on one device does NOT stop the others.

    AGENT WORKFLOW:
      1. pyats_list_devices  →  collect names
      2. Filter to relevant subset
      3. pyats_run_show_command_multi  →  collect all results in one call

    Args:
        device_names: List of exact device names.
        command:      Show command to run on every device (same rules as
                      pyats_run_show_command — must start with 'show').

    Returns:
        {
          "status": "completed",
          "command": "show ip bgp summary",
          "summary": {"total": 3, "success": 2, "failed": 1},
          "results": [ {per-device result}, ... ]
        }
    """
    if not device_names:
        return json.dumps(_err("pyats_run_show_command_multi", None, command,
                               "device_names is empty.",
                               "Call pyats_list_devices to get valid names."), indent=2)
    err = validate_show_command(command)
    if err:
        return json.dumps(_err("pyats_run_show_command_multi", None, command, err), indent=2)

    try:
        loop = asyncio.get_running_loop()
        tasks = [loop.run_in_executor(None, partial(_execute_show_raw, name, command))
                 for name in device_names]
        results: List[Dict[str, Any]] = list(await asyncio.gather(*tasks))
        success = sum(1 for r in results if r.get("status") == "completed")
        for r in results:
            _log_op("pyats_run_show_command_multi", r.get("device"), command,
                    r.get("status", "error"), r.get("error"))
        return json.dumps({
            "status": "completed", "command": command,
            "summary": {"total": len(results), "success": success, "failed": len(results) - success},
            "results": results,
        }, indent=2)
    except Exception as exc:
        logger.error("pyats_run_show_command_multi failed: %s", exc, exc_info=True)
        return json.dumps(_err("pyats_run_show_command_multi", None, command, str(exc)), indent=2)


def _pcall_show_command(device_names: List[str], command: str) -> List[Dict[str, Any]]:
    """
    Run _execute_show_raw once per device, each in its own forked OS process
    via pyats.async_.pcall — a process-isolated alternative to the thread
    pool used by pyats_run_show_command_multi.

    Each child process is a fork() of this one, so it starts with its own
    private copy of _testbed_cache / _conn_cache / _STATE_LOCK (fork gives
    every child independent memory — nothing it does propagates back to the
    parent). It always connects and disconnects fresh; it never reuses a
    connection from the parent's _conn_cache. Prefer this over the
    thread-pool version for very large device counts where true OS-level
    isolation (one process crashing can't affect another) matters more than
    the extra fork overhead.
    """
    results = pcall(_execute_show_raw, iargs=[(name, command) for name in device_names])
    return list(results)


@mcp.tool()
async def pyats_pcall_show_command(device_names: List[str], command: str) -> str:
    """
    Run ONE show command across MULTIPLE devices, each in its own OS process
    (pyats.async_.pcall) rather than a shared thread pool.

    WHEN TO USE:
      Prefer pyats_run_show_command_multi for everyday fan-out — it has far
      less overhead. Reach for this tool instead when you specifically want
      process-level isolation across a large device count (e.g. a parser
      crash or a runaway command on one device cannot affect any other,
      since each device runs in its own forked process, not a shared
      thread pool).

    Args:
        device_names: List of exact device names.
        command:      Show command to run on every device (same rules as
                      pyats_run_show_command — must start with 'show').

    Returns:
        {
          "status": "completed",
          "command": "show ip bgp summary",
          "concurrency": "pcall (process per device)",
          "summary": {"total": 3, "success": 2, "failed": 1},
          "results": [ {per-device result}, ... ]
        }
    """
    if not device_names:
        return json.dumps(_err("pyats_pcall_show_command", None, command,
                               "device_names is empty.",
                               "Call pyats_list_devices to get valid names."), indent=2)
    err = validate_show_command(command)
    if err:
        return json.dumps(_err("pyats_pcall_show_command", None, command, err), indent=2)

    try:
        results: List[Dict[str, Any]] = await _run_in_executor(_pcall_show_command, device_names, command)
        success = sum(1 for r in results if r.get("status") == "completed")
        for r in results:
            _log_op("pyats_pcall_show_command", r.get("device"), command,
                    r.get("status", "error"), r.get("error"))
        return json.dumps({
            "status": "completed", "command": command,
            "concurrency": "pcall (process per device)",
            "summary": {"total": len(results), "success": success, "failed": len(results) - success},
            "results": results,
        }, indent=2)
    except Exception as exc:
        logger.error("pyats_pcall_show_command failed: %s", exc, exc_info=True)
        return json.dumps(_err("pyats_pcall_show_command", None, command, str(exc)), indent=2)


@mcp.tool()
async def pyats_show_running_config(device_name: str) -> str:
    """
    Retrieve the full running configuration from a device.

    WHEN TO USE:
      Use when you need the complete config for analysis or before making
      changes.  Returns raw text (not parsed).

    Args:
        device_name: Exact device name.

    Returns:
        { "status": "completed", "device": "...", "output": "<full running-config>" }
    """
    try:
        result = await _run_in_executor(
            lambda n: {
                "status": "completed", "device": n,
                "output": _execute_get_running_config_str(n),
            },
            device_name,
        )
        _log_op("pyats_show_running_config", device_name, "show running-config", "completed")
        return json.dumps(result, indent=2)
    except Exception as exc:
        logger.error("pyats_show_running_config failed: %s", exc, exc_info=True)
        return json.dumps(_err("pyats_show_running_config", device_name,
                               "show running-config", str(exc)), indent=2)


@mcp.tool()
async def pyats_show_logging(device_name: str) -> str:
    """
    Retrieve device system logs via 'show logging'.

    WHEN TO USE:
      Use when investigating errors, interface flaps, or protocol restarts.
      Returns raw text output.

    Args:
        device_name: Exact device name.

    Returns:
        { "status": "completed", "device": "...", "output": "<log text>" }
    """
    def _get_logs(name: str) -> Dict[str, Any]:
        device = None
        try:
            device = _get_device(name)
            device.enable()
            return {"status": "completed", "device": name,
                    "output": clean_output(device.execute("show logging"))}
        except Exception as exc:
            return {"status": "error", "device": name, "error": str(exc)}
        finally:
            _disconnect_device(device)

    try:
        result = await _run_in_executor(_get_logs, device_name)
        _log_op("pyats_show_logging", device_name, "show logging", result.get("status", "error"))
        return json.dumps(result, indent=2)
    except Exception as exc:
        logger.error("pyats_show_logging failed: %s", exc, exc_info=True)
        return json.dumps(_err("pyats_show_logging", device_name, "show logging", str(exc)), indent=2)


@mcp.tool()
async def pyats_device_health(device_name: str) -> str:
    """
    Collect a full structured health snapshot from a device in one call.

    WHEN TO USE:
      Call this FIRST when starting any investigation or troubleshooting task.
      It returns version, interfaces, CPU, memory, BGP, and OSPF in a single
      round-trip — replacing 5–6 individual show-command calls.

    WHAT IS COLLECTED (all best-effort; missing keys = feature not running):
      - version       : platform & software version
      - interfaces    : interface summary with IP and status
      - cpu           : process CPU utilization
      - memory        : process memory utilization
      - bgp_summary   : BGP peer table (omitted if BGP not configured)
      - ospf_neighbors: OSPF adjacency table (omitted if OSPF not configured)

    Each key is either a parsed dict or a <key>_raw string fallback.

    Args:
        device_name: Exact device name (use pyats_search_devices if unsure).

    Returns:
        { "status": "completed", "device": "...", "version": {...},
          "interfaces": {...}, "cpu": {...}, ... }

    NEXT STEPS:
      - Interface errors?  →  pyats_run_show_command("show interfaces <name>")
      - BGP down?          →  pyats_run_show_command("show ip bgp neighbors")
      - Want full config?  →  pyats_show_running_config
    """
    try:
        result = await _run_in_executor(_execute_health, device_name)
        _log_op("pyats_device_health", device_name, "health_snapshot",
                result.get("status", "error"), result.get("error"))
        return json.dumps(result, indent=2)
    except Exception as exc:
        logger.error("pyats_device_health failed: %s", exc, exc_info=True)
        return json.dumps(_err("pyats_device_health", device_name, None, str(exc)), indent=2)


@mcp.tool()
async def pyats_learn_feature(
    device_name: str,
    feature: str,
    snapshot_label: Optional[str] = None,
) -> str:
    """
    Run Genie's device.learn(feature) and optionally save the result as a
    named snapshot for later comparison with pyats_diff_learned_snapshots.

    WHEN TO USE:
      Use for a vendor-neutral, structured view of an entire feature (e.g.
      'interface', 'ospf', 'bgp', 'routing', 'platform') rather than parsing
      individual show commands one at a time. Also the first half of a
      before/after comparison — call once before a change with
      snapshot_label="before", again after with snapshot_label="after",
      then pyats_diff_learned_snapshots("before", "after").

    Args:
        device_name:    Exact device name.
        feature:        Genie feature name (e.g. "interface", "ospf", "bgp").
        snapshot_label: If given, store this learn result under this label
                        for this device+feature so it can be diffed later.
                        Labels are process-global and overwrite any prior
                        snapshot with the same device+feature+label.

    Returns:
        { "status": "completed", "device": "...", "feature": "ospf",
          "learned": {...}, "snapshot_saved": "before" }
    """
    try:
        result = await _run_in_executor(_execute_learn_feature, device_name, feature)
        if result.get("status") == "completed" and snapshot_label:
            with _STATE_LOCK:
                _learn_snapshots[f"{device_name}:{feature}:{snapshot_label}"] = result["learned"]
            result["snapshot_saved"] = snapshot_label
        _log_op("pyats_learn_feature", device_name, feature,
                result.get("status", "error"), result.get("error"))
        return json.dumps(result, indent=2)
    except Exception as exc:
        logger.error("pyats_learn_feature failed: %s", exc, exc_info=True)
        return json.dumps(_err("pyats_learn_feature", device_name, feature, str(exc)), indent=2)


@mcp.tool()
async def pyats_diff_learned_snapshots(
    device_name: str,
    feature: str,
    label_a: str,
    label_b: str,
) -> str:
    """
    Compare two previously saved pyats_learn_feature snapshots for one
    device+feature and return a unified diff.

    PRECONDITION:
      Both snapshots must already exist — call pyats_learn_feature twice
      first, once per label, for the same device_name and feature.

    Args:
        device_name: Exact device name.
        feature:     Genie feature name — must match what was learned.
        label_a:     Label of the "before" snapshot.
        label_b:     Label of the "after" snapshot.

    Returns:
        { "status": "completed", "device": "...", "feature": "ospf",
          "diff": "  neighbors:\\n-  10.0.0.1: up\\n+  10.0.0.1: down" }
      diff is empty string if the two snapshots are identical.
    """
    key_a = f"{device_name}:{feature}:{label_a}"
    key_b = f"{device_name}:{feature}:{label_b}"
    with _STATE_LOCK:
        snap_a = _learn_snapshots.get(key_a)
        snap_b = _learn_snapshots.get(key_b)

    missing = [lbl for lbl, snap in ((label_a, snap_a), (label_b, snap_b)) if snap is None]
    if missing:
        return json.dumps(_err(
            "pyats_diff_learned_snapshots", device_name, feature,
            f"No snapshot found for label(s): {', '.join(missing)}.",
            "Call pyats_learn_feature with snapshot_label set for each label first.",
        ), indent=2)

    try:
        diff = Diff(snap_a, snap_b)
        diff.findDiff()
        diff_text = str(diff)
        _log_op("pyats_diff_learned_snapshots", device_name, f"{feature}:{label_a}->{label_b}", "completed")
        return json.dumps({
            "status": "completed", "device": device_name, "feature": feature,
            "label_a": label_a, "label_b": label_b, "diff": diff_text,
        }, indent=2)
    except Exception as exc:
        logger.error("pyats_diff_learned_snapshots failed: %s", exc, exc_info=True)
        return json.dumps(_err("pyats_diff_learned_snapshots", device_name, feature, str(exc)), indent=2)


@mcp.tool()
async def pyats_ping_from_network_device(device_name: str, command: str) -> str:
    """
    Execute a ping from a network device and return parsed or raw results.

    WHEN TO USE:
      Use for connectivity testing FROM a device (not to it).
      Preferred over pyats_run_show_command for reachability checks because
      the server enters Privileged EXEC mode first.

    Args:
        device_name: Exact device name.
        command:     Ping command string, e.g. 'ping 1.1.1.1' or
                     'ping 1.1.1.1 repeat 100 source Loopback0'.

    Returns:
        Parsed JSON (success rate, RTT) when a Genie parser exists,
        otherwise raw text output.
        { "status": "completed", "device": "...", "command": "...",
          "output": <dict|str>, "parsed": true|false }
    """
    cmd = (command or "").strip()
    if not cmd.lower().startswith("ping"):
        return json.dumps(_err("pyats_ping_from_network_device", device_name, command,
                               f"'{command}' is not a ping command."), indent=2)

    def _ping(name: str, c: str) -> Dict[str, Any]:
        device = None
        try:
            device = _get_device(name)
            try:
                device.enable()
            except Exception as exc:
                logger.warning("Could not enable %s: %s", name, exc)
            try:
                return {"status": "completed", "device": name, "command": c,
                        "output": device.parse(c), "parsed": True}
            except Exception:
                return {"status": "completed", "device": name, "command": c,
                        "output": clean_output(device.execute(c)), "parsed": False}
        except Exception as exc:
            return {"status": "error", "device": name, "command": c, "error": str(exc)}
        finally:
            _disconnect_device(device)

    try:
        result = await _run_in_executor(_ping, device_name, cmd)
        _log_op("pyats_ping_from_network_device", device_name, cmd,
                result.get("status", "error"), result.get("error"))
        return json.dumps(result, indent=2)
    except Exception as exc:
        logger.error("pyats_ping_from_network_device failed: %s", exc, exc_info=True)
        return json.dumps(_err("pyats_ping_from_network_device", device_name, cmd, str(exc)), indent=2)


@mcp.tool()
async def pyats_get_neighbors(device_name: str) -> str:
    """
    Discover directly connected neighbors via CDP or LLDP.

    WHEN TO USE:
      Use when you need to know what is adjacent to a device — either to map
      the topology or to identify the next hop to investigate.

    HOW IT WORKS:
      Tries CDP first (Cisco-native), then LLDP (vendor-neutral).
      Returns a normalised adjacency list regardless of which protocol replies.

    Args:
        device_name: Exact device name.

    Returns:
        {
          "status": "completed", "device": "router-1", "protocol": "cdp",
          "neighbors": [
            { "neighbor": "switch-1", "local_interface": "Gi0/0",
              "remote_interface": "Gi1/0/24", "platform": "WS-C3850",
              "ip": "10.0.0.2", "protocol": "cdp" },
            ...
          ]
        }

    TOPOLOGY TRAVERSAL PATTERN:
      pyats_get_neighbors("router-1")       →  find neighbor "switch-1"
      pyats_search_devices("switch-1")      →  confirm exact name in testbed
      pyats_get_neighbors("switch-1")       →  continue hop by hop
    """
    try:
        result = await _run_in_executor(_execute_get_neighbors, device_name)
        _log_op("pyats_get_neighbors", device_name, "get_neighbors",
                result.get("status", "error"), result.get("error"))
        return json.dumps(result, indent=2)
    except Exception as exc:
        logger.error("pyats_get_neighbors failed: %s", exc, exc_info=True)
        return json.dumps(_err("pyats_get_neighbors", device_name, None, str(exc)), indent=2)


@mcp.tool()
async def pyats_find_interface_by_ip(
    ip_address: str,
    device_names: Optional[List[str]] = None,
) -> str:
    """
    Find which device and interface is associated with a given IP address.

    WHEN TO USE:
      Use when you have an IP and need to locate it in the network without
      knowing which device owns it.  Partial IP strings work (e.g. '10.0.0'
      will match '10.0.0.1/24').

    HOW IT WORKS:
      Queries all specified devices concurrently.  Tries structured parsing
      first; falls back to raw text search.  If device_names is omitted,
      searches every device in the testbed.

    Args:
        ip_address:   IP address or prefix to search for (partial match OK).
        device_names: Optional list of device names to search.
                      Omit to search all devices.

    Returns:
        {
          "status": "completed",
          "ip_searched": "10.0.0.1",
          "total_devices_searched": 5,
          "matches": [
            { "device": "router-1", "interface": "GigabitEthernet0/0", "address": "10.0.0.1/24" }
          ],
          "device_results": [ ... ]   ← per-device detail
        }

    NOTE:
      Empty matches means the IP is not assigned to any interface on the
      searched devices.  For routing lookups use:
        pyats_run_show_command(device, "show ip route <ip>")
    """
    try:
        tb = _load_testbed()
        names = device_names if device_names else list(tb.devices.keys())
        loop = asyncio.get_running_loop()
        tasks = [loop.run_in_executor(None, partial(_execute_find_interface_by_ip, n, ip_address))
                 for n in names]
        device_results: List[Dict[str, Any]] = list(await asyncio.gather(*tasks))
        all_matches = [m for dr in device_results for m in dr.get("matches", [])]
        _log_op("pyats_find_interface_by_ip", None, ip_address, "completed")
        return json.dumps({
            "status": "completed", "ip_searched": ip_address,
            "total_devices_searched": len(names),
            "matches": all_matches, "device_results": device_results,
        }, indent=2)
    except Exception as exc:
        logger.error("pyats_find_interface_by_ip failed: %s", exc, exc_info=True)
        return json.dumps(_err("pyats_find_interface_by_ip", None, ip_address, str(exc)), indent=2)


@mcp.tool()
async def pyats_run_linux_command(device_name: str, command: str) -> str:
    """
    Execute a Linux shell command on a Linux-based network device.

    WHEN TO USE:
      Use for Linux-based devices (e.g. iosxr, nxos in bash mode, servers)
      where standard IOS show commands do not apply.

    Args:
        device_name: Exact device name.
        command:     Shell command. Pipes and redirects are wrapped in
                     'sh -c "..."' automatically.

    Returns:
        { "status": "completed", "device": "...", "command": "...", "output": "..." }
    """
    def _linux(name: str, cmd: str) -> Dict[str, Any]:
        device = None
        try:
            tb = _load_testbed()
            if name not in tb.devices:
                return {"status": "error", "device": name,
                        "error": f"Device '{name}' not found in testbed."}
            device = tb.devices[name]
            if not device.is_connected():
                device.connect()
            # Wrap piped/redirected commands for shell execution
            exec_cmd = f'sh -c "{cmd}"' if (">" in cmd or "|" in cmd) else cmd
            try:
                output = device.parse(exec_cmd) if get_parser(exec_cmd, device) else device.execute(exec_cmd)
            except Exception:
                output = device.execute(exec_cmd)
            return {"status": "completed", "device": name, "command": cmd,
                    "output": clean_output(output) if isinstance(output, str) else output}
        except Exception as exc:
            return {"status": "error", "device": name, "error": str(exc)}
        finally:
            _disconnect_device(device)

    try:
        result = await _run_in_executor(_linux, device_name, command)
        _log_op("pyats_run_linux_command", device_name, command,
                result.get("status", "error"), result.get("error"))
        return json.dumps(result, indent=2)
    except Exception as exc:
        logger.error("pyats_run_linux_command failed: %s", exc, exc_info=True)
        return json.dumps(_err("pyats_run_linux_command", device_name, command, str(exc)), indent=2)


# ===========================================================================
# CONFIGURATION TOOLS
# ===========================================================================

@mcp.tool()
async def pyats_configure_device(device_name: str, config_commands: Any) -> str:
    """
    Apply configuration to a device.

    WHEN TO USE:
      Use for simple config pushes where you do not need a diff or rollback.
      For audited/reversible changes use pyats_configure_with_diff instead.

    IMPORTANT RULES:
      - Do NOT include 'configure terminal', 'conf t', or 'end' — the server
        handles config mode entry/exit automatically.
      - Preserve indentation for submode commands (interface, router ospf, etc.).
      - Dangerous commands (reload, erase, delete, format) are blocked.

    Args:
        device_name:     Exact device name.
        config_commands: List of strings OR a multiline string.
            List:   ["interface Gi0/0", " description WAN", " no shutdown"]
            String: "interface Gi0/0\\n description WAN\\n no shutdown"

    Returns:
        { "status": "success", "device": "...", "commands_applied": [...], "output": "..." }
    """
    try:
        result = await apply_device_configuration_async(device_name, config_commands)
        _log_op("pyats_configure_device", device_name, str(config_commands)[:120],
                result.get("status", "error"), result.get("error"))
        return json.dumps(result, indent=2)
    except Exception as exc:
        logger.error("pyats_configure_device failed: %s", exc, exc_info=True)
        return json.dumps(_err("pyats_configure_device", device_name, None, str(exc)), indent=2)


@mcp.tool()
async def pyats_configure_with_diff(
    device_name: str,
    config_commands: Any,
    save_rollback_snapshot: bool = True,
) -> str:
    """
    Apply configuration AND return a unified diff of what changed.

    WHEN TO USE:
      Prefer this over pyats_configure_device whenever you want to:
        - Verify exactly which lines were added or removed
        - Keep a rollback point in case something goes wrong
        - Produce a change audit trail

    HOW IT WORKS:
      1. Captures running-config BEFORE the change.
         (Saves as rollback snapshot when save_rollback_snapshot=True.)
      2. Applies the config.
      3. Captures running-config AFTER the change.
      4. Returns a unified diff ('+' = added, '-' = removed).

    ROLLBACK:
      If the diff shows unexpected lines, call pyats_rollback_config
      immediately to restore the pre-change state.

    Args:
        device_name:            Exact device name.
        config_commands:        Same format as pyats_configure_device.
        save_rollback_snapshot: Save a snapshot for rollback (default True).

    Returns:
        { "status": "success", ..., "diff": "+ip route ...", "snapshot_saved": true }
    """
    try:
        result = await _apply_config_with_diff(device_name, config_commands, save_rollback_snapshot)
        _log_op("pyats_configure_with_diff", device_name, str(config_commands)[:120],
                result.get("status", "error"), result.get("error"))
        return json.dumps(result, indent=2)
    except Exception as exc:
        logger.error("pyats_configure_with_diff failed: %s", exc, exc_info=True)
        return json.dumps(_err("pyats_configure_with_diff", device_name, None, str(exc)), indent=2)


@mcp.tool()
async def pyats_rollback_config(device_name: str) -> str:
    """
    Restore a device to the state captured before the last pyats_configure_with_diff call.

    WHEN TO USE:
      Call immediately when pyats_configure_with_diff produced unexpected
      changes or when a subsequent show command reveals a problem.

    PRECONDITION:
      A snapshot must exist for the device.  It is created automatically
      when pyats_configure_with_diff is called with save_rollback_snapshot=True
      (the default).  If no snapshot exists this tool returns an error.

    Args:
        device_name: Exact device name to roll back.

    Returns:
        { "status": "success", "device": "...", "message": "Rollback applied.",
          "snapshot_lines": 142 }

    AFTER ROLLBACK:
      Confirm success with pyats_show_running_config or pyats_device_health.
    """
    with _STATE_LOCK:
        snapshot = _config_snapshots.get(device_name)

    if snapshot is None:
        return json.dumps(_err(
            "pyats_rollback_config", device_name, None,
            f"No rollback snapshot found for '{device_name}'.",
            "Snapshots are saved automatically by pyats_configure_with_diff. "
            "If you used pyats_configure_device no snapshot was created.",
        ), indent=2)
    # Strip comment lines before re-applying
    lines = [l for l in snapshot.splitlines() if l.strip() and not l.strip().startswith("!")]

    try:
        result = await apply_device_configuration_async(device_name, lines)
        result["message"] = "Rollback applied successfully."
        result["snapshot_lines"] = len(lines)
        _log_op("pyats_rollback_config", device_name, "rollback",
                result.get("status", "error"), result.get("error"))
        return json.dumps(result, indent=2)
    except Exception as exc:
        logger.error("pyats_rollback_config failed: %s", exc, exc_info=True)
        return json.dumps(_err("pyats_rollback_config", device_name, None, str(exc)), indent=2)


@mcp.tool()
async def pyats_configure_devices_multi(
    device_names: List[str],
    config_commands: Any,
) -> str:
    """
    Push the SAME configuration to MULTIPLE devices in parallel.

    WHEN TO USE:
      Use for fleet-wide uniform changes:
        - Adding an NTP server to all devices
        - Pushing a new ACL or prefix-list everywhere
        - Updating SNMP community strings
      A failure on one device does NOT prevent others from being configured.

    IMPORTANT:
      - The identical config is sent to every device — only use for changes
        that are truly uniform across all target devices.
      - Verify OS compatibility first with pyats_list_devices (IOS vs NX-OS
        syntax differs).
      - For changes that need diff/rollback, use pyats_configure_with_diff
        individually per device instead.

    Args:
        device_names:    List of exact device names.
        config_commands: Config payload applied identically to all devices.
                         Same format rules as pyats_configure_device.

    Returns:
        {
          "status": "completed",
          "summary": {"total": 3, "success": 2, "failed": 1},
          "results": [ {per-device result}, ... ]
        }
    """
    if not device_names:
        return json.dumps(_err("pyats_configure_devices_multi", None, None,
                               "device_names is empty.",
                               "Call pyats_list_devices to get valid names."), indent=2)
    try:
        loop = asyncio.get_running_loop()
        tasks = [loop.run_in_executor(None, partial(_execute_config, n, config_commands))
                 for n in device_names]
        results: List[Dict[str, Any]] = list(await asyncio.gather(*tasks))
        success = sum(1 for r in results if r.get("status") == "success")
        for r in results:
            _log_op("pyats_configure_devices_multi", r.get("device"),
                    str(config_commands)[:80], r.get("status", "error"), r.get("error"))
        return json.dumps({
            "status": "completed",
            "summary": {"total": len(results), "success": success, "failed": len(results) - success},
            "results": results,
        }, indent=2)
    except Exception as exc:
        logger.error("pyats_configure_devices_multi failed: %s", exc, exc_info=True)
        return json.dumps(_err("pyats_configure_devices_multi", None, None, str(exc)), indent=2)


def _pcall_configure_devices(device_names: List[str], config_commands: Any) -> List[Dict[str, Any]]:
    """Process-isolated sibling of pyats_configure_devices_multi — see _pcall_show_command."""
    results = pcall(_execute_config, iargs=[(name, config_commands) for name in device_names])
    return list(results)


@mcp.tool()
async def pyats_pcall_configure_devices(
    device_names: List[str],
    config_commands: Any,
) -> str:
    """
    Push the SAME configuration to MULTIPLE devices, each in its own OS
    process (pyats.async_.pcall) rather than a shared thread pool.

    WHEN TO USE:
      Same use case as pyats_configure_devices_multi (uniform fleet-wide
      config push) — reach for this variant instead when you want process-
      level isolation across a large device count, at the cost of fork
      overhead per device. Guardrails (_config_guardrails) still apply
      identically inside each child process.

    Args:
        device_names:    List of exact device names.
        config_commands: Config payload applied identically to all devices.
                         Same format rules as pyats_configure_device.

    Returns:
        {
          "status": "completed",
          "concurrency": "pcall (process per device)",
          "summary": {"total": 3, "success": 2, "failed": 1},
          "results": [ {per-device result}, ... ]
        }
    """
    if not device_names:
        return json.dumps(_err("pyats_pcall_configure_devices", None, None,
                               "device_names is empty.",
                               "Call pyats_list_devices to get valid names."), indent=2)
    try:
        results: List[Dict[str, Any]] = await _run_in_executor(
            _pcall_configure_devices, device_names, config_commands
        )
        success = sum(1 for r in results if r.get("status") == "success")
        for r in results:
            _log_op("pyats_pcall_configure_devices", r.get("device"),
                    str(config_commands)[:80], r.get("status", "error"), r.get("error"))
        return json.dumps({
            "status": "completed",
            "concurrency": "pcall (process per device)",
            "summary": {"total": len(results), "success": success, "failed": len(results) - success},
            "results": results,
        }, indent=2)
    except Exception as exc:
        logger.error("pyats_pcall_configure_devices failed: %s", exc, exc_info=True)
        return json.dumps(_err("pyats_pcall_configure_devices", None, None, str(exc)), indent=2)


# ===========================================================================
# REST / RESTCONF TOOL
#
# Separate from the CLI/SSH tools above — uses pyATS's rest.connector.Rest
# instead of Unicon. Requires the testbed device to define its own 'rest'
# connection block (class: rest.connector.Rest); does NOT reuse _get_device
# / _conn_cache, which are Unicon/CLI-specific. RESTCONF's connect() is a
# no-op per the connector's own docs (no real handshake), so there is no
# connection-caching complexity to add here — each call just reuses
# device.rest's underlying requests.Session if already connected, or
# connects once on first use.
# ===========================================================================
_REST_METHODS: frozenset = frozenset({"GET", "POST", "PUT", "PATCH", "DELETE"})


def _get_rest_device(device_name: str):
    """Return device.rest, connecting via the testbed's 'rest' alias if needed."""
    tb = _load_testbed()
    device = tb.devices.get(device_name)
    if not device:
        raise ValueError(
            f"Device '{device_name}' not found in testbed. "
            "Use pyats_list_devices or pyats_search_devices to find valid names."
        )
    connections = getattr(device, "connections", {}) or {}
    if "rest" not in connections:
        raise ValueError(
            f"Device '{device_name}' has no 'rest' connection block in the testbed. "
            "Add connections.rest with class: rest.connector.Rest (see README)."
        )
    conn = getattr(device, "rest", None)
    if conn is None or not getattr(conn, "connected", False):
        device.connect(alias="rest", via="rest")
    return device.rest


def _execute_rest_request(
    device_name: str,
    method: str,
    api_url: str,
    payload: Optional[str] = None,
    content_type: Optional[str] = None,
    headers: Optional[Dict[str, str]] = None,
    timeout: int = 30,
) -> Dict[str, Any]:
    method_u = (method or "").upper()
    if method_u not in _REST_METHODS:
        return {"status": "error", "device": device_name, "method": method, "api_url": api_url,
                "error": f"Unsupported method '{method}'. Use one of {sorted(_REST_METHODS)}."}
    try:
        rest = _get_rest_device(device_name)
        kwargs: Dict[str, Any] = {"timeout": timeout}
        if content_type:
            kwargs["content_type"] = content_type
        if headers:
            kwargs["headers"] = headers

        if method_u == "GET":
            resp = rest.get(api_url, **kwargs)
        elif method_u == "DELETE":
            resp = rest.delete(api_url, **kwargs)
        else:
            fn = {"POST": rest.post, "PUT": rest.put, "PATCH": rest.patch}[method_u]
            resp = fn(api_url, payload=payload or "", **kwargs)

        body_text = resp.text
        try:
            body: Any = json.loads(body_text) if body_text else None
        except ValueError:
            body = body_text

        return {
            "status": "completed", "device": device_name, "method": method_u, "api_url": api_url,
            "status_code": resp.status_code, "body": body,
        }
    except Exception as exc:
        return {"status": "error", "device": device_name, "method": method_u, "api_url": api_url,
                "error": str(exc)}


@mcp.tool()
async def pyats_rest_request(
    device_name: str,
    method: str,
    api_url: str,
    payload: Optional[str] = None,
    content_type: Optional[str] = None,
    headers: Optional[Dict[str, str]] = None,
    timeout: int = 30,
) -> str:
    """
    Make a generic REST/RESTCONF API call against a device using pyATS's
    REST connector, for devices or controllers better automated over a
    REST API than the CLI (RESTCONF on IOS-XE, NX-API on NX-OS, or any
    other generic JSON/XML REST endpoint pyATS's rest.connector supports).

    PRECONDITION:
      The device's testbed entry must define a 'rest' connection block:
        connections:
          rest:
            class: rest.connector.Rest
            ip: <address>
            port: "443"
            protocol: https
            credentials:
              rest:
                username: ...
                password: ...

    Args:
        device_name:  Exact device name (must have a 'rest' connection block).
        method:       One of GET, POST, PUT, PATCH, DELETE.
        api_url:      Path portion of the URL, e.g.
                      "/restconf/data/ietf-interfaces:interfaces".
        payload:      JSON/XML body string for POST/PUT/PATCH (ignored for
                      GET/DELETE).
        content_type: "json" or "xml" (defaults to the connector's default).
        headers:      Extra HTTP headers as a dict.
        timeout:      Request timeout in seconds (default 30).

    Returns:
        { "status": "completed", "device": "...", "method": "GET",
          "api_url": "...", "status_code": 200, "body": {...} }
    """
    if not (device_name or "").strip():
        return json.dumps(_err("pyats_rest_request", device_name, api_url, "device_name is empty."), indent=2)
    if not (api_url or "").strip():
        return json.dumps(_err("pyats_rest_request", device_name, api_url, "api_url is empty."), indent=2)
    try:
        result = await _run_in_executor(
            _execute_rest_request, device_name, method, api_url, payload, content_type, headers, timeout
        )
        _log_op("pyats_rest_request", device_name, f"{method} {api_url}",
                result.get("status", "error"), result.get("error"))
        return json.dumps(result, indent=2)
    except Exception as exc:
        logger.error("pyats_rest_request failed: %s", exc, exc_info=True)
        return json.dumps(_err("pyats_rest_request", device_name, api_url, str(exc)), indent=2)


# ===========================================================================
# DEVICE CLEAN TOOL (Genie / Kleenex)
#
# Genie's clean/Kleenex engine has no supported public in-process API — the
# DeviceClean class is an internal implementation detail tightly coupled to
# pyats.aetest's global executer state, normally driven only by the `pyats
# clean` CLI. That CLI *is* a fully supported entry point, so this tool
# shells out to it (subprocess, timeout-bounded) rather than reaching into
# genie.libs.clean internals directly.
#
# Genie's real clean stage catalog (ChangeBootVariable, Reload, WriteErase,
# InstallImage, CopyToDevice, ...) is inherently destructive — that is the
# point of "clean" (staged device reset/reprovisioning). This tool does not
# expose those stages. It only ever generates a clean.yaml with the
# 'connect' and 'execute_command' stages (schema confirmed against the
# installed genie.libs.clean package's own test fixtures), so the worst
# this tool can do is run read/exec-style commands on the device through
# the clean framework — the same commands are still screened by
# _config_guardrails. dry_run defaults to True, and running for real
# requires an explicit literal confirm string.
# ===========================================================================
_CLEAN_CONFIRM_PHRASE = "I UNDERSTAND THIS IS A REAL DEVICE OPERATION"


def _build_clean_yaml(device_name: str, commands: List[str]) -> str:
    """Build a minimal, non-destructive clean.yaml (connect + execute_command only)."""
    doc = {
        "cleaners": {
            "DeviceClean": {
                "module": "genie.libs.clean",
                "devices": [device_name],
            },
        },
        "devices": {
            device_name: {
                "connect": None,
                "execute_command": {"commands": commands},
                "order": ["connect", "execute_command"],
            },
        },
    }
    return yaml.safe_dump(doc, sort_keys=False)


def _execute_clean_device(
    device_name: str, commands: List[str], timeout_s: int = 300
) -> Dict[str, Any]:
    """Write the generated clean.yaml and run `pyats clean` against it."""
    ts = time.strftime("%Y%m%d_%H%M%S")
    run_dir = ARTIFACTS_DIR / f"clean_{ts}_{os.getpid()}"
    run_dir.mkdir(parents=True, exist_ok=True)
    clean_path = run_dir / "clean.yaml"

    try:
        clean_yaml = _build_clean_yaml(device_name, commands)
        clean_path.write_text(clean_yaml, encoding="utf-8")

        cmd = [
            shutil.which("pyats") or "pyats", "clean",
            "--testbed-file", TESTBED_PATH,
            "--clean-file", str(clean_path),
            "--clean-devices", device_name,
            "--no-mail",
        ]
        try:
            proc = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout_s)
        except subprocess.TimeoutExpired:
            return {"status": "error", "device": device_name,
                    "error": f"pyats clean timed out after {timeout_s}s",
                    "clean_yaml": clean_yaml, "artifacts_dir": str(run_dir)}

        payload = {
            "status": "completed" if proc.returncode == 0 else "error",
            "device": device_name,
            "returncode": proc.returncode,
            "stdout": proc.stdout,
            "stderr": proc.stderr,
            "clean_yaml": clean_yaml,
            "artifacts_dir": str(run_dir),
        }
        if proc.returncode != 0:
            payload["error"] = f"pyats clean exited with code {proc.returncode}"
        if not KEEP_ARTIFACTS:
            shutil.rmtree(run_dir, ignore_errors=True)
        return payload
    except Exception as exc:
        logger.error("_execute_clean_device failed: %s", exc, exc_info=True)
        return {"status": "error", "device": device_name, "error": str(exc), "artifacts_dir": str(run_dir)}


@mcp.tool()
async def pyats_clean_device(
    device_name: str,
    commands: List[str],
    dry_run: bool = True,
    confirm: Optional[str] = None,
) -> str:
    """
    Run a conservative Genie Clean (Kleenex) stage sequence against a device:
    connect, then execute a list of commands. Real Genie clean stages that
    reboot, erase, or reimage a device are NOT exposed by this tool — only
    'connect' + 'execute_command' are ever generated.

    WHEN TO USE:
      Use to validate the Genie clean/Kleenex pipeline itself (testbed
      wiring, clean-file schema, CLI invocation) against a real device
      without risking a reload or config wipe — e.g. as a smoke test before
      trusting clean in a bigger workflow, or to run a batch of commands
      through the clean framework specifically (as opposed to
      pyats_run_show_command, which does not use Kleenex at all).

    SAFETY:
      - dry_run=True (default): returns the generated clean.yaml and does
        NOT touch the device or spawn any subprocess.
      - dry_run=False: requires confirm to exactly equal
        "I UNDERSTAND THIS IS A REAL DEVICE OPERATION", then shells out to
        `pyats clean` for real. Commands are still screened by the same
        guardrails as pyats_configure_device (reload/erase/delete/format
        are blocked).

    Args:
        device_name: Exact device name.
        commands:    List of commands to run via the clean execute_command
                     stage (e.g. ["show version", "show boot"]).
        dry_run:     If True (default), only generate and return the
                     clean.yaml — no device contact.
        confirm:     Required literal string when dry_run=False:
                     "I UNDERSTAND THIS IS A REAL DEVICE OPERATION"

    Returns:
        { "status": "completed", "device": "...", "returncode": 0,
          "stdout": "...", "clean_yaml": "...", "artifacts_dir": "..." }
    """
    if not commands:
        return json.dumps(_err("pyats_clean_device", device_name, None,
                               "commands is empty."), indent=2)
    guard = _config_guardrails(commands)
    if guard:
        return json.dumps(_err("pyats_clean_device", device_name, None, guard), indent=2)

    if dry_run:
        clean_yaml = _build_clean_yaml(device_name, commands)
        _log_op("pyats_clean_device", device_name, "dry_run", "completed")
        return json.dumps({
            "status": "completed", "device": device_name, "dry_run": True,
            "clean_yaml": clean_yaml,
            "message": "Dry run only — no subprocess spawned, device not contacted.",
        }, indent=2)

    if confirm != _CLEAN_CONFIRM_PHRASE:
        return json.dumps(_err(
            "pyats_clean_device", device_name, None,
            "confirm did not match the required phrase.",
            f"Pass confirm=\"{_CLEAN_CONFIRM_PHRASE}\" to run for real, or leave dry_run=True.",
        ), indent=2)

    try:
        result = await _run_in_executor(_execute_clean_device, device_name, commands, 300)
        _log_op("pyats_clean_device", device_name, "clean_execute_command",
                result.get("status", "error"), result.get("error"))
        return json.dumps(result, indent=2)
    except Exception as exc:
        logger.error("pyats_clean_device failed: %s", exc, exc_info=True)
        return json.dumps(_err("pyats_clean_device", device_name, None, str(exc)), indent=2)


# ===========================================================================
# TESTING TOOL
# ===========================================================================

@mcp.tool()
async def pyats_run_dynamic_test(test_script_content: str) -> str:
    """
    Execute a standalone pyATS AEtest script for programmatic validation.

    WHEN TO USE:
      Use when you need structured PASS/FAIL validation logic that goes beyond
      a simple show command — e.g. comparing BGP peer counts, verifying route
      prefixes, checking interface error thresholds.

    CRITICAL REQUIREMENTS:
      - Script must NOT connect to devices.  Embed all data in TEST_DATA.
      - Script must define TEST_DATA as a Python dict literal (no json.loads).
      - The following imports are blocked for security:
          os, sys, subprocess, shutil, socket, pathlib, pickle,
          yaml, requests, urllib, http, ssl
      - eval(), exec(), compile(), open() are also blocked.

    Args:
        test_script_content: Complete pyATS AEtest script as a string.
                             Must contain 'TEST_DATA = {...}'.

    Returns:
        { "status": "completed", "overall_result": "PASSED|FAILED",
          "returncode": 0, "stdout": "...", "stderr": "...",
          "report": {...}, "artifacts_dir": "/path/to/run_dir" }
    """
    if not (test_script_content or "").strip():
        return json.dumps(_err("pyats_run_dynamic_test", None, None,
                               "Empty test script content."), indent=2)
    reason = reject_unsafe_script(test_script_content)
    if reason:
        return json.dumps(_err("pyats_run_dynamic_test", None, None, reason), indent=2)
    try:
        result = await _run_in_executor(_run_test_script, test_script_content, 300)
        _log_op("pyats_run_dynamic_test", None, "dynamic_test",
                result.get("status", "error"), result.get("error"))
        return json.dumps(result, indent=2)
    except Exception as exc:
        logger.error("pyats_run_dynamic_test failed: %s", exc, exc_info=True)
        return json.dumps(_err("pyats_run_dynamic_test", None, None, str(exc)), indent=2)


# ===========================================================================
# DECLARATIVE TEST TOOL (pyATS Blitz)
#
# Blitz (genie.libs.sdk.triggers.blitz.blitz.Blitz) has no supported
# in-process API — its class imports the global pyats.easypy.runtime
# singleton, so like Genie clean it must run inside a real job/easypy
# runner. This tool generates a trigger datafile + job file and shells out
# to `pyats run job`, the same subprocess-with-timeout shape as
# pyats_run_dynamic_test/_run_test_script above — including the structured
# report, read via _extract_job_report() from the job's own archive zip
# rather than the CLI's `--json-job` flag (verified dead: silently
# accepted, never produces a report file in this pyATS version).
# ===========================================================================
_BLITZ_TRIGGER_NAME = "PyatsMcpBlitz"


def _blitz_guardrails(actions_yaml: str) -> Optional[str]:
    """Best-effort denylist scan over raw blitz YAML text, same spirit as _config_guardrails."""
    lowered = (actions_yaml or "").lower()
    dangerous = [
        (r"\bwrite\s+erase\b", "write erase"),
        (r"\breload\b", "reload"),
        (r"\berase\b", "erase"),
        (r"\bformat\b", "format"),
    ]
    for pattern, label in dangerous:
        if re.search(pattern, lowered, flags=re.MULTILINE):
            return f"Dangerous command detected in blitz actions: '{label}'. Operation aborted."
    return None


def _build_scoped_testbed(device_names: List[str], run_dir: Path) -> str:
    """
    Write a trimmed copy of the real testbed containing only *device_names*.

    genie.harness's common_setup (used by `pyats run job`) connects to
    EVERY device in whatever testbed it's given, not just the ones a
    specific trigger targets — so a full-testbed run fails outright the
    moment any device in the testbed (even one this call doesn't care
    about) is unreachable. Scoping the testbed file itself is the fix.

    Any %ENV{...} placeholders are left untouched (this is a plain YAML
    read/filter/write, not a pyats.topology.loader.load() — substitution
    still happens normally when the scoped file is loaded by the
    subprocess, which inherits this process's environment).
    """
    with open(TESTBED_PATH, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)
    all_devices = raw.get("devices", {}) or {}
    raw["devices"] = {name: all_devices[name] for name in device_names if name in all_devices}
    scoped_path = run_dir / "scoped_testbed.yaml"
    scoped_path.write_text(yaml.safe_dump(raw, sort_keys=False), encoding="utf-8")
    return str(scoped_path)


def _run_blitz(actions_yaml: str, device_names: List[str], timeout_s: int = 300) -> Dict[str, Any]:
    """
    Wrap *actions_yaml* (a YAML list matching Blitz's test_sections schema)
    into a trigger datafile and run it via `pyats run job` + genie.harness's
    gRun, against a testbed scoped to just *device_names*.
    """
    ts = time.strftime("%Y%m%d_%H%M%S")
    run_dir = ARTIFACTS_DIR / f"blitz_{ts}_{os.getpid()}"
    run_dir.mkdir(parents=True, exist_ok=True)
    trigger_path = run_dir / "trigger_datafile.yaml"
    job_path = run_dir / "blitz_job.py"

    try:
        try:
            sections = yaml.safe_load(actions_yaml)
        except Exception as exc:
            return {"status": "error", "error": f"actions_yaml is not valid YAML: {exc}",
                    "artifacts_dir": str(run_dir)}
        if not isinstance(sections, list):
            return {"status": "error",
                    "error": "actions_yaml must parse to a YAML list (Blitz test_sections).",
                    "artifacts_dir": str(run_dir)}

        trigger_doc = {
            _BLITZ_TRIGGER_NAME: {
                "source": {"pkg": "genie.libs.sdk", "class": "triggers.blitz.blitz.Blitz"},
                "devices": device_names,
                "test_sections": sections,
            },
        }
        trigger_path.write_text(yaml.safe_dump(trigger_doc, sort_keys=False), encoding="utf-8")
        job_path.write_text(
            "from genie.harness.main import gRun\n"
            "def main(runtime):\n"
            f"    gRun(trigger_datafile=r'{trigger_path}', trigger_uids=['{_BLITZ_TRIGGER_NAME}'])\n",
            encoding="utf-8",
        )
        scoped_testbed_path = _build_scoped_testbed(device_names, run_dir)

        cmd = [shutil.which("pyats") or "pyats", "run", "job", str(job_path),
               "--testbed-file", scoped_testbed_path, "--no-mail"]
        try:
            proc = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout_s)
        except subprocess.TimeoutExpired:
            return {"status": "error", "error": f"blitz job timed out after {timeout_s}s",
                    "artifacts_dir": str(run_dir)}

        report_info = _extract_job_report(proc.stdout)
        payload = {
            "status": "completed",
            "returncode": proc.returncode,
            "overall_result": _extract_overall_result(proc.stdout),
            "stdout": proc.stdout,
            "stderr": proc.stderr,
            "report": report_info["report"],
            "trigger_datafile": str(trigger_path),
            "archive": report_info["archive_path"],
            "artifacts_dir": str(run_dir),
        }
        if not KEEP_ARTIFACTS:
            shutil.rmtree(run_dir, ignore_errors=True)
        return payload
    except Exception as exc:
        logger.error("_run_blitz failed: %s", exc, exc_info=True)
        return {"status": "error", "error": str(exc), "artifacts_dir": str(run_dir)}


@mcp.tool()
async def pyats_run_blitz(actions_yaml: str, device_names: List[str]) -> str:
    """
    Run a declarative pyATS Blitz test against one or more devices.

    WHEN TO USE:
      Multi-step, declarative device workflows (execute, configure, parse,
      learn, and more, chained across named steps) expressed as data rather
      than code — e.g. "run show version, then configure an ACL, then
      verify it applied." For pure PASS/FAIL logic over already-known data
      use pyats_run_dynamic_test instead; for one ad-hoc command use
      pyats_run_show_command.

    Args:
        actions_yaml: YAML text for Blitz's test_sections — a list of named
                      steps, each a list of actions. Example:
                        - step1:
                          - execute:
                              device: R1
                              command: show version
        device_names: Devices this blitz trigger runs against (the
                      trigger's top-level 'devices:' list).

    Returns:
        { "status": "completed", "overall_result": "PASSED|FAILED",
          "returncode": 0, "stdout": "...", "stderr": "...",
          "trigger_datafile": "/path/...", "artifacts_dir": "..." }
    """
    if not (actions_yaml or "").strip():
        return json.dumps(_err("pyats_run_blitz", None, None, "actions_yaml is empty."), indent=2)
    if not device_names:
        return json.dumps(_err("pyats_run_blitz", None, None,
                               "device_names is empty.",
                               "Call pyats_list_devices to get valid names."), indent=2)
    guard = _blitz_guardrails(actions_yaml)
    if guard:
        return json.dumps(_err("pyats_run_blitz", None, None, guard), indent=2)

    try:
        result = await _run_in_executor(_run_blitz, actions_yaml, device_names, 300)
        _log_op("pyats_run_blitz", ",".join(device_names), "blitz",
                result.get("status", "error"), result.get("error"))
        return json.dumps(result, indent=2)
    except Exception as exc:
        logger.error("pyats_run_blitz failed: %s", exc, exc_info=True)
        return json.dumps(_err("pyats_run_blitz", None, None, str(exc)), indent=2)


# ===========================================================================
# ROBOT FRAMEWORK TOOL
#
# This pyATS version has no `pyats robot` CLI subcommand — pyats.robot and
# genie.libs.robot are Robot Framework *libraries*, imported from inside
# the .robot suite itself (verified installed: robotframework 7.4.2,
# pyats.robot.pyATSRobot, genie.libs.robot.GenieRobot). So this tool runs
# the suite via the standalone `robot` CLI (subprocess, timeout-bounded),
# the same shape as the other external-runner tools above. Keyword syntax
# below is confirmed against the installed libraries' @keyword decorators,
# not guessed: 'Use Testbed "${testbed}"', 'Connect To Device "${device}"',
# 'Parse "${parser}" on device "${device}"', 'Learn "${feature}" on device
# "${device}"'.
# ===========================================================================

def _robot_guardrails(script: str) -> Optional[str]:
    """Best-effort denylist scan over raw Robot script text, same spirit as _config_guardrails."""
    lowered = (script or "").lower()
    dangerous = [
        (r"\bwrite\s+erase\b", "write erase"),
        (r"\breload\b", "reload"),
        (r"\berase\b", "erase"),
        (r"\bformat\b", "format"),
    ]
    for pattern, label in dangerous:
        if re.search(pattern, lowered, flags=re.MULTILINE):
            return f"Dangerous command detected in robot script: '{label}'. Operation aborted."
    return None


def _run_robot_script(script_content: str, timeout_s: int = 300) -> Dict[str, Any]:
    """
    Write *script_content* to a .robot file — substituting the literal
    token {{TESTBED_PATH}} with this server's real, absolute testbed
    path — and run it via the `robot` CLI.
    """
    ts = time.strftime("%Y%m%d_%H%M%S")
    run_dir = ARTIFACTS_DIR / f"robot_{ts}_{os.getpid()}"
    run_dir.mkdir(parents=True, exist_ok=True)
    suite_path = run_dir / "suite.robot"

    try:
        rendered = script_content.replace("{{TESTBED_PATH}}", TESTBED_PATH)
        suite_path.write_text(rendered, encoding="utf-8")

        cmd = [shutil.which("robot") or "robot", "--outputdir", str(run_dir), str(suite_path)]
        try:
            proc = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout_s)
        except subprocess.TimeoutExpired:
            return {"status": "error", "error": f"robot run timed out after {timeout_s}s",
                    "artifacts_dir": str(run_dir)}

        payload = {
            "status": "completed",
            "overall_result": "PASSED" if proc.returncode == 0 else "FAILED",
            "returncode": proc.returncode,
            "stdout": proc.stdout,
            "stderr": proc.stderr,
            "artifacts_dir": str(run_dir),
            "paths": {
                "suite": str(suite_path),
                "output_xml": str(run_dir / "output.xml"),
                "log_html": str(run_dir / "log.html"),
                "report_html": str(run_dir / "report.html"),
            },
        }
        if not KEEP_ARTIFACTS:
            shutil.rmtree(run_dir, ignore_errors=True)
        return payload
    except Exception as exc:
        logger.error("_run_robot_script failed: %s", exc, exc_info=True)
        return {"status": "error", "error": str(exc), "artifacts_dir": str(run_dir)}


@mcp.tool()
async def pyats_run_robot(robot_script_content: str) -> str:
    """
    Run a Robot Framework test suite with pyATS/Genie keyword libraries
    against the real testbed.

    WHEN TO USE:
      Keyword-driven, non-Python test suites — a good fit if your team
      already standardizes on Robot Framework for authoring/reporting.
      For Python logic use pyats_run_dynamic_test; for declarative YAML
      actions use pyats_run_blitz.

    HOW IT WORKS:
      Writes robot_script_content to a .robot file and runs it via the
      standalone `robot` CLI (subprocess, timeout-bounded). Example
      suite content:

        *** Settings ***
        Library    pyats.robot.pyATSRobot
        Library    genie.libs.robot.GenieRobot

        *** Test Cases ***
        Check Version
            Use Testbed "{{TESTBED_PATH}}"
            Connect To Device "R1"
            ${result}=    Parse "show version" on device "R1"
            Disconnect From Device "R1"

      The literal token {{TESTBED_PATH}} is substituted with this
      server's real, absolute testbed path before the suite runs — you
      never need to know or hard-code that path yourself.

      IMPORTANT — Robot's embedded-argument keywords (Use Testbed "...",
      Connect To Device "...", Parse "..." on device "...", etc.) must
      have exactly ONE space before each quoted part, not the usual
      multi-space/tab column separator — extra spaces make Robot split
      it into a bogus multi-cell call and fail with "No keyword with
      name '...' found" (verified against the installed library).

    Args:
        robot_script_content: Complete .robot suite text (Settings +
                              Test Cases sections).

    Returns:
        { "status": "completed", "overall_result": "PASSED|FAILED",
          "returncode": 0, "stdout": "...", "stderr": "...",
          "artifacts_dir": "...", "paths": {"output_xml": "...", ...} }
    """
    if not (robot_script_content or "").strip():
        return json.dumps(_err("pyats_run_robot", None, None, "robot_script_content is empty."), indent=2)
    guard = _robot_guardrails(robot_script_content)
    if guard:
        return json.dumps(_err("pyats_run_robot", None, None, guard), indent=2)
    try:
        result = await _run_in_executor(_run_robot_script, robot_script_content, 300)
        _log_op("pyats_run_robot", None, "robot_suite",
                result.get("status", "error"), result.get("error"))
        return json.dumps(result, indent=2)
    except Exception as exc:
        logger.error("pyats_run_robot failed: %s", exc, exc_info=True)
        return json.dumps(_err("pyats_run_robot", None, None, str(exc)), indent=2)


# ===========================================================================
# XPRESSO REST API TOOL
#
# CAUTION — UNVERIFIED: built directly from Cisco's published XPresso REST
# API v2 documentation (Authorization: Jwt <token> + Group: <name> headers,
# offset/limit pagination), but not exercised against a live XPresso
# instance — none was available while writing this tool. Treat the first
# real call against your XPresso server as the actual verification step,
# and expect to adjust header/path details if your instance's behavior
# differs from the published docs.
# ===========================================================================

_XPRESSO_METHODS: frozenset = frozenset({"GET", "POST", "PUT", "PATCH", "DELETE"})


def _execute_xpresso_request(
    method: str,
    path: str,
    payload: Optional[Dict[str, Any]] = None,
    params: Optional[Dict[str, Any]] = None,
    timeout: int = 30,
) -> Dict[str, Any]:
    if not XPRESSO_URL or not XPRESSO_API_TOKEN or not XPRESSO_GROUP:
        return {"status": "error",
                "error": "XPresso is not configured — set XPRESSO_URL, XPRESSO_API_TOKEN, "
                         "and XPRESSO_GROUP in .env."}
    method_u = (method or "").upper()
    if method_u not in _XPRESSO_METHODS:
        return {"status": "error",
                "error": f"Unsupported method '{method}'. Use one of {sorted(_XPRESSO_METHODS)}."}

    url = f"{XPRESSO_URL}{path if path.startswith('/') else '/' + path}"
    headers = {
        "Authorization": f"Jwt {XPRESSO_API_TOKEN}",
        "Group": XPRESSO_GROUP,
    }
    try:
        resp = requests.request(
            method_u, url, headers=headers, params=params,
            json=payload if payload is not None else None, timeout=timeout,
        )
        try:
            body: Any = resp.json()
        except ValueError:
            body = resp.text
        return {
            "status": "completed", "method": method_u, "url": url,
            "status_code": resp.status_code, "body": body,
        }
    except Exception as exc:
        return {"status": "error", "method": method_u, "url": url, "error": str(exc)}


@mcp.tool()
async def pyats_xpresso_request(
    method: str,
    path: str,
    payload: Optional[Dict[str, Any]] = None,
    params: Optional[Dict[str, Any]] = None,
    timeout: int = 30,
) -> str:
    """
    Make an authenticated call to Cisco XPresso's REST API v2 (test
    request/bundle submission and retrieval, job/bundle/profile lookup,
    lab equipment — testbeds, clean instructions, topologies — test
    harness/execution engine lookup, and image retrieval/pull/build).

    UNVERIFIED: built from XPresso's published REST API v2 docs, not
    tested against a live instance. Validate your first real call
    carefully.

    PRECONDITION:
      Set in .env: XPRESSO_URL (e.g. https://xpresso.example.com),
      XPRESSO_API_TOKEN (your API automation token, from XPresso's
      Profile > API Token menu), XPRESSO_GROUP (your XPresso group name).

    Args:
        method:  One of GET, POST, PUT, PATCH, DELETE.
        path:    API path, e.g. "/api/v2/testbeds" or "/api/v2/requests".
        payload: JSON body dict for POST/PUT/PATCH (ignored for GET/DELETE).
        params:  Query-string params dict — XPresso list/search endpoints
                 use offset/limit pagination, e.g. {"offset": 50, "limit": 100}.
        timeout: Request timeout in seconds (default 30).

    Returns:
        { "status": "completed", "method": "GET", "url": "...",
          "status_code": 200, "body": {...} }
    """
    if not (path or "").strip():
        return json.dumps(_err("pyats_xpresso_request", None, path, "path is empty."), indent=2)
    try:
        result = await _run_in_executor(_execute_xpresso_request, method, path, payload, params, timeout)
        _log_op("pyats_xpresso_request", None, f"{method} {path}",
                result.get("status", "error"), result.get("error"))
        return json.dumps(result, indent=2)
    except Exception as exc:
        logger.error("pyats_xpresso_request failed: %s", exc, exc_info=True)
        return json.dumps(_err("pyats_xpresso_request", None, path, str(exc)), indent=2)


# ===========================================================================
# SESSION / AUDIT TOOL
# ===========================================================================

@mcp.tool()
async def pyats_get_operation_log(
    limit: int = 50,
    device_filter: Optional[str] = None,
) -> str:
    """
    Return a log of every tool call made in the current server session.

    WHEN TO USE:
      - Review what has already been tried before deciding the next step.
      - Avoid re-running commands already executed earlier in the session.
      - Check whether a previous config change succeeded or failed.
      - Produce an audit trail of all operations performed during a session.

    HOW IT WORKS:
      Every tool in this server writes an entry to an in-memory log on each
      call.  The log persists for the lifetime of the server process and is
      cleared on restart.  Entries are in chronological order, newest last.

    Args:
        limit:         Maximum entries to return (default 50, hard cap 500).
        device_filter: Return only entries for this device name.
                       Omit or pass null to return entries for all devices.

    Returns:
        {
          "status": "completed",
          "total_entries": 24,
          "returned": 10,
          "log": [
            { "ts": "2024-01-15T10:30:00Z", "tool": "pyats_run_show_command",
              "device": "router-1", "detail": "show ip bgp summary",
              "status": "completed" },
            ...
          ]
        }

    AGENT PATTERN:
      At the start of a long session call this tool to check what was already
      investigated and avoid redundant round-trips.
    """
    with _STATE_LOCK:
        entries = _OP_LOG[:]
        total_entries = len(_OP_LOG)
    if device_filter:
        entries = [e for e in entries if e.get("device") == device_filter]
    entries = entries[-min(limit, _OP_LOG_MAX):]
    return json.dumps({
        "status": "completed",
        "total_entries": total_entries,
        "returned": len(entries),
        "filter_device": device_filter,
        "log": entries,
    }, indent=2)


# ---------------------------------------------------------------------------
# Entry point
#
# Streamable HTTP only (STDIO removed). PYATS_MCP_TRANSPORT_MODE selects:
#   stateful  (default) — stateless_http=False, server retains HTTP-session
#                          state for clients still on the pre-SEP-2575 (legacy
#                          2025-06-18) handshake-based protocol.
#   stateless           — stateless_http=True, no HTTP-session state kept
#                          between requests even for legacy-protocol clients.
# Clients speaking the current 2026-07-28 protocol core are handshake-free
# and per-request-metadata-based regardless of this flag — that behavior
# comes from the mcp>=2.0.0 SDK itself, not from anything configured here.
# ---------------------------------------------------------------------------
_TRANSPORT_MODE: str = os.getenv("PYATS_MCP_TRANSPORT_MODE", "stateful").strip().lower()
if _TRANSPORT_MODE not in ("stateful", "stateless"):
    logger.warning(
        "Invalid PYATS_MCP_TRANSPORT_MODE=%r; defaulting to 'stateful'", _TRANSPORT_MODE
    )
    _TRANSPORT_MODE = "stateful"

_HTTP_HOST: str = os.getenv("PYATS_MCP_HTTP_HOST", "0.0.0.0")
_HTTP_PORT: int = _parse_int_env("PYATS_MCP_HTTP_PORT", 8080)

if __name__ == "__main__":
    logger.info(
        "Starting pyATS MCP Server — transport=streamable-http mode=%s host=%s port=%d",
        _TRANSPORT_MODE, _HTTP_HOST, _HTTP_PORT,
    )
    mcp.run(
        transport="streamable-http",
        host=_HTTP_HOST,
        port=_HTTP_PORT,
        json_response=True,
        stateless_http=(_TRANSPORT_MODE == "stateless"),
    )
