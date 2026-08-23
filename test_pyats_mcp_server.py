"""
test_pyats_mcp_server.py
========================
Unit tests for pyats_mcp_server.py.

All pyATS / Genie / device I/O is mocked so these tests run without a real
testbed or network connection.

Run with:
    pytest test_pyats_mcp_server.py -v
    pytest test_pyats_mcp_server.py -v --tb=short   # shorter tracebacks
"""

from __future__ import annotations

import asyncio
import importlib
import json
import os
import sys
import types
import unittest
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

# ---------------------------------------------------------------------------
# Stub out pyATS imports before the module is loaded so that tests can run
# without pyATS installed in the test environment.
# ---------------------------------------------------------------------------
def _make_stub_modules():
    """Register minimal stub modules for pyATS, Genie, dotenv, and FastMCP."""
    stubs = {
        "dotenv": types.ModuleType("dotenv"),
        "pyats": types.ModuleType("pyats"),
        "pyats.topology": types.ModuleType("pyats.topology"),
        "pyats.async_": types.ModuleType("pyats.async_"),
        "genie": types.ModuleType("genie"),
        "genie.libs": types.ModuleType("genie.libs"),
        "genie.libs.parser": types.ModuleType("genie.libs.parser"),
        "genie.libs.parser.utils": types.ModuleType("genie.libs.parser.utils"),
        "genie.utils": types.ModuleType("genie.utils"),
        "genie.utils.diff": types.ModuleType("genie.utils.diff"),
        "mcp": types.ModuleType("mcp"),
        "mcp.server": types.ModuleType("mcp.server"),
        "mcp.server.mcpserver": types.ModuleType("mcp.server.mcpserver"),
    }

    # dotenv
    stubs["dotenv"].load_dotenv = lambda: None  # type: ignore[attr-defined]

    # pyats.topology.loader
    mock_loader = MagicMock()
    stubs["pyats.topology"].loader = mock_loader  # type: ignore[attr-defined]

    # pyats.async_.pcall — real tests patch srv._pcall_show_command /
    # srv._pcall_configure_devices directly, so this stub only needs to exist.
    stubs["pyats.async_"].pcall = MagicMock()  # type: ignore[attr-defined]

    # genie parser utility
    stubs["genie.libs.parser.utils"].get_parser = MagicMock(return_value=None)  # type: ignore[attr-defined]

    # genie.utils.diff.Diff — real tests patch srv.Diff directly when needed.
    stubs["genie.utils.diff"].Diff = MagicMock()  # type: ignore[attr-defined]

    # MCPServer stub — decorators become no-ops that return the original function
    class _FakeMCPServer:
        def __init__(self, *args, **kwargs):
            pass
        def tool(self):
            def decorator(fn):
                return fn
            return decorator
        def run(self, *args, **kwargs):
            pass

    stubs["mcp.server.mcpserver"].MCPServer = _FakeMCPServer  # type: ignore[attr-defined]

    for name, mod in stubs.items():
        sys.modules.setdefault(name, mod)


_make_stub_modules()

# Patch out the startup guard (TESTBED_PATH validation) before importing
os.environ.setdefault("PYATS_TESTBED_PATH", "/fake/testbed.yaml")
with patch("os.path.exists", return_value=True):
    import pyats_mcp_server as srv


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _json(result: str) -> dict:
    """Parse the JSON string returned by every MCP tool."""
    return json.loads(result)


def _run(coro):
    """Run a coroutine synchronously."""
    return asyncio.run(coro)


def _mock_device(name: str = "router-1", connected: bool = True) -> MagicMock:
    """Return a minimal mock pyATS device."""
    dev = MagicMock()
    dev.name = name
    dev.os = "iosxe"
    dev.type = "router"
    dev.platform = "ISR4000"
    dev.connections = {"default": MagicMock()}
    dev.is_connected.return_value = connected
    return dev


def _mock_testbed(*device_names: str) -> MagicMock:
    """Return a mock testbed containing *device_names*."""
    tb = MagicMock()
    tb.devices = {n: _mock_device(n) for n in device_names}
    return tb


# ---------------------------------------------------------------------------
# Helper / pure-function tests  (no I/O)
# ---------------------------------------------------------------------------

class TestCleanOutput(unittest.TestCase):
    def test_strips_ansi(self):
        assert srv.clean_output("\x1b[32mhello\x1b[0m") == "hello"

    def test_strips_non_printable(self):
        assert srv.clean_output("hello\x00world") == "helloworld"

    def test_passthrough_plain(self):
        assert srv.clean_output("show ip interface brief") == "show ip interface brief"


class TestValidateShowCommand(unittest.TestCase):
    def test_valid_show(self):
        assert srv.validate_show_command("show ip route") is None

    def test_rejects_non_show(self):
        assert srv.validate_show_command("ping 1.1.1.1") is not None

    def test_rejects_pipe(self):
        assert srv.validate_show_command("show ip route | include 0.0.0.0") is not None

    def test_rejects_redirect(self):
        assert srv.validate_show_command("show version > file") is not None

    def test_rejects_dangerous_keyword(self):
        assert srv.validate_show_command("show reload") is not None

    def test_case_insensitive(self):
        assert srv.validate_show_command("SHOW VERSION") is None

    def test_empty_command(self):
        assert srv.validate_show_command("") is not None


class TestNormalizeConfigLines(unittest.TestCase):
    def test_list_input(self):
        result = srv._normalize_config_lines(["interface Gi0/0", " no shutdown"])
        assert result == ["interface Gi0/0", " no shutdown"]

    def test_strips_wrappers(self):
        result = srv._normalize_config_lines([
            "configure terminal", "ntp server 1.1.1.1", "end"
        ])
        assert "configure terminal" not in result
        assert "end" not in result
        assert "ntp server 1.1.1.1" in result

    def test_semicolon_splitting(self):
        result = srv._normalize_config_lines(["ntp server 1.1.1.1; ntp server 2.2.2.2"])
        assert result == ["ntp server 1.1.1.1", "ntp server 2.2.2.2"]

    def test_multiline_string(self):
        result = srv._normalize_config_lines("interface Gi0/0\n description WAN")
        assert result == ["interface Gi0/0", " description WAN"]

    def test_none_input(self):
        assert srv._normalize_config_lines(None) == []

    def test_preserves_exit(self):
        result = srv._normalize_config_lines(["interface Gi0/0", " exit"])
        assert "exit" in result[-1]

    def test_blank_lines_skipped(self):
        result = srv._normalize_config_lines(["ntp server 1.1.1.1", "", "   "])
        assert result == ["ntp server 1.1.1.1"]


class TestConfigGuardrails(unittest.TestCase):
    def test_blocks_write_erase(self):
        assert srv._config_guardrails(["write erase"]) is not None

    def test_blocks_reload(self):
        assert srv._config_guardrails(["reload"]) is not None

    def test_blocks_delete(self):
        assert srv._config_guardrails(["delete flash:config"]) is not None

    def test_blocks_format(self):
        assert srv._config_guardrails(["format bootflash:"]) is not None

    def test_allows_safe_commands(self):
        assert srv._config_guardrails(["ntp server 1.1.1.1"]) is None
        assert srv._config_guardrails(["interface Gi0/0", " no shutdown"]) is None

    def test_erase_standalone(self):
        assert srv._config_guardrails(["erase startup-config"]) is not None


class TestRejectUnsafeScript(unittest.TestCase):
    _base = "TEST_DATA = {}\n"

    def test_allows_safe_script(self):
        assert srv.reject_unsafe_script(self._base + "import re\n") is None

    def test_blocks_os_import(self):
        assert srv.reject_unsafe_script("import os\n" + self._base) is not None

    def test_blocks_subprocess(self):
        assert srv.reject_unsafe_script("import subprocess\n" + self._base) is not None

    def test_blocks_eval(self):
        assert srv.reject_unsafe_script(self._base + "eval('1+1')\n") is not None

    def test_blocks_exec(self):
        assert srv.reject_unsafe_script(self._base + "exec('pass')\n") is not None

    def test_requires_test_data(self):
        assert srv.reject_unsafe_script("import re\n") is not None

    def test_blocks_open(self):
        assert srv.reject_unsafe_script(self._base + "open('/etc/passwd')\n") is not None


class TestOperationLog(unittest.TestCase):
    def setUp(self):
        srv._OP_LOG.clear()

    def test_log_op_appends_entry(self):
        srv._log_op("test_tool", "router-1", "show version", "completed")
        assert len(srv._OP_LOG) == 1
        entry = srv._OP_LOG[0]
        assert entry["tool"] == "test_tool"
        assert entry["device"] == "router-1"
        assert entry["status"] == "completed"

    def test_log_op_includes_error(self):
        srv._log_op("test_tool", "router-1", "show version", "error", "Connection refused")
        assert srv._OP_LOG[0]["error"] == "Connection refused"

    def test_log_op_evicts_oldest_when_full(self):
        srv._OP_LOG_MAX = 3
        for i in range(5):
            srv._log_op("tool", "dev", f"cmd-{i}", "completed")
        assert len(srv._OP_LOG) == 3
        assert srv._OP_LOG[0]["detail"] == "cmd-2"  # oldest kept
        srv._OP_LOG_MAX = 500  # restore

    def test_err_builds_payload(self):
        payload = srv._err("my_tool", "router-1", "show ip route",
                           "Device unreachable", "Check connectivity first.")
        assert payload["status"] == "error"
        assert payload["tool"] == "my_tool"
        assert payload["suggestion"] == "Check connectivity first."
        assert len(srv._OP_LOG) == 1

    def test_err_without_suggestion(self):
        payload = srv._err("tool", None, None, "Something went wrong")
        assert "suggestion" not in payload


# ---------------------------------------------------------------------------
# Async tool tests
# ---------------------------------------------------------------------------

class TestPyatsListDevices(unittest.TestCase):
    def setUp(self):
        srv._OP_LOG.clear()

    def test_returns_device_list(self):
        tb = _mock_testbed("router-1", "router-2")
        with patch.object(srv, "_load_testbed", return_value=tb):
            result = _json(_run(srv.pyats_list_devices()))
        assert result["status"] == "completed"
        assert "router-1" in result["devices"]
        assert "router-2" in result["devices"]
        assert result["devices"]["router-1"]["os"] == "iosxe"

    def test_handles_exception(self):
        with patch.object(srv, "_load_testbed", side_effect=RuntimeError("disk error")):
            result = _json(_run(srv.pyats_list_devices()))
        assert result["status"] == "error"
        assert "disk error" in result["error"]


class TestPyatsSearchDevices(unittest.TestCase):
    def setUp(self):
        srv._OP_LOG.clear()
        self.tb = _mock_testbed("core-router-1", "core-router-2", "edge-switch-1")

    def test_substring_match(self):
        with patch.object(srv, "_load_testbed", return_value=self.tb):
            result = _json(_run(srv.pyats_search_devices("core")))
        assert result["status"] == "completed"
        names = [m["name"] for m in result["matches"]]
        assert "core-router-1" in names
        assert "core-router-2" in names
        assert "edge-switch-1" not in names

    def test_fuzzy_match(self):
        with patch.object(srv, "_load_testbed", return_value=self.tb):
            result = _json(_run(srv.pyats_search_devices("edge-swch", min_score=0.5)))
        names = [m["name"] for m in result["matches"]]
        assert "edge-switch-1" in names

    def test_empty_query_returns_error(self):
        with patch.object(srv, "_load_testbed", return_value=self.tb):
            result = _json(_run(srv.pyats_search_devices("")))
        assert result["status"] == "error"

    def test_sorted_by_score_descending(self):
        with patch.object(srv, "_load_testbed", return_value=self.tb):
            result = _json(_run(srv.pyats_search_devices("core-router-1")))
        scores = [m["score"] for m in result["matches"]]
        assert scores == sorted(scores, reverse=True)

    def test_min_score_filtering(self):
        with patch.object(srv, "_load_testbed", return_value=self.tb):
            # Very high threshold — only exact matches
            result = _json(_run(srv.pyats_search_devices("core-router-1", min_score=0.99)))
        names = [m["name"] for m in result["matches"]]
        assert names == ["core-router-1"]


class TestPyatsRunShowCommand(unittest.TestCase):
    def setUp(self):
        srv._OP_LOG.clear()

    def test_returns_parsed_output(self):
        parsed = {"interfaces": {"Gi0/0": {"ip": "10.0.0.1"}}}
        async def _fake_show(device_name, command):
            return {"status": "completed", "device": device_name,
                    "command": command, "output": parsed, "parsed": True}
        with patch.object(srv, "run_show_command_async", side_effect=_fake_show):
            result = _json(_run(srv.pyats_run_show_command("router-1", "show ip interface brief")))
        assert result["status"] == "completed"
        assert result["parsed"] is True

    def test_rejects_invalid_command(self):
        with patch.object(srv, "run_show_command_async") as mock_show:
            mock_show.return_value = {"status": "error", "error": "not a show command"}
            result = _json(_run(srv.pyats_run_show_command("router-1", "ping 1.1.1.1")))
        # validate_show_command should catch this
        assert result["status"] == "error"

    def test_timeout_returns_error(self):
        async def _slow(*_):
            await asyncio.sleep(100)
        with patch.object(srv, "run_show_command_async", side_effect=_slow):
            result = _json(_run(
                srv.pyats_run_show_command("router-1", "show version", timeout=1)
            ))
        assert result["status"] == "error"
        assert "timed out" in result["error"].lower()

    def test_retries_on_failure(self):
        call_count = {"n": 0}
        async def _flaky(device_name, command):
            call_count["n"] += 1
            if call_count["n"] < 3:
                return {"status": "error", "error": "temporary failure"}
            return {"status": "completed", "device": device_name,
                    "command": command, "output": "ok", "parsed": False}
        with patch.object(srv, "run_show_command_async", side_effect=_flaky):
            with patch("asyncio.sleep", new=AsyncMock()):  # skip back-off sleep
                result = _json(_run(
                    srv.pyats_run_show_command("router-1", "show version", retries=3)
                ))
        assert result["status"] == "completed"
        assert call_count["n"] == 3

    def test_retries_exhausted_returns_error(self):
        async def _always_fail(device_name, command):
            return {"status": "error", "error": "connection refused"}
        with patch.object(srv, "run_show_command_async", side_effect=_always_fail):
            with patch("asyncio.sleep", new=AsyncMock()):
                result = _json(_run(
                    srv.pyats_run_show_command("router-1", "show version", retries=2)
                ))
        assert result["status"] == "error"
        assert result["attempts_made"] == 2


class TestPyatsRunShowCommandMulti(unittest.TestCase):
    def setUp(self):
        srv._OP_LOG.clear()

    def test_runs_across_all_devices(self):
        def _fake_exec(name, cmd):
            return {"status": "completed", "device": name, "command": cmd,
                    "output": f"output-{name}", "parsed": False}
        with patch.object(srv, "_execute_show_raw", side_effect=_fake_exec):
            result = _json(_run(
                srv.pyats_run_show_command_multi(["r1", "r2", "r3"], "show version")
            ))
        assert result["summary"]["total"] == 3
        assert result["summary"]["success"] == 3

    def test_empty_device_list_returns_error(self):
        result = _json(_run(srv.pyats_run_show_command_multi([], "show version")))
        assert result["status"] == "error"

    def test_invalid_command_returns_error(self):
        result = _json(_run(
            srv.pyats_run_show_command_multi(["r1"], "configure terminal")
        ))
        assert result["status"] == "error"

    def test_partial_failure_reflected_in_summary(self):
        def _fake_exec(name, cmd):
            if name == "r2":
                return {"status": "error", "device": name, "error": "timeout"}
            return {"status": "completed", "device": name, "command": cmd,
                    "output": "ok", "parsed": False}
        with patch.object(srv, "_execute_show_raw", side_effect=_fake_exec):
            result = _json(_run(
                srv.pyats_run_show_command_multi(["r1", "r2"], "show version")
            ))
        assert result["summary"]["success"] == 1
        assert result["summary"]["failed"] == 1


class TestPyatsDeviceHealth(unittest.TestCase):
    def setUp(self):
        srv._OP_LOG.clear()

    def test_returns_snapshot(self):
        snapshot = {"status": "completed", "device": "router-1",
                    "version": {"v": "17.3"}, "interfaces": {}}
        with patch.object(srv, "_execute_health", return_value=snapshot):
            result = _json(_run(srv.pyats_device_health("router-1")))
        assert result["status"] == "completed"
        assert "version" in result

    def test_propagates_error(self):
        with patch.object(srv, "_execute_health",
                          return_value={"status": "error", "device": "r1", "error": "conn fail"}):
            result = _json(_run(srv.pyats_device_health("router-1")))
        assert result["status"] == "error"


class TestPyatsConfigureDevice(unittest.TestCase):
    def setUp(self):
        srv._OP_LOG.clear()

    def test_success_path(self):
        ok = {"status": "success", "device": "r1", "commands_applied": ["ntp server 1.1.1.1"]}
        with patch.object(srv, "apply_device_configuration_async",
                          new=AsyncMock(return_value=ok)):
            result = _json(_run(srv.pyats_configure_device("r1", ["ntp server 1.1.1.1"])))
        assert result["status"] == "success"

    def test_guardrails_block_dangerous(self):
        def _fake_config(name, cmds):
            return {"status": "error", "device": name,
                    "error": "Dangerous command detected: 'reload'"}
        with patch.object(srv, "apply_device_configuration_async",
                          new=AsyncMock(side_effect=_fake_config)):
            result = _json(_run(srv.pyats_configure_device("r1", ["reload"])))
        assert result["status"] == "error"


class TestPyatsConfigureWithDiff(unittest.TestCase):
    def setUp(self):
        srv._OP_LOG.clear()
        srv._config_snapshots.clear()

    def test_diff_returned(self):
        diff_result = {
            "status": "success", "device": "r1",
            "diff": "+ntp server 1.1.1.1\n", "snapshot_saved": True,
        }
        with patch.object(srv, "_apply_config_with_diff",
                          new=AsyncMock(return_value=diff_result)):
            result = _json(_run(
                srv.pyats_configure_with_diff("r1", ["ntp server 1.1.1.1"])
            ))
        assert result["status"] == "success"
        assert "diff" in result

    def test_snapshot_saved_flag(self):
        diff_result = {
            "status": "success", "device": "r1",
            "diff": "+ntp server 1.1.1.1\n", "snapshot_saved": True,
        }
        with patch.object(srv, "_apply_config_with_diff",
                          new=AsyncMock(return_value=diff_result)):
            result = _json(_run(
                srv.pyats_configure_with_diff("r1", ["ntp server 1.1.1.1"],
                                              save_rollback_snapshot=True)
            ))
        assert result["snapshot_saved"] is True


class TestPyatsRollbackConfig(unittest.TestCase):
    def setUp(self):
        srv._OP_LOG.clear()
        srv._config_snapshots.clear()

    def test_no_snapshot_returns_error(self):
        result = _json(_run(srv.pyats_rollback_config("router-1")))
        assert result["status"] == "error"
        assert "snapshot" in result["error"].lower()

    def test_rollback_applies_snapshot(self):
        srv._config_snapshots["router-1"] = "ntp server 1.1.1.1\n! comment\n"
        ok = {"status": "success", "device": "router-1", "commands_applied": ["ntp server 1.1.1.1"]}
        with patch.object(srv, "apply_device_configuration_async",
                          new=AsyncMock(return_value=ok)):
            result = _json(_run(srv.pyats_rollback_config("router-1")))
        assert result["status"] == "success"
        assert result["snapshot_lines"] == 1  # comment line stripped

    def test_rollback_strips_comment_lines(self):
        srv._config_snapshots["router-1"] = "! comment\nntp server 1.1.1.1\n! another\n"
        ok = {"status": "success", "device": "router-1",
              "commands_applied": ["ntp server 1.1.1.1"]}
        with patch.object(srv, "apply_device_configuration_async",
                          new=AsyncMock(return_value=ok)):
            result = _json(_run(srv.pyats_rollback_config("router-1")))
        assert result["snapshot_lines"] == 1


class TestPyatsGetNeighbors(unittest.TestCase):
    def setUp(self):
        srv._OP_LOG.clear()

    def test_returns_neighbor_list(self):
        neighbors_result = {
            "status": "completed", "device": "r1", "protocol": "cdp",
            "neighbors": [{"neighbor": "switch-1", "local_interface": "Gi0/0",
                           "remote_interface": "Gi1/0/1", "platform": "WS-C3850",
                           "ip": "10.0.0.2", "protocol": "cdp"}],
        }
        with patch.object(srv, "_execute_get_neighbors", return_value=neighbors_result):
            result = _json(_run(srv.pyats_get_neighbors("r1")))
        assert result["status"] == "completed"
        assert len(result["neighbors"]) == 1
        assert result["neighbors"][0]["neighbor"] == "switch-1"

    def test_error_propagated(self):
        with patch.object(srv, "_execute_get_neighbors",
                          return_value={"status": "error", "device": "r1", "error": "no cdp"}):
            result = _json(_run(srv.pyats_get_neighbors("r1")))
        assert result["status"] == "error"


class TestPyatsConfigureDevicesMulti(unittest.TestCase):
    def setUp(self):
        srv._OP_LOG.clear()

    def test_configures_all_devices(self):
        def _fake_exec(name, cmds):
            return {"status": "success", "device": name, "commands_applied": cmds}
        with patch.object(srv, "_execute_config", side_effect=_fake_exec):
            result = _json(_run(
                srv.pyats_configure_devices_multi(
                    ["r1", "r2"], ["ntp server 1.1.1.1"]
                )
            ))
        assert result["summary"]["total"] == 2
        assert result["summary"]["success"] == 2

    def test_empty_list_returns_error(self):
        result = _json(_run(srv.pyats_configure_devices_multi([], ["ntp server 1.1.1.1"])))
        assert result["status"] == "error"

    def test_partial_failure_counted(self):
        def _fake_exec(name, cmds):
            if name == "r2":
                return {"status": "error", "device": name, "error": "connection failed"}
            return {"status": "success", "device": name, "commands_applied": cmds}
        with patch.object(srv, "_execute_config", side_effect=_fake_exec):
            result = _json(_run(
                srv.pyats_configure_devices_multi(["r1", "r2"], ["ntp server 1.1.1.1"])
            ))
        assert result["summary"]["success"] == 1
        assert result["summary"]["failed"] == 1


class TestPyatsFindInterfaceByIp(unittest.TestCase):
    def setUp(self):
        srv._OP_LOG.clear()

    def test_finds_ip_on_device(self):
        match_result = {
            "status": "completed", "device": "r1",
            "ip_searched": "10.0.0.1",
            "matches": [{"device": "r1", "interface": "Gi0/0", "address": "10.0.0.1/24"}],
        }
        tb = _mock_testbed("r1")
        with patch.object(srv, "_load_testbed", return_value=tb):
            with patch.object(srv, "_execute_find_interface_by_ip", return_value=match_result):
                result = _json(_run(srv.pyats_find_interface_by_ip("10.0.0.1", ["r1"])))
        assert result["status"] == "completed"
        assert len(result["matches"]) == 1

    def test_no_match_returns_empty_list(self):
        no_match = {
            "status": "completed", "device": "r1",
            "ip_searched": "192.0.2.99", "matches": [],
        }
        tb = _mock_testbed("r1")
        with patch.object(srv, "_load_testbed", return_value=tb):
            with patch.object(srv, "_execute_find_interface_by_ip", return_value=no_match):
                result = _json(_run(srv.pyats_find_interface_by_ip("192.0.2.99", ["r1"])))
        assert result["matches"] == []

    def test_searches_all_devices_when_none_specified(self):
        tb = _mock_testbed("r1", "r2", "r3")
        no_match = lambda n, ip: {"status": "completed", "device": n,
                                   "ip_searched": ip, "matches": []}
        with patch.object(srv, "_load_testbed", return_value=tb):
            with patch.object(srv, "_execute_find_interface_by_ip", side_effect=no_match):
                result = _json(_run(srv.pyats_find_interface_by_ip("10.0.0.1")))
        assert result["total_devices_searched"] == 3


class TestPyatsPingFromNetworkDevice(unittest.TestCase):
    def setUp(self):
        srv._OP_LOG.clear()

    def test_valid_ping_command(self):
        def _fake_ping(name, cmd):
            return {"status": "completed", "device": name, "command": cmd,
                    "output": {"success_rate": 100}, "parsed": True}
        with patch.object(srv, "_run_in_executor",
                          new=AsyncMock(side_effect=lambda fn, *a: fn(*a))):
            with patch("pyats_mcp_server._execute_show_raw"):  # not used here
                pass
            # Patch the inner function directly
            with patch.object(srv, "_run_in_executor",
                               new=AsyncMock(return_value={
                                   "status": "completed", "device": "r1",
                                   "command": "ping 1.1.1.1",
                                   "output": {"success_rate": 100}, "parsed": True
                               })):
                result = _json(_run(srv.pyats_ping_from_network_device("r1", "ping 1.1.1.1")))
        assert result["status"] == "completed"

    def test_rejects_non_ping_command(self):
        result = _json(_run(srv.pyats_ping_from_network_device("r1", "show version")))
        assert result["status"] == "error"
        assert "not a ping" in result["error"].lower()


class TestPyatsGetOperationLog(unittest.TestCase):
    def setUp(self):
        srv._OP_LOG.clear()

    def test_returns_all_entries(self):
        srv._log_op("tool_a", "r1", "show version", "completed")
        srv._log_op("tool_b", "r2", "show logging", "completed")
        result = _json(_run(srv.pyats_get_operation_log()))
        assert result["status"] == "completed"
        assert result["returned"] == 2

    def test_limit_respected(self):
        for i in range(10):
            srv._log_op("tool", "r1", f"cmd-{i}", "completed")
        result = _json(_run(srv.pyats_get_operation_log(limit=3)))
        assert result["returned"] == 3

    def test_device_filter(self):
        srv._log_op("tool", "r1", "show version", "completed")
        srv._log_op("tool", "r2", "show version", "completed")
        result = _json(_run(srv.pyats_get_operation_log(device_filter="r1")))
        assert all(e["device"] == "r1" for e in result["log"])

    def test_newest_entries_returned_when_limited(self):
        for i in range(5):
            srv._log_op("tool", "r1", f"cmd-{i}", "completed")
        result = _json(_run(srv.pyats_get_operation_log(limit=2)))
        details = [e["detail"] for e in result["log"]]
        assert "cmd-4" in details
        assert "cmd-3" in details

    def test_empty_log(self):
        result = _json(_run(srv.pyats_get_operation_log()))
        assert result["returned"] == 0
        assert result["log"] == []


class TestPyatsRunDynamicTest(unittest.TestCase):
    def setUp(self):
        srv._OP_LOG.clear()

    _safe_script = "TEST_DATA = {}\nimport re\n"

    def test_empty_script_returns_error(self):
        result = _json(_run(srv.pyats_run_dynamic_test("")))
        assert result["status"] == "error"

    def test_unsafe_script_blocked(self):
        result = _json(_run(srv.pyats_run_dynamic_test("import os\nTEST_DATA = {}\n")))
        assert result["status"] == "error"
        assert "blocked" in result["error"].lower()

    def test_safe_script_runs(self):
        mock_result = {
            "status": "completed", "returncode": 0,
            "overall_result": "PASSED", "stdout": "", "stderr": "",
            "report": None, "artifacts_dir": "/tmp/run",
        }
        with patch.object(srv, "_run_test_script", return_value=mock_result):
            result = _json(_run(srv.pyats_run_dynamic_test(self._safe_script)))
        assert result["status"] == "completed"
        assert result["overall_result"] == "PASSED"


# ---------------------------------------------------------------------------
# Internal helper unit tests
# ---------------------------------------------------------------------------

class TestExecuteHealthInternals(unittest.TestCase):
    """Test the _execute_health helper with a mocked device."""

    def _make_device(self, parse_responses: dict, execute_responses: dict) -> MagicMock:
        dev = _mock_device()
        def _parse(cmd):
            if cmd in parse_responses:
                return parse_responses[cmd]
            raise Exception(f"No parser for {cmd}")
        def _execute(cmd):
            if cmd in execute_responses:
                return execute_responses[cmd]
            raise Exception(f"No execute for {cmd}")
        dev.parse = _parse
        dev.execute = _execute
        return dev

    def test_version_falls_back_to_raw(self):
        dev = self._make_device(
            parse_responses={},
            execute_responses={"show version": "Cisco IOS XE 17.3"},
        )
        with patch.object(srv, "_get_device", return_value=dev):
            with patch.object(srv, "_disconnect_device"):
                result = srv._execute_health("router-1")
        assert result["status"] == "completed"
        assert "version_raw" in result

    def test_parsed_version_preferred(self):
        dev = self._make_device(
            parse_responses={"show version": {"version": {"version": "17.3"}}},
            execute_responses={},
        )
        with patch.object(srv, "_get_device", return_value=dev):
            with patch.object(srv, "_disconnect_device"):
                result = srv._execute_health("router-1")
        assert "version" in result
        assert "version_raw" not in result

    def test_connection_error_returns_error_dict(self):
        with patch.object(srv, "_get_device", side_effect=ValueError("not found")):
            result = srv._execute_health("no-such-device")
        assert result["status"] == "error"
        assert "not found" in result["error"]


class TestExecuteGetNeighborsInternals(unittest.TestCase):
    def test_cdp_parsed_correctly(self):
        cdp_parsed = {
            "index": {
                1: {
                    "device_id": "switch-core-1",
                    "local_interface": "GigabitEthernet0/0",
                    "port_id": "GigabitEthernet1/0/24",
                    "platform": "WS-C3850",
                    "management_addresses": {"ipv4": "10.0.0.2"},
                }
            }
        }
        dev = _mock_device()
        dev.parse = lambda cmd: cdp_parsed if "cdp" in cmd else (_ for _ in ()).throw(Exception())

        with patch.object(srv, "_get_device", return_value=dev):
            with patch.object(srv, "_disconnect_device"):
                result = srv._execute_get_neighbors("router-1")

        assert result["status"] == "completed"
        assert result["protocol"] == "cdp"
        assert len(result["neighbors"]) == 1
        nb = result["neighbors"][0]
        assert nb["neighbor"] == "switch-core-1"
        assert nb["ip"] == "10.0.0.2"

    def test_falls_back_to_lldp(self):
        lldp_parsed = {"index": {1: {"system_name": "switch-2", "port_id": "Gi1/0/1",
                                     "local_interface": "Gi0/1", "platform": ""}}}

        def _parse(cmd):
            if "lldp" in cmd:
                return lldp_parsed
            raise Exception("no cdp")

        dev = _mock_device()
        dev.parse = _parse

        with patch.object(srv, "_get_device", return_value=dev):
            with patch.object(srv, "_disconnect_device"):
                result = srv._execute_get_neighbors("router-1")

        assert result["protocol"] == "lldp"


class TestNormalizeConfigEdgeCases(unittest.TestCase):
    def test_preserves_leading_spaces_for_submode(self):
        lines = _normalize_config_lines = srv._normalize_config_lines
        result = lines(["interface Gi0/0", "  ip address 10.0.0.1 255.255.255.0"])
        assert result[1].startswith("  ")

    def test_various_wrapper_variants_stripped(self):
        fn = srv._normalize_config_lines
        for wrapper in ["configure terminal", "conf t", "config t", "configure t", "end"]:
            result = fn([wrapper, "ntp server 1.1.1.1"])
            assert wrapper not in result


# ---------------------------------------------------------------------------
# New tools: pcall, learn/diff, clean, blitz, robot, rest, xpresso
# ---------------------------------------------------------------------------

class TestPyatsPcallShowCommand(unittest.TestCase):
    def setUp(self):
        srv._OP_LOG.clear()

    def test_runs_across_all_devices(self):
        canned = [
            {"status": "completed", "device": "r1", "command": "show version", "output": "ok"},
            {"status": "completed", "device": "r2", "command": "show version", "output": "ok"},
        ]
        with patch.object(srv, "_pcall_show_command", return_value=canned):
            result = _json(_run(srv.pyats_pcall_show_command(["r1", "r2"], "show version")))
        assert result["status"] == "completed"
        assert result["concurrency"] == "pcall (process per device)"
        assert result["summary"] == {"total": 2, "success": 2, "failed": 0}

    def test_empty_device_list_returns_error(self):
        result = _json(_run(srv.pyats_pcall_show_command([], "show version")))
        assert result["status"] == "error"

    def test_invalid_command_returns_error(self):
        result = _json(_run(srv.pyats_pcall_show_command(["r1"], "reload")))
        assert result["status"] == "error"

    def test_partial_failure_reflected_in_summary(self):
        canned = [
            {"status": "completed", "device": "r1", "command": "show version", "output": "ok"},
            {"status": "error", "device": "r2", "error": "timeout"},
        ]
        with patch.object(srv, "_pcall_show_command", return_value=canned):
            result = _json(_run(srv.pyats_pcall_show_command(["r1", "r2"], "show version")))
        assert result["summary"] == {"total": 2, "success": 1, "failed": 1}


class TestPyatsPcallConfigureDevices(unittest.TestCase):
    def setUp(self):
        srv._OP_LOG.clear()

    def test_configures_all_devices(self):
        canned = [
            {"status": "success", "device": "r1", "commands_applied": ["ntp server 1.1.1.1"]},
            {"status": "success", "device": "r2", "commands_applied": ["ntp server 1.1.1.1"]},
        ]
        with patch.object(srv, "_pcall_configure_devices", return_value=canned):
            result = _json(_run(
                srv.pyats_pcall_configure_devices(["r1", "r2"], ["ntp server 1.1.1.1"])
            ))
        assert result["concurrency"] == "pcall (process per device)"
        assert result["summary"] == {"total": 2, "success": 2, "failed": 0}

    def test_empty_device_list_returns_error(self):
        result = _json(_run(srv.pyats_pcall_configure_devices([], ["ntp server 1.1.1.1"])))
        assert result["status"] == "error"


class TestPyatsLearnFeature(unittest.TestCase):
    def setUp(self):
        srv._OP_LOG.clear()
        srv._learn_snapshots.clear()

    def test_learn_without_snapshot(self):
        learned = {"status": "completed", "device": "r1", "feature": "interface",
                   "learned": {"Gi0/0": {"oper_status": "up"}}}
        with patch.object(srv, "_execute_learn_feature", return_value=learned):
            result = _json(_run(srv.pyats_learn_feature("r1", "interface")))
        assert result["status"] == "completed"
        assert "snapshot_saved" not in result
        assert srv._learn_snapshots == {}

    def test_learn_with_snapshot_label_saves_it(self):
        learned = {"status": "completed", "device": "r1", "feature": "interface",
                   "learned": {"Gi0/0": {"oper_status": "up"}}}
        with patch.object(srv, "_execute_learn_feature", return_value=learned):
            result = _json(_run(srv.pyats_learn_feature("r1", "interface", "baseline")))
        assert result["snapshot_saved"] == "baseline"
        assert srv._learn_snapshots["r1:interface:baseline"] == learned["learned"]

    def test_error_result_not_saved_as_snapshot(self):
        errored = {"status": "error", "device": "r1", "feature": "interface", "error": "conn fail"}
        with patch.object(srv, "_execute_learn_feature", return_value=errored):
            result = _json(_run(srv.pyats_learn_feature("r1", "interface", "baseline")))
        assert result["status"] == "error"
        assert srv._learn_snapshots == {}


class TestPyatsDiffLearnedSnapshots(unittest.TestCase):
    def setUp(self):
        srv._OP_LOG.clear()
        srv._learn_snapshots.clear()

    def test_missing_snapshot_returns_error(self):
        result = _json(_run(
            srv.pyats_diff_learned_snapshots("r1", "interface", "before", "after")
        ))
        assert result["status"] == "error"
        assert "before" in result["error"] and "after" in result["error"]

    def test_diff_of_two_snapshots(self):
        srv._learn_snapshots["r1:interface:before"] = {"Gi0/0": {"status": "up"}}
        srv._learn_snapshots["r1:interface:after"] = {"Gi0/0": {"status": "down"}}
        with patch.object(srv, "Diff") as mock_diff_cls:
            mock_diff = MagicMock()
            mock_diff.__str__.return_value = "- status: up\n+ status: down"
            mock_diff_cls.return_value = mock_diff
            result = _json(_run(
                srv.pyats_diff_learned_snapshots("r1", "interface", "before", "after")
            ))
        assert result["status"] == "completed"
        assert "down" in result["diff"]
        mock_diff.findDiff.assert_called_once()


class TestBlitzGuardrails(unittest.TestCase):
    def test_allows_safe_actions(self):
        assert srv._blitz_guardrails("- step1:\n  - execute:\n      command: show version\n") is None

    def test_blocks_reload(self):
        assert srv._blitz_guardrails("- step1:\n  - execute:\n      command: reload\n") is not None

    def test_blocks_write_erase(self):
        assert srv._blitz_guardrails("command: write erase") is not None


class TestPyatsRunBlitz(unittest.TestCase):
    def setUp(self):
        srv._OP_LOG.clear()

    def test_empty_actions_yaml_returns_error(self):
        result = _json(_run(srv.pyats_run_blitz("", ["r1"])))
        assert result["status"] == "error"

    def test_empty_device_list_returns_error(self):
        result = _json(_run(srv.pyats_run_blitz("- step1:\n  - execute:\n      command: show version\n", [])))
        assert result["status"] == "error"

    def test_dangerous_action_blocked(self):
        result = _json(_run(srv.pyats_run_blitz("command: reload", ["r1"])))
        assert result["status"] == "error"

    def test_success_path(self):
        mock_result = {"status": "completed", "returncode": 0, "overall_result": "PASSED",
                       "stdout": "", "stderr": "", "report": None,
                       "trigger_datafile": "/tmp/t.yaml", "archive": None,
                       "artifacts_dir": "/tmp/run"}
        with patch.object(srv, "_run_blitz", return_value=mock_result):
            result = _json(_run(
                srv.pyats_run_blitz("- step1:\n  - execute:\n      command: show version\n", ["r1"])
            ))
        assert result["overall_result"] == "PASSED"


class TestRobotGuardrails(unittest.TestCase):
    def test_allows_safe_script(self):
        assert srv._robot_guardrails('Connect To Device "R1"') is None

    def test_blocks_reload(self):
        assert srv._robot_guardrails("Execute Command    reload") is not None


class TestPyatsRunRobot(unittest.TestCase):
    def setUp(self):
        srv._OP_LOG.clear()

    def test_empty_script_returns_error(self):
        result = _json(_run(srv.pyats_run_robot("")))
        assert result["status"] == "error"

    def test_dangerous_script_blocked(self):
        result = _json(_run(srv.pyats_run_robot("Execute Command    erase startup-config")))
        assert result["status"] == "error"

    def test_success_path(self):
        mock_result = {"status": "completed", "overall_result": "PASSED", "returncode": 0,
                       "stdout": "1 test, 1 passed, 0 failed", "stderr": "",
                       "artifacts_dir": "/tmp/run", "paths": {}}
        with patch.object(srv, "_run_robot_script", return_value=mock_result):
            result = _json(_run(srv.pyats_run_robot("*** Test Cases ***\nT1\n    No Operation\n")))
        assert result["overall_result"] == "PASSED"


class TestPyatsCleanDevice(unittest.TestCase):
    def setUp(self):
        srv._OP_LOG.clear()

    def test_empty_commands_returns_error(self):
        result = _json(_run(srv.pyats_clean_device("r1", [])))
        assert result["status"] == "error"

    def test_dangerous_command_blocked(self):
        result = _json(_run(srv.pyats_clean_device("r1", ["reload"])))
        assert result["status"] == "error"

    def test_dry_run_returns_yaml_without_subprocess(self):
        with patch.object(srv, "_execute_clean_device") as mock_exec:
            result = _json(_run(srv.pyats_clean_device("r1", ["show version"])))
        mock_exec.assert_not_called()
        assert result["status"] == "completed"
        assert result["dry_run"] is True
        assert "execute_command" in result["clean_yaml"]

    def test_real_run_requires_confirm_phrase(self):
        result = _json(_run(
            srv.pyats_clean_device("r1", ["show version"], dry_run=False, confirm="nope")
        ))
        assert result["status"] == "error"

    def test_real_run_with_correct_confirm(self):
        mock_result = {"status": "completed", "device": "r1", "returncode": 0,
                       "stdout": "", "stderr": "", "clean_yaml": "...", "artifacts_dir": "/tmp/run"}
        with patch.object(srv, "_execute_clean_device", return_value=mock_result) as mock_exec:
            result = _json(_run(srv.pyats_clean_device(
                "r1", ["show version"], dry_run=False, confirm=srv._CLEAN_CONFIRM_PHRASE,
            )))
        mock_exec.assert_called_once()
        assert result["status"] == "completed"


class TestPyatsRestRequest(unittest.TestCase):
    def setUp(self):
        srv._OP_LOG.clear()

    def test_no_rest_block_returns_clean_error(self):
        dev = _mock_device("r1")  # connections has no "rest" key
        tb = MagicMock()
        tb.devices = {"r1": dev}
        with patch.object(srv, "_load_testbed", return_value=tb):
            result = _json(_run(srv.pyats_rest_request("r1", "GET", "/restconf/data/x")))
        assert result["status"] == "error"
        assert "rest" in result["error"].lower()

    def test_unsupported_method_returns_error(self):
        dev = _mock_device("r1")
        dev.connections = {"rest": MagicMock()}
        tb = MagicMock()
        tb.devices = {"r1": dev}
        with patch.object(srv, "_load_testbed", return_value=tb):
            result = _json(_run(srv.pyats_rest_request("r1", "TRACE", "/restconf/data/x")))
        assert result["status"] == "error"

    def test_get_success(self):
        dev = _mock_device("r1")
        dev.connections = {"rest": MagicMock()}
        dev.rest = MagicMock()
        dev.rest.connected = True
        resp = MagicMock()
        resp.status_code = 200
        resp.text = '{"ietf-interfaces:interfaces": {}}'
        dev.rest.get.return_value = resp
        tb = MagicMock()
        tb.devices = {"r1": dev}
        with patch.object(srv, "_load_testbed", return_value=tb):
            result = _json(_run(
                srv.pyats_rest_request("r1", "GET", "/restconf/data/ietf-interfaces:interfaces")
            ))
        assert result["status"] == "completed"
        assert result["status_code"] == 200
        assert result["body"] == {"ietf-interfaces:interfaces": {}}


class TestPyatsXpressoRequest(unittest.TestCase):
    def setUp(self):
        srv._OP_LOG.clear()
        self._url, self._token, self._group = srv.XPRESSO_URL, srv.XPRESSO_API_TOKEN, srv.XPRESSO_GROUP

    def tearDown(self):
        srv.XPRESSO_URL, srv.XPRESSO_API_TOKEN, srv.XPRESSO_GROUP = self._url, self._token, self._group

    def test_not_configured_returns_clear_error(self):
        srv.XPRESSO_URL, srv.XPRESSO_API_TOKEN, srv.XPRESSO_GROUP = "", "", ""
        result = _json(_run(srv.pyats_xpresso_request("GET", "/api/v2/testbeds")))
        assert result["status"] == "error"
        assert "XPRESSO_URL" in result["error"]

    def test_empty_path_returns_error(self):
        result = _json(_run(srv.pyats_xpresso_request("GET", "")))
        assert result["status"] == "error"

    def test_get_success(self):
        srv.XPRESSO_URL, srv.XPRESSO_API_TOKEN, srv.XPRESSO_GROUP = (
            "https://xpresso.example.com", "tok123", "mygroup",
        )
        resp = MagicMock()
        resp.status_code = 200
        resp.json.return_value = {"items": []}
        with patch.object(srv.requests, "request", return_value=resp) as mock_req:
            result = _json(_run(srv.pyats_xpresso_request("GET", "/api/v2/testbeds")))
        assert result["status"] == "completed"
        assert result["status_code"] == 200
        called_headers = mock_req.call_args.kwargs["headers"]
        assert called_headers["Authorization"] == "Jwt tok123"
        assert called_headers["Group"] == "mygroup"


# ---------------------------------------------------------------------------
# Run directly
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    unittest.main(verbosity=2)
