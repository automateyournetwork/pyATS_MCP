# pyATS MCP Server
[![Trust Score](https://archestra.ai/mcp-catalog/api/badge/quality/automateyournetwork/pyATS_MCP)](https://archestra.ai/mcp-catalog/automateyournetwork__pyats_mcp)

[![Available on CodeGuilds](https://img.shields.io/badge/Available_on-CodeGuilds-6366f1)](https://codeguilds.dev/packages/pyats-mcp)

Cisco pyATS and Genie already know how to talk to a network — parsing show commands, pushing configuration, learning feature state, running declarative tests. What they didn't have was a way for an AI agent to drive any of it directly. This server closes that gap: it wraps pyATS/Genie as a set of structured, guarded MCP tools that an agent like Claude can call against a real testbed, over the Model Context Protocol's current Streamable HTTP transport.

Point an agent at it and it can look up a device, run and parse a show command, apply configuration with a rollback point, learn and diff a feature's state before and after a change, fan a command out across a fleet — one thread pool or one process per device — run a declarative Blitz or Robot Framework test, or call a device's REST/RESTCONF API directly. Every risky path is guarded before it reaches a device, and every call lands in an in-memory audit log the agent can review mid-session.

---

## At a glance

- **Transport** — Streamable HTTP (`mcp>=2.0.0`), stateful or stateless, chosen with one environment variable. STDIO is gone.
- **26 tools** across discovery, show commands, configuration, Genie learn/diff, Genie Clean, declarative testing (Blitz, Robot Framework, AEtest), generic REST/RESTCONF, and Cisco XPresso.
- **Two ways to fan out** a command across many devices — a shared thread pool for everyday use, or one OS process per device (`pyats.async_.pcall`) when you want real isolation at scale.
- **Guardrails, not honor systems** — dangerous commands are blocked before they reach a device, Genie Clean can never run a stage that reboots or reimages one, and destructive actions require an exact confirmation phrase.
- **Nothing hard-coded** — every credential and device detail lives in `.env`, pulled into `testbed.yaml` at runtime via `%ENV{}` substitution.

---

## Prerequisites

- Python 3.10+
- A pyATS `testbed.yaml` pointed at real or virtual network devices — a physical lab, Cisco Modeling Labs / VIRL / GNS3, or anything else Unicon can reach over SSH/Telnet. pyATS MCP doesn't simulate a network; it drives one.
- An MCP-capable client to talk to it — see [Connect Your Agent](#connect-your-agent) below.

---

## Quick Start

```bash
# 1. Clone and install
git clone https://github.com/automateyournetwork/pyATS_MCP
cd pyATS_MCP
pip install -r requirements.txt

# 2. Configure your environment
cp .env.example .env
# Edit .env — see Configuration below

# 3. Run — starts a Streamable HTTP server on 0.0.0.0:8080 by default
python3 pyats_mcp_server.py
```

The MCP endpoint is then reachable at `http://<host>:<port>/mcp`.

---

## Configuration

All device details and credentials live in a `.env` file — nothing is hard-coded in the repo.

### 1. Copy the template

```bash
cp .env.example .env
```

### 2. Set the server variables

```dotenv
PYATS_TESTBED_PATH=/absolute/path/to/your/testbed.yaml
PYATS_MCP_ARTIFACTS_DIR=          # default: ~/.pyats-mcp/artifacts
PYATS_MCP_KEEP_ARTIFACTS=1        # 1 = keep, 0 = delete after each run
PYATS_MCP_TESTBED_CACHE_TTL=30    # seconds before testbed reloads from disk
PYATS_MCP_CONN_CACHE_TTL=0        # seconds to keep connections alive (0 = off)
PYATS_MCP_OP_LOG_MAX=500          # max entries in the in-memory operation log

# Transport (Streamable HTTP only — STDIO is not supported)
PYATS_MCP_TRANSPORT_MODE=stateful # stateful (default) | stateless
PYATS_MCP_HTTP_HOST=0.0.0.0
PYATS_MCP_HTTP_PORT=8080

# Optional — only needed for pyats_xpresso_request
XPRESSO_URL=
XPRESSO_API_TOKEN=
XPRESSO_GROUP=
```

`PYATS_MCP_TRANSPORT_MODE=stateless` sets `stateless_http=True` on the Streamable HTTP transport, so no server-side session state is retained between requests from clients still negotiating the older, handshake-based protocol. Clients speaking the current MCP protocol (2026-07-28, SEP-2575) are handshake-free by default regardless of this setting — that comes from the `mcp>=2.0.0` SDK itself, not anything configured here.

### 3. Add a block for each device

Every device in your `testbed.yaml` uses `%ENV{VAR}` substitution, so credentials and connection details are read from `.env` at runtime.

Use the `{DEVICENAME}_{FIELD}` naming convention:

```dotenv
# Supported os values: iosxe | iosxr | nxos | ios | eos | junos | panos | linux | windows
# Set os=generic and platform="" to let Unicon autodetect on first connect.

CORE1_IP=10.1.1.1
CORE1_PORT=22
CORE1_OS=iosxe
CORE1_PLATFORM=cat9k
CORE1_USERNAME=admin
CORE1_PASSWORD=s3cr3t
CORE1_ENABLE_PASSWORD=s3cr3t

FW1_IP=10.1.1.2
FW1_PORT=22
FW1_OS=panos
FW1_PLATFORM=
FW1_USERNAME=admin
FW1_PASSWORD=s3cr3t
# (no enable password for Palo Alto)

LINUX1_IP=10.1.1.3
LINUX1_PORT=22
LINUX1_OS=linux
LINUX1_PLATFORM=ubuntu
LINUX1_USERNAME=admin
LINUX1_PASSWORD=s3cr3t
# (no enable password for Linux)
```

If a group of devices shares credentials, define group-level vars and reference them across devices:

```dotenv
SITE_A_USERNAME=netops
SITE_A_PASSWORD=s3cr3t
SITE_A_ENABLE_PASSWORD=s3cr3t
```

### 4. Reference the variables in testbed.yaml

```yaml
devices:
  CORE1:
    alias: "Core Switch 1"
    type: "switch"
    os: "%ENV{CORE1_OS}"
    platform: "%ENV{CORE1_PLATFORM}"
    credentials:
      default:
        username: "%ENV{CORE1_USERNAME}"
        password: "%ENV{CORE1_PASSWORD}"
      enable:
        password: "%ENV{CORE1_ENABLE_PASSWORD}"
    connections:
      cli:
        protocol: ssh
        ip: "%ENV{CORE1_IP}"
        port: "%ENV{CORE1_PORT}"
        arguments:
          connection_timeout: 360
```

> For devices with unknown OS, set `os: "%ENV{DEVICE_OS}"` with `DEVICE_OS=generic` in `.env`
> and optionally add `learn_os: true` under `arguments:` — Unicon will detect and cache the OS
> after the first connection.

---

## Docker

### Build

```bash
docker build -t pyats-mcp-server .
```

### Run (pass .env directly)

```bash
docker run -p 8080:8080 --rm \
  --env-file /absolute/path/to/.env \
  -v /absolute/path/to/testbed.yaml:/app/testbed.yaml \
  pyats-mcp-server
```

Either way, the server is a long-running process you start once and point clients at — it isn't something an agent spawns per session. See below for exactly how each client connects to it.

---

## Connect Your Agent

The server exposes one thing: an MCP endpoint at `http://<host>:<port>/mcp` (Streamable HTTP). Every client below just needs that URL — no `command`/`args`, no local process for the client to manage.

### Claude Code

```bash
claude mcp add --transport http pyats http://localhost:8080/mcp

# Behind auth (e.g. a reverse proxy in front of the server)
claude mcp add --transport http pyats http://localhost:8080/mcp \
  --header "Authorization: Bearer your-token"
```

Or drop it straight into `.mcp.json` (project-scoped, committed to the repo) or `~/.claude.json` (user-scoped):

```json
{
  "mcpServers": {
    "pyats": { "type": "http", "url": "http://localhost:8080/mcp" }
  }
}
```

### VS Code (GitHub Copilot Chat)

Add a `.vscode/mcp.json` in the workspace (or run **MCP: Add Server** from the Command Palette):

```json
{
  "servers": {
    "pyats": { "type": "http", "url": "http://localhost:8080/mcp" }
  }
}
```

### OpenAI Codex CLI

```bash
codex mcp add pyats --url http://localhost:8080/mcp
```

Or in `~/.codex/config.toml`:

```toml
[mcp_servers.pyats]
url = "http://localhost:8080/mcp"
```

### Claude Desktop

Claude Desktop's `claude_desktop_config.json` is stdio-only — putting a `url` field in it doesn't work (it's a known issue, not a supported path). Remote/HTTP servers are added instead as a **Custom Connector** under Settings → Connectors, and Desktop connects to it from Anthropic's cloud, not your local machine — so it needs a real, publicly-reachable HTTPS URL, not `localhost`.

To point Desktop at a server running on your own machine anyway, bridge it through [`mcp-remote`](https://www.npmjs.com/package/mcp-remote) as a local stdio proxy:

```json
{
  "mcpServers": {
    "pyats": {
      "command": "npx",
      "args": ["-y", "mcp-remote", "http://localhost:8080/mcp", "--transport", "http-only"]
    }
  }
}
```

### Raw Python (LangGraph, custom agents, anything else)

```python
from mcp import ClientSession
from mcp.client.streamable_http import streamablehttp_client

async def main():
    async with streamablehttp_client("http://localhost:8080/mcp") as (read, write, _session_id):
        async with ClientSession(read, write) as session:
            await session.initialize()
            tools = await session.list_tools()
            result = await session.call_tool(
                "pyats_run_show_command",
                arguments={"device_name": "CORE1", "command": "show version"},
            )
```

---

## What To Ask It

Once connected, talk to it like you'd talk to someone who already knows the network:

- *"What devices are in the testbed?"* → `pyats_list_devices`
- *"Show me the BGP summary on CORE1"* → `pyats_run_show_command`, parsed into structured JSON
- *"Snapshot CORE1's OSPF state, then apply this config and show me what changed"* → `pyats_learn_feature` (before) → `pyats_configure_with_diff` → `pyats_learn_feature` (after) → `pyats_diff_learned_snapshots`
- *"Run `show ip interface brief` across every switch"* → `pyats_run_show_command_multi` (or `pyats_pcall_show_command` for process-per-device isolation at real scale)
- *"If that config change breaks anything, roll it back"* → `pyats_rollback_config`
- *"Run this Blitz test against R1 and R2"* / *"Run this Robot Framework suite"* → `pyats_run_blitz` / `pyats_run_robot`

The agent chains these itself — you describe the outcome, it picks the tools.

---

## Available Tools

26 tools, grouped by what they do.

**Discovery**

| Tool | Description |
|------|-------------|
| `pyats_list_devices` | List all devices in the testbed |
| `pyats_search_devices` | Fuzzy-search devices by name or alias |

**Show commands**

| Tool | Description |
|------|-------------|
| `pyats_run_show_command` | Run a validated show command; returns parsed JSON or raw output |
| `pyats_run_show_command_multi` | Run a show command across multiple devices concurrently (thread pool) |
| `pyats_pcall_show_command` | Same, but one OS process per device (`pyats.async_.pcall`) instead of a shared thread pool |
| `pyats_show_running_config` | Retrieve the full running configuration (raw text) |
| `pyats_show_logging` | Retrieve device system logs via `show logging` |
| `pyats_ping_from_network_device` | Execute a ping from a network device |
| `pyats_run_linux_command` | Run a command on a Linux host |

**Configuration**

| Tool | Description |
|------|-------------|
| `pyats_configure_device` | Apply configuration commands with safety guardrails |
| `pyats_configure_devices_multi` | Apply configuration across multiple devices concurrently (thread pool) |
| `pyats_pcall_configure_devices` | Same, but one OS process per device |
| `pyats_configure_with_diff` | Apply config and return a before/after diff |
| `pyats_rollback_config` | Roll back to the last saved configuration snapshot |

**State & diagnostics**

| Tool | Description |
|------|-------------|
| `pyats_device_health` | Snapshot CPU, memory, interfaces, and routing state |
| `pyats_get_neighbors` | Retrieve CDP/LLDP neighbors |
| `pyats_find_interface_by_ip` | Find which interface owns a given IP address |
| `pyats_learn_feature` | Genie `device.learn()` for a whole feature (interface, ospf, bgp, …), optionally saved as a named snapshot |
| `pyats_diff_learned_snapshots` | Diff two snapshots saved by `pyats_learn_feature` |

**Testing & automation**

| Tool | Description |
|------|-------------|
| `pyats_clean_device` | Genie Clean (Kleenex), restricted to non-destructive `connect`+`execute_command` stages; `dry_run=True` by default |
| `pyats_run_blitz` | Run a declarative pyATS Blitz YAML test |
| `pyats_run_robot` | Run a Robot Framework suite using the `pyats.robot`/`genie.libs.robot` keyword libraries |
| `pyats_run_dynamic_test` | Execute a sandboxed pyATS AEtest script |

**APIs**

| Tool | Description |
|------|-------------|
| `pyats_rest_request` | Generic REST/RESTCONF/NX-API call via pyATS's `rest.connector` (a separate connection type from CLI/SSH) |
| `pyats_xpresso_request` | Authenticated call to Cisco XPresso's REST API v2 (test requests, jobs, testbeds, images, …) |

**Session**

| Tool | Description |
|------|-------------|
| `pyats_get_operation_log` | Retrieve the in-memory operation log |

---

## Security

- Show commands are validated — pipes, redirects, and dangerous keywords are blocked.
- Config changes are checked for `reload`, `erase`, `write erase`, `delete`, `format` — the same check runs inside `pyats_clean_device`, `pyats_run_blitz`, and `pyats_run_robot`.
- Dynamic test scripts run in a restricted sandbox (banned imports: `os`, `sys`, `subprocess`, etc.).
- `pyats_clean_device` never runs a real Genie Clean stage that reboots, erases, or reimages a device — only `connect`+`execute_command` are ever generated — and defaults to `dry_run=True`; running for real also requires an exact confirmation phrase.
- Every process-global cache (connection cache, testbed cache, config/learn snapshots, operation log) is protected by a lock, so concurrent HTTP clients can't corrupt shared state.
- All credentials come from `.env` — never stored in the testbed file or source code.

---

## Project Structure

```
.
├── pyats_mcp_server.py      # MCP server
├── test_pyats_mcp_server.py # Unit tests (119 tests)
├── benchmark/               # Pre/post, stateful/stateless transport benchmark
├── Dockerfile               # Container definition
├── requirements.txt         # Pinned runtime dependencies
├── requirements-dev.txt     # Dev/test dependencies
├── pyproject.toml           # Tool config (black, isort, pytest, mypy)
├── .env.example             # Configuration template — copy to .env
├── .gitignore
├── LICENSE
└── CONTRIBUTING.md
```

---

## Development

```bash
# Install dev dependencies with uv
uv venv .venv && uv pip install -r requirements-dev.txt

# Run tests
.venv/bin/python -m pytest

# Lint and format
.venv/bin/black .
.venv/bin/isort .
.venv/bin/flake8 . --max-line-length=100
```

See [CONTRIBUTING.md](CONTRIBUTING.md) for the full setup and PR workflow.

---

## Benchmark

`benchmark/` compares STDIO (legacy) against Streamable HTTP in both stateful and stateless mode, against a real testbed. See `benchmark/scenarios.py` for the scenario list and `benchmark/aggregate.py` for building the comparison report; `benchmark/results/summary.md` has the most recent run's numbers.

---

## License

[MIT](LICENSE)
