# pyATS MCP Server
[![Trust Score](https://archestra.ai/mcp-catalog/api/badge/quality/automateyournetwork/pyATS_MCP)](https://archestra.ai/mcp-catalog/automateyournetwork__pyats_mcp)

[![Available on CodeGuilds](https://img.shields.io/badge/Available_on-CodeGuilds-6366f1)](https://codeguilds.dev/packages/pyats-mcp)

Cisco pyATS and Genie already know how to talk to a network — parsing show commands, pushing configuration, learning feature state, running declarative tests. What they didn't have was a way for an AI agent to drive any of it directly. This server closes that gap: it wraps pyATS/Genie as a set of structured, guarded MCP tools that an agent like Claude can call against a real testbed, over the Model Context Protocol's current Streamable HTTP transport.

Point an agent at it and it can look up a device, run and parse a show command, apply configuration with a rollback point, learn and diff a feature's state before and after a change, fan a command out across a fleet — one thread pool or one process per device — run a declarative Blitz or Robot Framework test, or call a device's REST/RESTCONF API directly. Every risky path is guarded before it reaches a device, and every call lands in an in-memory audit log the agent can review mid-session.

---

> [!TIP]
> ### Testbed-free: build testbeds dynamically, in memory
> You don't need a `testbed.yaml`. The `pyats_testbed_*` tools build a pyATS testbed **on the fly** from Itential Inventory Manager (the default source), NetBox, Nautobot, Catalyst Center, any JSON REST API, or inline/file CSV, JSON, YAML and XML, using a declarative field mapping. The testbed lives in server memory, is activated with one call, and every existing tool (`pyats_run_show_command`, `pyats_configure_device`, ...) uses it immediately. Start the server with `PYATS_MCP_DYNAMIC_ONLY=1` and no YAML at all. Credentials stay on the server as `%ENV{}` references and never pass through the model. See [Dynamic Testbeds](#dynamic-testbeds) and the [technical write-up](docs/dynamic-testbeds.md).

---

## At a glance

- **Transport** — Streamable HTTP (`mcp>=2.3.0`), stateful or stateless, chosen with one environment variable. STDIO is gone.
- **34 tools** across discovery, show commands, configuration, Genie learn/diff, Genie Clean, declarative testing (Blitz, Robot Framework, AEtest), generic REST/RESTCONF, and Cisco XPresso.
- **Two ways to fan out** a command across many devices — a shared thread pool for everyday use, or one OS process per device (`pyats.async_.pcall`) when you want real isolation at scale.
- **Guardrails, not honor systems** — dangerous commands are blocked before they reach a device, Genie Clean can never run a stage that reboots or reimages one, and destructive actions require an exact confirmation phrase.
- **Nothing hard-coded** — every credential and device detail lives in `.env`, pulled into `testbed.yaml` at runtime via `%ENV{}` substitution.

---

## Prerequisites

- Python 3.10+
- A pyATS `testbed.yaml` pointed at real or virtual network devices — a physical lab, Cisco Modeling Labs / VIRL / GNS3, or anything else Unicon can reach over SSH/Telnet. pyATS MCP doesn't simulate a network; it drives one. A static `testbed.yaml` is optional: see [Dynamic Testbeds](#dynamic-testbeds).
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

### With [uv](https://docs.astral.sh/uv/)

```bash
git clone https://github.com/automateyournetwork/pyATS_MCP
cd pyATS_MCP
cp .env.example .env   # edit — see Configuration below

uv sync                              # creates .venv and installs dependencies
uv run python pyats_mcp_server.py    # serves http://<host>:<port>/mcp
```

Requires Python 3.10+ (3.12 or older is the safest fit for `pyats[full]`). For development, add the dev extras with `uv sync --extra dev` and run `uv run pytest`.

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

`PYATS_MCP_TRANSPORT_MODE=stateless` sets `stateless_http=True` on the Streamable HTTP transport, so no server-side session state is retained between requests from clients still negotiating the older, handshake-based protocol. Clients speaking the current MCP protocol (2026-07-28, SEP-2575) are handshake-free by default regardless of this setting — that comes from the `mcp>=2.3.0` SDK itself, not anything configured here.

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

## Asynchronous MCP Tasks

MCP supports returning a task handle immediately and retrieving the result later.
This server implements the **2026-07-28 `io.modelcontextprotocol/tasks` extension**
using the Python SDK's extension hooks (`mcp>=2.3.0`). It does not implement the
older experimental 2025-11-25 `tasks/result` / `tasks/list` protocol.
See the [Tasks specification](https://tasks.extensions.modelcontextprotocol.io/specification/2026-07-28/tasks)
and [Python SDK extension API](https://py.sdk.modelcontextprotocol.io/advanced/extensions/).

All 22 device I/O, remote request and test execution tools use tasks when the client
advertises this extension on the request. Device inventory/search, snapshot diffs,
and operation history remain immediate. Existing clients that do not advertise
Tasks keep the ordinary tool result behavior. Client support must match this
extension; merely supporting the older experimental Tasks API is insufficient.

The initial `tools/call` request includes this metadata (alongside `name` and `arguments`):

```json
{
  "_meta": {
    "io.modelcontextprotocol/protocolVersion": "2026-07-28",
    "io.modelcontextprotocol/clientCapabilities": {
      "extensions": {"io.modelcontextprotocol/tasks": {}}
    }
  }
}
```

The response has `resultType: "task"`, `taskId`, `status: "working"`, timestamps,
and `pollIntervalMs`. Poll `tasks/get` with `{"taskId": "..."}` and the same metadata.
Its response has `resultType: "complete"`; inspect **status** to determine whether
the work has finished. A completed task contains the original MCP tool response in
`result`. A protocol failure uses `status: "failed"` and `error`; a tool error still
uses `status: "completed"` and preserves the tool's error payload. HTTP requests use
`Mcp-Method` and `Mcp-Name` headers; for task methods `Mcp-Name` is the task ID.

Run the [example polling client](examples/task_client.py):

```bash
python examples/task_client.py http://localhost:8080/mcp pyats_run_show_command \
  '{"device_name":"router-1","command":"show version"}'
```

`tasks/cancel` acknowledges cancellation intent. Queued work is cancelled before
execution. Once a tool starts, it finishes and retains its result: cancelling an
await cannot safely stop an SSH command, reverse a configuration change, or kill
a test subprocess. `tasks/update` acknowledges and ignores input response keys;
these tools do not request interactive input. Notifications are not implemented;
clients use polling and honor `pollIntervalMs` (1 second).

| Variable | Default | Meaning |
|---|---|---|
| `PYATS_MCP_TASK_DB` | `<artifacts directory>/tasks.sqlite3` | Persistent task/result store |
| `PYATS_MCP_TASK_WORKERS` | `4` | Maximum simultaneous long-running tool calls |
| `PYATS_MCP_TASK_MAX` | `1000` | Maximum retained tasks, including queued/running tasks |
| `PYATS_MCP_TASK_RETENTION` | `86400` | Seconds to retain terminal results |

The worker limit applies to ordinary calls too. Multi-device tools retain their
internal fan-out. Calls targeting the same device are serialized for the full tool
operation, including config snapshots and diffs; independent devices can execute
concurrently. Dynamic Python/Robot tests reserve all testbed devices. Scripts that
connect to devices outside the supplied testbed remain responsible for their own
concurrency. Connection TTL eviction occurs when that device is next used.

Tasks survive HTTP disconnects in both stateful and stateless modes. SQLite commits
the handle before returning it and preserves terminal results across restarts.
Running tasks have `ttlMs: null`; completion sets a TTL measured from creation that
includes the retention period. Expired results are removed on subsequent task
submission. Admission is rejected when the retained-task limit is reached.
On restart, unfinished tasks become failed with an unknown-outcome message; they
are **never automatically replayed**. Inspect device state before retrying changes.
Graceful shutdown cancels queued tasks and drains active operations.

Run **one server process per task database**; a file lock enforces this. For Docker,
mount a persistent directory and point `PYATS_MCP_TASK_DB` into it. Horizontal
replicas require separate stores and routing each task ID back to its owning
replica. This implementation is not a distributed task queue.

The database contains tool results, potentially including configurations, and is
created with owner-only permissions. Authenticated tasks are bound to the verified
client/issuer/subject identity. With this project's default unauthenticated server,
the unguessable task ID acts as a bearer credential; protect it and the database.

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

34 tools, grouped by what they do.

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

**Dynamic testbeds** — see [Dynamic Testbeds](#dynamic-testbeds)

| Tool | Description |
|------|-------------|
| `pyats_testbed_sources` | Which sources are configured on the server, allow-lists, active testbed |
| `pyats_testbed_source_inventories` | List Itential Inventory Manager inventories with node counts |
| `pyats_testbed_build` | Build an in-memory testbed from Itential, NetBox, Nautobot, Catalyst Center, csv/json/yaml/xml or any REST API |
| `pyats_testbed_validate` | pyATS schema check, credential env check, optional TCP reachability |
| `pyats_testbed_activate` | Switch every other tool to a dynamic testbed (or back to `static`) |
| `pyats_testbed_list` / `pyats_testbed_get` / `pyats_testbed_delete` | Inspect and manage in-memory testbeds |

**Session**

| Tool | Description |
|------|-------------|
| `pyats_get_operation_log` | Retrieve the in-memory operation log |

---

## Dynamic Testbeds

Build a testbed **in memory, at runtime, from whatever your source of truth is** — no
`testbed.yaml` round-trip. A testbed is a plain pyATS testbed dict handed straight to
`pyats.topology.loader.load()`; a file is only materialized (mode `0600`) when a
subprocess tool such as `pyats run job` or `pyats clean` needs a path.

```
pyats_testbed_sources              what can I build from, and what is active?
pyats_testbed_source_inventories   list Itential Inventory Manager inventories (+ node counts)
pyats_testbed_build                fetch -> map -> filter -> build (dry_run / activate / merge_with)
pyats_testbed_validate             pyATS schema load + credential env check (+ TCP reachability)
pyats_testbed_activate             make every other tool use it ("static" switches back)
pyats_testbed_list / _get / _delete
```

| `source=` | Reads from | Server config |
|-----------|-----------|---------------|
| `itential_inventory` (default) | Itential Platform **Inventory Manager** (`/inventory_manager/v1`) | `PLATFORM_URL`, `AUTH_METHOD`, `CLIENT_ID`/`CLIENT_SECRET` (oauth) or `PLATFORM_USERNAME`/`PLATFORM_PASSWORD` (local) |
| `netbox` / `nautobot` | `/api/dcim/devices/` | `NETBOX_URL`+`NETBOX_TOKEN` / `NAUTOBOT_URL`+`NAUTOBOT_TOKEN` |
| `catalyst_center` | `/dna/intent/api/v1/network-device` | `CATALYST_CENTER_URL`, `_USERNAME`, `_PASSWORD` |
| `inline` / `file` | csv, json, yaml or xml text / file | file must live under `PYATS_MCP_TESTBED_SOURCE_DIRS` |
| `rest` | any JSON REST API (offset, page or next-URL pagination) | host must be listed in `PYATS_MCP_TESTBED_SOURCE_HOSTS` |

Run more than one Itential Platform (for example a local devstack and a cloud SE
Platform) with named profiles: `ITENTIAL_DEVSTACK_URL`, `ITENTIAL_DEVSTACK_AUTH_METHOD`,
`ITENTIAL_DEVSTACK_CLIENT_ID`, `ITENTIAL_DEVSTACK_CLIENT_SECRET` (or `_USERNAME`/`_PASSWORD`),
then `pyats_testbed_build(..., profile="devstack")`. Set `ITENTIAL_<PROFILE>_VERIFY_TLS=0`
for self-signed lab certificates.

**Mapping.** Presets map each source's records to devices. For anything else pass a
declarative `mapping` — dotted paths (`a.b[0].c`, `tags[*].slug`), templates
(`{site}-{name}`), `literal:` values or `{"path", "map", "regex", "default"}` objects:

```json
{"name": "build", "source": "inline", "format": "csv",
 "data": "host,addr,nos\nr1,10.0.0.1,ios-xe\nr2,10.0.0.2,nx-os\n",
 "mapping": {"name": "host", "ip": "addr", "os": "nos"}}
```

`os` values are normalized to pyATS names (`IOS-XE`→`iosxe`, `NX-OS`→`nxos`, `frr`→`linux`,
...); use `os_map` for vendor-specific values or `default_os` when the source has no platform
field (several real Itential inventories leave `itential_platform` blank). `filters` and `tags`
narrow large sources, e.g. `tags=["firewall"]` against a 1,340-node inventory.

**Credentials never pass through the model.** Generated devices reference server
environment variables (`%ENV{PYATS_DEVICE_USERNAME}` / `_PASSWORD` / `_ENABLE_PASSWORD` by
default; override with `credentials={"username_env": ..., "password_env": ...}`, or reuse the
static testbed's block with `credentials={"from": "static"}`). Inline secrets are rejected,
source-API tokens come from server env vars, and anything credential-shaped is redacted from
previews. `merge_with="static"` adds the static testbed's devices to a dynamic one.

`pyats_testbed_build` and `pyats_testbed_validate` do network I/O and run as
[Asynchronous MCP Tasks](#asynchronous-mcp-tasks) when the client supports them. Set
`PYATS_MCP_DYNAMIC_ONLY=1` to start the server without any static testbed file.

---

## Security

- Show commands are validated — pipes, redirects, and dangerous keywords are blocked.
- Config changes are checked for `reload`, `erase`, `write erase`, `delete`, `format` — the same check runs inside `pyats_clean_device`, `pyats_run_blitz`, and `pyats_run_robot`.
- Dynamic test scripts run in a restricted sandbox (banned imports: `os`, `sys`, `subprocess`, etc.).
- `pyats_clean_device` never runs a real Genie Clean stage that reboots, erases, or reimages a device — only `connect`+`execute_command` are ever generated — and defaults to `dry_run=True`; running for real also requires an exact confirmation phrase.
- Every process-global cache (connection cache, testbed cache, config/learn snapshots, operation log) is protected by a lock, so concurrent HTTP clients can't corrupt shared state.
- All credentials come from `.env` — never stored in the testbed file or source code.
- Dynamic testbeds hold only `%ENV{}` credential references; the generic `rest` source can only call allow-listed hosts, `file` only allow-listed directories, and pagination never follows a `next` link to another host.

---

## Project Structure

```
.
├── pyats_mcp_server.py      # MCP server
├── pyats_tasks.py           # MCP Tasks extension
├── pyats_testbed_builder.py # Dynamic testbed sources, mapping and registry
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
