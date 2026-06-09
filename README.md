# pyATS MCP Server
[![Trust Score](https://archestra.ai/mcp-catalog/api/badge/quality/automateyournetwork/pyATS_MCP)](https://archestra.ai/mcp-catalog/automateyournetwork__pyats_mcp)

[![Available on CodeGuilds](https://img.shields.io/badge/Available_on-CodeGuilds-6366f1)](https://codeguilds.dev/packages/pyats-mcp)


MCP server that wraps Cisco pyATS and Genie, letting AI agents (Claude, LangGraph, etc.) run show commands, apply configuration, and query network state over STDIO using JSON-RPC 2.0.

> All communication is via STDIN/STDOUT — no HTTP ports, no REST endpoints.

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

# 3. Run
python3 pyats_mcp_server.py
```

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
```

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

If a group of devices shares credentials you can define group-level vars and reference them across devices:

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
docker run -i --rm \
  --env-file /absolute/path/to/.env \
  -v /absolute/path/to/testbed.yaml:/app/testbed.yaml \
  pyats-mcp-server
```

### MCP client config (Docker)

```json
{
  "mcpServers": {
    "pyats": {
      "command": "docker",
      "args": [
        "run", "-i", "--rm",
        "--env-file", "/absolute/path/to/.env",
        "-v", "/absolute/path/to/testbed.yaml:/app/testbed.yaml",
        "pyats-mcp-server"
      ]
    }
  }
}
```

### MCP client config (local Python)

```json
{
  "mcpServers": {
    "pyats": {
      "command": "python3",
      "args": ["-u", "/path/to/pyats_mcp_server.py"],
      "env": {
        "PYATS_TESTBED_PATH": "/absolute/path/to/testbed.yaml"
      }
    }
  }
}
```

---

## Available Tools

| Tool | Description |
|------|-------------|
| `pyats_list_devices` | List all devices in the testbed |
| `pyats_search_devices` | Fuzzy-search devices by name or alias |
| `pyats_run_show_command` | Run a validated show command; returns parsed JSON or raw output |
| `pyats_run_show_command_on_multiple_devices` | Run a show command across multiple devices concurrently |
| `pyats_ping_from_network_device` | Execute a ping from a network device |
| `pyats_run_linux_command` | Run a command on a Linux host |
| `pyats_configure_device` | Apply configuration commands with safety guardrails |
| `pyats_configure_devices_multi` | Apply configuration across multiple devices concurrently |
| `pyats_configure_with_diff` | Apply config and return a before/after diff |
| `pyats_rollback_config` | Roll back to the last saved configuration snapshot |
| `pyats_device_health` | Snapshot CPU, memory, interfaces, and routing state |
| `pyats_get_neighbors` | Retrieve CDP/LLDP neighbors |
| `pyats_find_interface_by_ip` | Find which interface owns a given IP address |
| `pyats_run_dynamic_test` | Execute a sandboxed pyATS test script |
| `pyats_get_operation_log` | Retrieve the in-memory operation log |

---

## Security

- Show commands are validated — pipes, redirects, and dangerous keywords are blocked
- Config changes are checked for `reload`, `erase`, `write erase`, `delete`, `format`
- Dynamic test scripts run in a restricted sandbox (banned imports: `os`, `sys`, `subprocess`, etc.)
- All credentials come from `.env` — never stored in the testbed file or source code

---

## Project Structure

```
.
├── pyats_mcp_server.py      # MCP server
├── test_pyats_mcp_server.py # Unit tests (85 tests)
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
