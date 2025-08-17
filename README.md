# pyATS MCP Server
[![Trust Score](https://archestra.ai/mcp-catalog/api/badge/quality/automateyournetwork/pyATS_MCP)](https://archestra.ai/mcp-catalog/automateyournetwork__pyats_mcp)

This project implements a Model Context Protocol (MCP) Server that wraps Cisco pyATS and Genie functionality. It enables structured, model-driven interaction with network devices over STDIO using the JSON-RPC 2.0 protocol.

🚨 This server does not use HTTP or SSE. All communication is done via STDIN/STDOUT (standard input/output), making it ideal for secure, embedded, containerized, or LangGraph-based tool integrations.

🔧 What It Does

Connects to Cisco IOS/NX-OS devices defined in a pyATS testbed

Supports safe execution of validated CLI commands (show, ping)

Allows controlled configuration changes

Returns structured (parsed) or raw output

Exposes a set of well-defined tools via tools/discover and tools/call

Operates entirely via STDIO for minimal surface area and maximum portability

🚀 Usage

1. Set your testbed path

```bash

export PYATS_TESTBED_PATH=/absolute/path/to/testbed.yaml

```

2. Run the server

Continuous STDIO Mode (default)

```bash

python3 pyats_mcp_server.py

```

Launches a long-running process that reads JSON-RPC requests from stdin and writes responses to stdout.

One-Shot Mode

``` bash

echo '{"jsonrpc": "2.0", "id": 1, "method": "tools/discover"}' | python3 pyats_mcp_server.py --oneshot

```
Processes a single JSON-RPC request and exits.

📦 Docker Support

Build the container

```bash

docker build -t pyats-mcp-server .

```

Run the container (STDIO Mode)
```bash
docker run -i --rm \
  -e PYATS_TESTBED_PATH=/app/testbed.yaml \
  -v /your/testbed/folder:/app \
  pyats-mcp-server
```

🧠 Available MCP Tools

Tool	Description

run_show_command	Executes show commands safely with optional parsing

run_ping_command	Executes ping tests and returns parsed or raw results

apply_configuration	Applies safe configuration commands (multi-line supported)

learn_config	Fetches running config (show run brief)

learn_logging	Fetches system logs (show logging last 250)

All inputs are validated using Pydantic schemas for safety and consistency.

🤖 LangGraph Integration

Add the MCP server as a tool node in your LangGraph pipeline like so:

```python

("pyats-mcp", ["python3", "pyats_mcp_server.py", "--oneshot"], "tools/discover", "tools/call")

```

Name: pyats-mcp

Command: python3 pyats_mcp_server.py --oneshot

Discover Method: tools/discover

Call Method: tools/call

STDIO-based communication ensures tight integration with LangGraph’s tool invocation model without opening HTTP ports or exposing REST endpoints.

📜 Example Requests

Discover Tools

```json

{
  "jsonrpc": "2.0",
  "id": 1,
  "method": "tools/discover"
}

```

Run Show Command

``` json

{
  "jsonrpc": "2.0",
  "id": 2,
  "method": "tools/call",
  "params": {
    "name": "run_show_command",
    "arguments": {
      "device_name": "router1",
      "command": "show ip interface brief"
    }
  }
}
``` 
🔒 Security Features

Input validation using Pydantic

Blocks unsafe commands like erase, reload, write

Prevents pipe/redirect abuse (e.g., | include, >, copy, etc.)

Gracefully handles parsing fallbacks and errors

📁 Project Structure

```graphql

.
├── pyats_mcp_server.py     # MCP server with JSON-RPC and pyATS integration
├── Dockerfile              # Docker container definition
├── testbed.yaml            # pyATS testbed (user-provided)
└── README.md               # This file

```

📥 MCP Server Config Example (pyATS MCP via Docker)

To run the pyATS MCP Server as a container with STDIO integration, configure your mcpServers like this:

``` json
{
  "mcpServers": {
    "pyats": {
      "command": "docker",
      "args": [
        "run",
        "-i",
        "--rm",
        "-e",
        "PYATS_TESTBED_PATH",
        "-v",
        "/absolute/path/to/testbed/folder:/app",
        "pyats-mcp-server"
      ],
      "env": {
        "PYATS_TESTBED_PATH": "/app/testbed.yaml"
      }
    }
  }
}

```

```json
{
  "servers": { 
    "pyats": {
      "type": "stdio",
      "command": "python3",
      "args": [
        "-u",
        "/Users/johncapobianco/pyATS_MCP/pyats_mcp_server.py"
      ],
      "env": {
        "PYATS_TESTBED_PATH": "/Users/johncapobianco/pyATS_MCP/testbed.yaml"
      }
  }
}
```
🧾 Explanation:
command: Uses Docker to launch the containerized pyATS MCP server

args:

-i: Keeps STDIN open for communication

--rm: Automatically removes the container after execution

-e: Injects the environment variable PYATS_TESTBED_PATH

-v: Mounts your local testbed directory into the container

pyats-mcp-server: Name of the Docker image

env:

Sets the path to the testbed file inside the container (/app/testbed.yaml)


✍️ Author

John Capobianco

Product Marketing Evangelist, Selector AI

Author, Automate Your Network

Let me know if you’d like to add:

A sample LangGraph graph config

Companion client script

CI/CD integration (e.g., GitHub Actions)

Happy to help!

# The testbed.yaml file works with the Cisco DevNet Cisco Modeling Labs (CML) Sandbox! 
