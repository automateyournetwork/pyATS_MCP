# Dynamic Testbeds for pyATS MCP: technical write-up

Audience: Itential solution engineers and platform owners who run, extend or demo the pyATS MCP server.

## The problem

pyATS is built around a static `testbed.yaml`. That works for a lab, but real estates keep their device truth somewhere else: Itential Inventory Manager, NetBox, Nautobot, Catalyst Center, a CMDB export. Hand-maintaining a YAML copy of that data means two sources of truth that drift apart, and an AI agent driving the network can't safely edit YAML on its own.

## What changed

The server can now **build, hold and use a pyATS testbed entirely in memory**, sourced from any inventory, with no `testbed.yaml` at all.

```
 source (Itential Inventory / NetBox / Nautobot / Catalyst Center / REST / csv|json|yaml|xml)
        |  fetch (paginated, allow-listed)
        v
 declarative field mapping  +  filters  +  OS normalization
        |
        v
 in-memory testbed dict  ---- loader.load(dict) ---->  every existing pyATS tool
        |
        +-- materialized as a 0600 temp file only when a subprocess needs a path
            (pyats run job, pyats clean, robot, blitz)
```

pyATS accepts a dict in `pyats.topology.loader.load()`, and `%ENV{}` substitution works from a dict, so nothing needs to touch disk for normal operation. The existing ~50 tools call one `_load_testbed()` function, which now returns the active dynamic testbed when one is activated. They needed no changes.

## Tools (8, 34 total)

| Tool | Purpose |
|------|---------|
| `pyats_testbed_sources` | Which sources are configured, which testbed is active |
| `pyats_testbed_source_inventories` | List Itential inventories with node counts |
| `pyats_testbed_build` | Fetch, map, filter, build. Supports `dry_run`, `activate`, `merge_with="static"` |
| `pyats_testbed_validate` | pyATS schema load, credential env check, optional TCP reachability |
| `pyats_testbed_activate` | Make all other tools use a testbed (`"static"` switches back) |
| `pyats_testbed_list` / `_get` / `_delete` | Registry management |

MCP conformance: every tool carries `readOnlyHint`, `destructiveHint`, `idempotentHint` and `openWorldHint` annotations. `build` and `validate` do network I/O and are registered as Tasks, so they run through the server's asynchronous Tasks support.

## Itential Inventory Manager as the default source

Inventory Manager (`/inventory_manager/v1`) is a first-class source:

- Auth: OAuth client credentials (`POST /oauth/token`) or local login, using the same `PLATFORM_URL`, `AUTH_METHOD`, `CLIENT_ID`, `CLIENT_SECRET` variables the other Itential tooling already uses. A 401 triggers one re-authentication.
- Node mapping: `itential_host` to `ip`, `itential_platform` to `os`, `itential_port` to `port`, node `tags` to device tags. A `/24` suffix on a host is stripped.
- Pagination: pages are walked until `totalPages`. 1,340 nodes took about 2 seconds.
- Multiple Platforms: named profiles, for example `ITENTIAL_DEVSTACK_URL` next to the default. Use `profile="devstack"` on the build call.

Real inventories are often incomplete. In the SE Platform, the 1,340-node `ncdc03` inventory has an empty `itential_platform` on every node. The build reports that clearly and you resolve it with `default_os` or `os_map`.

## Declarative field mapping

A mapping tells the builder where each device attribute lives in a source record. Presets exist for `itential_inventory`, `netbox`, `nautobot`, `catalyst_center` and `generic`. Any other source needs only a mapping:

```json
{"source": "inline", "format": "csv",
 "data": "host,addr,nos\nr1,10.0.0.1,ios-xe\n",
 "mapping": {"name": "host", "ip": "addr", "os": "nos"}}
```

Expression types: dotted paths with `[*]`, `{template}` strings, `literal:` values, and `{path, map, regex, default}` objects. `os` values are normalized to pyATS names (`IOS-XE` to `iosxe`, `frr` to `linux`, and so on).

## Security model

This is the part to read before enabling it in a shared environment.

1. **Secrets never travel through the model.** Tool arguments and results are visible to the model, so keeping secrets "in memory" is not enough. Devices get `%ENV{PYATS_DEVICE_USERNAME}` style references; the server resolves them from its own environment. Inline secrets are rejected, and credential-shaped values are redacted from previews.
2. **Source allow-lists.** REST sources must be on `PYATS_MCP_TESTBED_SOURCE_HOSTS` (the Platform host is added automatically). File sources must be under `PYATS_MCP_TESTBED_SOURCE_DIRS`. Pagination never follows a `next` link to a different host.
3. **Existing guardrails still apply.** Dangerous-command blocking and the confirmation phrases are in the tool layer, so dynamic testbeds get them for free.
4. **Subprocess files** are written `0600` in the artifacts directory and only when needed.
5. Activation clears the connection cache so a stale session from the previous testbed is never reused.

## Live verification (2026-10-28)

All run through the real MCP endpoint (Streamable HTTP) against the rebuilt Docker image.

| Check | Result |
|-------|--------|
| Tool listing | 34 tools, new ones annotated |
| Itential SE Platform (cloud) | 12 inventories listed with node counts |
| `clab-net` (15 nodes) build and activate | 15 devices, `pyats_search_devices` found them with no change to that tool |
| `ncdc03` (1,340 nodes) with `default_os` and `tags=["firewall"]` | 4 devices |
| Catalyst Center (DevNet sandbox) | 4 devices built, `os=iosxe` |
| Inline JSON to a local SSH target, `validate` | pyATS schema ok, credentials resolved, TCP reachable |
| **Real device login** via `pyats_run_linux_command` on the dynamic testbed | `uname -a && whoami` returned the host kernel and `pyatstest` |
| Rejections | inline secret, non-allow-listed REST host, file outside allowed dirs |

Not verified live: the `clab-net` lab hosts (10.1.1.126) did not answer on the ATL VPN, so that inventory was verified to the build and schema stage but not to a login. The local devstack Platform was not running, so named profiles were covered by unit tests only. NetBox and Nautobot mappings follow the documented API shapes and were not run against a real instance.

## Operating it

```bash
# Server env (never in tool calls)
PLATFORM_URL=...  AUTH_METHOD=oauth  CLIENT_ID=...  CLIENT_SECRET=...
PYATS_DEVICE_USERNAME=...  PYATS_DEVICE_PASSWORD=...  PYATS_DEVICE_ENABLE_PASSWORD=...
# optional: start with no testbed.yaml at all
PYATS_MCP_DYNAMIC_ONLY=1
```

A typical agent flow:

1. `pyats_testbed_source_inventories` to see what exists.
2. `pyats_testbed_build(inventory="clab-net", dry_run=true)` to preview.
3. `pyats_testbed_build(name="clab", inventory="clab-net", activate=true)`.
4. `pyats_testbed_validate(name="clab", check_reachability=true)`.
5. Use any pyATS tool as normal.

## Code map

- `pyats_testbed_builder.py`: pure logic (mapping, filters, clients, registry). Imports neither MCP nor pyATS, so it is cheap to unit test (43 tests).
- `pyats_mcp_server.py`: tool registration, `_load_testbed()` indirection, activation.
- `pyats_tasks.py`: `build` and `validate` registered as Tasks.
