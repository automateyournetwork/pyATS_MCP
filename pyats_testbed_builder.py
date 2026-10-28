"""
pyats_testbed_builder.py
========================
Build pyATS testbeds dynamically, in memory, from any inventory source.

Design
------
- A testbed is a plain pyATS testbed *dict* (the same shape as testbed.yaml).
  No YAML file is required: ``pyats.topology.loader.load(dict)`` accepts it.
  A file is only materialized lazily for subprocess-based tools
  (``pyats run job``, ``pyats clean``) that need a path.
- Sources (CSV / JSON / YAML / XML text or files, generic REST, and presets
  for Itential Inventory Manager, NetBox, Nautobot and Catalyst Center) are
  normalized to a list of flat-ish records.
- A declarative *mapping* turns each record into a device. Presets are just
  named mappings, so any other source is a mapping away.
- Secrets never travel through tool arguments or outputs: device credentials
  are ``%ENV{VAR}`` references, source-API tokens come from server-side env
  vars, and anything credential-shaped is redacted from previews.

This module deliberately imports neither MCP nor pyATS so it is unit-testable
on its own; pyats_mcp_server.py wires it into the tools.
"""

from __future__ import annotations

import csv
import io
import json
import os
import re
import socket
import threading
import time
import xml.etree.ElementTree as ET
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple
from urllib.parse import urlparse

import requests
import yaml

NAME_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,63}$")
ENV_NAME_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
_ENV_REF_RE = re.compile(r"%ENV\{([^}]+)\}")
_SECRET_KEY_RE = re.compile(r"pass|secret|token|key|credential|community|auth", re.I)

MAX_REGISTRY_ENTRIES = 50
DEFAULT_MAX_DEVICES = 5000
HTTP_TIMEOUT = 30


class TestbedBuildError(ValueError):
    """Raised for caller-fixable problems; the message is safe to show an agent."""

    __test__ = False  # not a pytest class


# ---------------------------------------------------------------------------
# Redaction
# ---------------------------------------------------------------------------
def redact(obj: Any) -> Any:
    """Return a copy of *obj* with credential-shaped values masked."""
    if isinstance(obj, dict):
        return {
            k: ("<redacted>" if isinstance(k, str) and _SECRET_KEY_RE.search(k) and v else redact(v))
            for k, v in obj.items()
        }
    if isinstance(obj, list):
        return [redact(v) for v in obj]
    return obj


# ---------------------------------------------------------------------------
# Path / expression resolution
# ---------------------------------------------------------------------------
_TOKEN_RE = re.compile(r"\[(\*|\d+)\]|([^.\[\]]+)")


def get_path(rec: Any, path: str) -> Any:
    """
    Resolve ``a.b[0].c`` or ``a[*].b`` against nested dicts/lists.
    ``[*]`` maps over a list and returns a list. Missing -> None.
    """
    cur: List[Any] = [rec]
    fanned = False
    for idx, key in _TOKEN_RE.findall(path):
        nxt: List[Any] = []
        for item in cur:
            if key:
                if isinstance(item, dict) and key in item:
                    nxt.append(item[key])
            elif idx == "*":
                if isinstance(item, list):
                    nxt.extend(item)
                    fanned = True
            else:
                i = int(idx)
                if isinstance(item, list) and -len(item) <= i < len(item):
                    nxt.append(item[i])
        cur = nxt
        if not cur:
            return None
    if fanned:
        return cur
    return cur[0] if cur else None


_TEMPLATE_RE = re.compile(r"\{([^{}]+)\}")


def resolve(rec: Dict[str, Any], spec: Any) -> Any:
    """
    Resolve a mapping expression against a record.

    - ``"a.b"``            dotted path
    - ``"literal:ios"``    constant
    - ``"{a}-{b.c}"``      template (missing parts -> empty string)
    - ``{"path"|"template"|"literal": ..., "map": {...}, "default": ...,
         "regex": "pattern-with-one-group"}``
    """
    if spec is None:
        return None
    if isinstance(spec, dict):
        if "literal" in spec:
            val: Any = spec["literal"]
        elif "template" in spec:
            val = resolve(rec, spec["template"] if "{" in spec["template"] else "literal:" + spec["template"])
        elif "path" in spec:
            val = get_path(rec, spec["path"])
        else:
            raise TestbedBuildError(f"mapping object needs path/template/literal: {spec}")
        if isinstance(val, list):
            val = val[0] if val else None
        if val is not None and spec.get("regex"):
            m = re.search(spec["regex"], str(val))
            val = (m.group(1) if m and m.groups() else (m.group(0) if m else None))
        if spec.get("map") and val is not None:
            mapped = spec["map"]
            val = mapped.get(val, mapped.get(str(val), mapped.get(str(val).lower(), val)))
        if val in (None, "") and "default" in spec:
            val = spec["default"]
        return val
    if not isinstance(spec, str):
        return spec
    if spec.startswith("literal:"):
        return spec[len("literal:"):]
    if "{" in spec:

        def sub(m: "re.Match[str]") -> str:
            v = get_path(rec, m.group(1).strip())
            if isinstance(v, list):
                v = v[0] if v else ""
            return "" if v is None else str(v)

        return _TEMPLATE_RE.sub(sub, spec)
    return get_path(rec, spec)


# ---------------------------------------------------------------------------
# OS normalization
# ---------------------------------------------------------------------------
_OS_ALIASES = {
    "ios": "ios",
    "ciscoios": "ios",
    "iosxe": "iosxe",
    "ciscoiosxe": "iosxe",
    "iosxr": "iosxr",
    "ciscoiosxr": "iosxr",
    "nxos": "nxos",
    "nexus": "nxos",
    "cisconxos": "nxos",
    "eos": "eos",
    "aristaeos": "eos",
    "junos": "junos",
    "juniperjunos": "junos",
    "asa": "asa",
    "ciscoasa": "asa",
    "linux": "linux",
    "ubuntu": "linux",
    "debian": "linux",
    "centos": "linux",
    "rhel": "linux",
    "frr": "linux",
    "sonic": "linux",
    "cumulus": "linux",
    "viptela": "viptela",
    "sdwan": "viptela",
    "aireos": "aireos",
    "wlc": "aireos",
    "fxos": "fxos",
    "ftd": "fxos",
    "iosxrv": "iosxr",
    "bigip": "bigip",
    "f5": "bigip",
    "fortios": "fortios",
}


def normalize_os(value: Any, extra: Optional[Dict[str, str]] = None) -> Tuple[Optional[str], bool]:
    """Return (pyats_os, recognized). Unknown values pass through lowercased."""
    if value in (None, ""):
        return None, False
    raw = str(value).strip()
    if extra:
        for k, v in extra.items():
            if k.lower() == raw.lower():
                return v, True
    key = re.sub(r"[^a-z0-9]", "", raw.lower())
    if key in _OS_ALIASES:
        return _OS_ALIASES[key], True
    return raw.lower(), False


# ---------------------------------------------------------------------------
# Presets
# ---------------------------------------------------------------------------
PRESETS: Dict[str, Dict[str, Any]] = {
    # Itential Inventory Manager node: {name, attributes:{itential_host,...}, tags}
    "itential_inventory": {
        "records_path": None,  # fetcher already returns node documents
        "mapping": {
            "name": "name",
            "ip": "attributes.itential_host",
            "os": "attributes.itential_platform",
            "port": "attributes.itential_port",
            "protocol": "literal:ssh",
            "custom": {
                "source": "literal:itential_inventory",
                "inventory": "inventory_name",
                "node_id": "_id",
                "tags": "tags",
                "cluster_id": "attributes.cluster_id",
            },
        },
    },
    "netbox": {
        "records_path": "results",
        "mapping": {
            "name": "name",
            "ip": "primary_ip4.address",
            "os": {"path": "platform.slug"},
            "type": "role.slug",
            "model": "device_type.model",
            "protocol": "literal:ssh",
            "custom": {
                "source": "literal:netbox",
                "site": "site.slug",
                "tags": "tags[*].slug",
                "netbox_id": "id",
            },
        },
    },
    "nautobot": {
        "records_path": "results",
        "mapping": {
            "name": "name",
            "ip": "primary_ip4.host",
            "os": {"path": "platform.network_driver", "default": None},
            "type": "role.name",
            "model": "device_type.model",
            "protocol": "literal:ssh",
            "custom": {
                "source": "literal:nautobot",
                "location": "location.name",
                "tags": "tags[*].name",
                "nautobot_id": "id",
            },
        },
    },
    "catalyst_center": {
        "records_path": "response",
        "mapping": {
            "name": "hostname",
            "ip": "managementIpAddress",
            "os": "softwareType",
            "platform": "platformId",
            "series": "series",
            "type": "family",
            "protocol": "literal:ssh",
            "custom": {
                "source": "literal:catalyst_center",
                "serial": "serialNumber",
                "software_version": "softwareVersion",
                "role": "role",
                "catalyst_id": "id",
            },
        },
        "filters": {"exclude": [{"field": "softwareType", "equals": "Cisco Controller"}]},
    },
    "generic": {"records_path": None, "mapping": {}},
}


# ---------------------------------------------------------------------------
# Parsing (csv / json / yaml / xml)
# ---------------------------------------------------------------------------
def _xml_to_obj(el: ET.Element) -> Any:
    children = list(el)
    if not children:
        return (el.text or "").strip() or dict(el.attrib) or None
    out: Dict[str, Any] = {f"@{k}": v for k, v in el.attrib.items()}
    for c in children:
        val = _xml_to_obj(c)
        tag = c.tag.split("}")[-1]
        if tag in out:
            if not isinstance(out[tag], list):
                out[tag] = [out[tag]]
            out[tag].append(val)
        else:
            out[tag] = val
    return out


def detect_format(text: str, hint: Optional[str] = None) -> str:
    if hint and hint.lower() not in ("auto", ""):
        return hint.lower().replace("yml", "yaml")
    s = text.lstrip()
    if s.startswith("<"):
        return "xml"
    if s[:1] in "[{":
        return "json"
    first = s.splitlines()[0] if s else ""
    if "," in first and ":" not in first.split(",")[0]:
        return "csv"
    return "yaml"


def parse_payload(text: str, fmt: Optional[str] = None) -> Any:
    """Parse csv/json/yaml/xml text into Python objects."""
    f = detect_format(text, fmt)
    try:
        if f == "json":
            return json.loads(text)
        if f == "yaml":
            return yaml.safe_load(text)
        if f == "csv":
            return list(csv.DictReader(io.StringIO(text)))
        if f == "xml":
            root = ET.fromstring(text)
            return {root.tag.split("}")[-1]: _xml_to_obj(root)}
    except Exception as exc:  # noqa: BLE001 - surface parse errors uniformly
        raise TestbedBuildError(f"could not parse input as {f}: {exc}") from exc
    raise TestbedBuildError(f"unsupported format '{f}' (use csv, json, yaml, xml)")


def extract_records(obj: Any, records_path: Optional[str]) -> List[Dict[str, Any]]:
    """Locate the device list inside a parsed document."""
    if records_path:
        obj = get_path(obj, records_path)
    elif isinstance(obj, dict):
        # Heuristic: a single top-level list of dicts, or a testbed-style dict.
        lists = [v for v in obj.values() if isinstance(v, list) and v and isinstance(v[0], dict)]
        if len(lists) == 1:
            obj = lists[0]
        elif "devices" in obj and isinstance(obj["devices"], dict):
            obj = [{"name": k, **v} for k, v in obj["devices"].items() if isinstance(v, dict)]
        else:
            obj = [obj]
    if isinstance(obj, dict):
        obj = [obj]
    if not isinstance(obj, list):
        raise TestbedBuildError(
            "no record list found; set records_path to the key holding the device list"
        )
    recs = [r for r in obj if isinstance(r, dict)]
    if not recs:
        raise TestbedBuildError("record list is empty or not made of objects")
    return recs


# ---------------------------------------------------------------------------
# Filters
# ---------------------------------------------------------------------------
def _match(rec: Dict[str, Any], rule: Dict[str, Any]) -> bool:
    val = get_path(rec, rule["field"]) if "field" in rule else None
    vals = val if isinstance(val, list) else [val]
    sv = [str(v) for v in vals if v is not None]
    if "equals" in rule:
        return str(rule["equals"]) in sv
    if "in" in rule:
        return any(s in {str(x) for x in rule["in"]} for s in sv)
    if "contains" in rule:
        return any(str(rule["contains"]).lower() in s.lower() for s in sv)
    if "regex" in rule:
        return any(re.search(rule["regex"], s) for s in sv)
    raise TestbedBuildError(f"filter rule needs equals/in/contains/regex: {rule}")


def apply_filters(records: List[Dict[str, Any]], filters: Optional[Dict[str, Any]]) -> List[Dict[str, Any]]:
    if not filters:
        return records
    inc, exc = filters.get("include") or [], filters.get("exclude") or []
    out = []
    for r in records:
        if inc and not all(_match(r, rule) for rule in inc):
            continue
        if exc and any(_match(r, rule) for rule in exc):
            continue
        out.append(r)
    return out


# ---------------------------------------------------------------------------
# Building
# ---------------------------------------------------------------------------
def _check_env_name(label: str, name: str) -> str:
    if not ENV_NAME_RE.match(name):
        raise TestbedBuildError(f"{label} must be an environment variable NAME (got {name!r}); never pass secrets inline")
    return name


def build_credentials(spec: Optional[Dict[str, Any]], static_raw: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """
    credentials spec -> testbed.credentials block made only of %ENV{} references.

    {"username_env": "X", "password_env": "Y", "enable_env": "Z"}
    {"from": "static"}   copy the static testbed's credentials block verbatim
    None                 PYATS_DEVICE_USERNAME / PYATS_DEVICE_PASSWORD /
                         PYATS_DEVICE_ENABLE_PASSWORD
    """
    spec = spec or {}
    if spec.get("from") == "static":
        creds = ((static_raw or {}).get("testbed") or {}).get("credentials")
        if not creds:
            raise TestbedBuildError("credentials.from='static' but the static testbed defines no testbed.credentials")
        return json.loads(json.dumps(creds))
    u = _check_env_name("username_env", spec.get("username_env", "PYATS_DEVICE_USERNAME"))
    p = _check_env_name("password_env", spec.get("password_env", "PYATS_DEVICE_PASSWORD"))
    e = _check_env_name("enable_env", spec.get("enable_env", "PYATS_DEVICE_ENABLE_PASSWORD"))
    return {
        "default": {"username": f"%ENV{{{u}}}", "password": f"%ENV{{{p}}}"},
        "enable": {"password": f"%ENV{{{e}}}"},
    }


def _clean_custom(rec: Dict[str, Any], spec: Dict[str, Any]) -> Dict[str, Any]:
    out = {}
    for k, expr in spec.items():
        v = resolve(rec, expr)
        if v not in (None, "", []):
            out[k] = v
    return redact(out)


def build_testbed(
    records: List[Dict[str, Any]],
    mapping: Dict[str, Any],
    *,
    name: str,
    credentials: Dict[str, Any],
    os_map: Optional[Dict[str, str]] = None,
    default_os: Optional[str] = None,
    default_protocol: str = "ssh",
    default_port: int = 22,
    max_devices: int = DEFAULT_MAX_DEVICES,
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Return (testbed_dict, report)."""
    if "name" not in mapping or "ip" not in mapping:
        raise TestbedBuildError("mapping must define at least 'name' and 'ip'")
    devices: Dict[str, Any] = {}
    skipped: List[Dict[str, Any]] = []
    warnings: List[str] = []
    os_counts: Dict[str, int] = {}
    unknown_os: set = set()

    for idx, rec in enumerate(records):
        dev_name = resolve(rec, mapping["name"])
        ip = resolve(rec, mapping["ip"])
        label = str(dev_name or f"record[{idx}]")
        if not dev_name:
            skipped.append({"record": label, "reason": "no name"})
            continue
        dev_name = str(dev_name).strip()
        if not ip:
            skipped.append({"record": label, "reason": "no management address"})
            continue
        ip = str(ip).split("/")[0].strip()
        if dev_name in devices:
            skipped.append({"record": label, "reason": "duplicate device name"})
            continue
        if len(devices) >= max_devices:
            warnings.append(f"stopped at max_devices={max_devices}; raise it or add filters")
            break

        os_val, known = normalize_os(resolve(rec, mapping.get("os")) or default_os, os_map)
        if os_val is None:
            skipped.append({"record": label, "reason": "no os/platform (set mapping.os, default_os or os_map)"})
            continue
        if not known:
            unknown_os.add(os_val)
        os_counts[os_val] = os_counts.get(os_val, 0) + 1

        try:
            port = int(resolve(rec, mapping.get("port")) or default_port)
        except (TypeError, ValueError):
            port = default_port
        protocol = str(resolve(rec, mapping.get("protocol")) or default_protocol).lower()

        dev: Dict[str, Any] = {
            "os": os_val,
            "type": str(resolve(rec, mapping.get("type")) or "network_device"),
            "connections": {
                "defaults": {"via": "cli"},
                "cli": {"protocol": protocol, "ip": ip, "port": port},
            },
        }
        for opt in ("platform", "model", "series", "alias"):
            v = resolve(rec, mapping.get(opt))
            if v not in (None, ""):
                dev[opt] = str(v)
        custom = _clean_custom(rec, mapping.get("custom") or {})
        if custom:
            dev["custom"] = custom
        devices[dev_name] = dev

    if unknown_os:
        warnings.append(
            f"os value(s) {sorted(unknown_os)} are not in the known alias table and were passed through "
            "unchanged; use os_map to translate them to pyATS os names"
        )
    if not devices:
        reasons: Dict[str, int] = {}
        for s in skipped:
            reasons[s["reason"]] = reasons.get(s["reason"], 0) + 1
        raise TestbedBuildError(f"no devices could be built from {len(records)} record(s); skip reasons: {reasons}")

    tb = {
        "testbed": {"name": name, "credentials": credentials},
        "devices": devices,
    }
    report = {
        "records_seen": len(records),
        "devices_built": len(devices),
        "devices_skipped": len(skipped),
        "skipped_sample": skipped[:10],
        "os_counts": os_counts,
        "warnings": warnings,
    }
    return tb, report


def merge_raw(base: Dict[str, Any], extra: Dict[str, Any], name: str) -> Tuple[Dict[str, Any], List[str]]:
    """Merge *extra* devices into a copy of *base*; base wins on name clashes."""
    out = json.loads(json.dumps(base))
    out.setdefault("devices", {})
    clashes = []
    for dname, dev in (extra.get("devices") or {}).items():
        if dname in out["devices"]:
            clashes.append(dname)
            continue
        out["devices"][dname] = dev
    out.setdefault("testbed", {})["name"] = name
    return out, clashes


# ---------------------------------------------------------------------------
# Fetchers
# ---------------------------------------------------------------------------
def _env(name: str) -> str:
    return os.getenv(name, "").strip()


def _hostname(url: str) -> str:
    return (urlparse(url).hostname or "").lower()


def allowed_hosts() -> set:
    """Hosts the generic REST fetcher may call: configured source URLs + explicit extras."""
    hosts = set()
    for k, v in os.environ.items():
        if v and (k in ("PLATFORM_URL", "NETBOX_URL", "NAUTOBOT_URL", "CATALYST_CENTER_URL")
                  or re.match(r"^(ITENTIAL_.+_URL|[A-Z0-9_]+_SOURCE_URL)$", k)):
            h = _hostname(v)
            if h:
                hosts.add(h)
    for h in _env("PYATS_MCP_TESTBED_SOURCE_HOSTS").split(","):
        if h.strip():
            hosts.add(h.strip().lower())
    return hosts


def allowed_dirs(extra: Iterable[Path] = ()) -> List[Path]:
    dirs = [Path(p).resolve() for p in extra]
    for p in _env("PYATS_MCP_TESTBED_SOURCE_DIRS").split(os.pathsep):
        if p.strip():
            dirs.append(Path(p).resolve())
    return dirs


def read_source_file(path: str, dirs: List[Path]) -> str:
    p = Path(path).expanduser().resolve()
    if not any(p == d or d in p.parents for d in dirs):
        raise TestbedBuildError(
            "file_path is outside the allowed source directories. Add its directory to "
            "PYATS_MCP_TESTBED_SOURCE_DIRS (os.pathsep-separated) on the server."
        )
    if not p.is_file():
        raise TestbedBuildError(f"file not found: {p.name}")
    if p.stat().st_size > 50 * 1024 * 1024:
        raise TestbedBuildError("source file exceeds 50MB")
    return p.read_text(encoding="utf-8")


def _verify(profile_prefix: str) -> bool:
    return _env(f"{profile_prefix}VERIFY_TLS") not in ("0", "false", "False")


def itential_config(profile: Optional[str]) -> Dict[str, Any]:
    """
    Env names for an Itential Platform. Default profile reuses the standard
    names (PLATFORM_URL, AUTH_METHOD, CLIENT_ID, CLIENT_SECRET,
    PLATFORM_USERNAME, PLATFORM_PASSWORD); a named profile (e.g. "devstack")
    reads ITENTIAL_DEVSTACK_URL, ITENTIAL_DEVSTACK_AUTH_METHOD, ... instead.
    """
    if profile:
        if not re.match(r"^[A-Za-z0-9_]{1,32}$", profile):
            raise TestbedBuildError("profile may contain letters, digits and underscore only")
        pre = f"ITENTIAL_{profile.upper()}_"
        keys = ("URL", "AUTH_METHOD", "CLIENT_ID", "CLIENT_SECRET", "USERNAME", "PASSWORD")
        cfg = {k: _env(pre + k) for k in keys}
        hint = f"ITENTIAL_{profile.upper()}_URL (+ _AUTH_METHOD, _CLIENT_ID/_CLIENT_SECRET or _USERNAME/_PASSWORD)"
    else:
        pre = "PLATFORM_"
        cfg = {
            "URL": _env("PLATFORM_URL"),
            "AUTH_METHOD": _env("AUTH_METHOD"),
            "CLIENT_ID": _env("CLIENT_ID"),
            "CLIENT_SECRET": _env("CLIENT_SECRET"),
            "USERNAME": _env("PLATFORM_USERNAME"),
            "PASSWORD": _env("PLATFORM_PASSWORD"),
        }
        hint = "PLATFORM_URL, AUTH_METHOD, CLIENT_ID/CLIENT_SECRET (or PLATFORM_USERNAME/PLATFORM_PASSWORD)"
    if not cfg["URL"]:
        raise TestbedBuildError(f"Itential Platform is not configured on the server: set {hint}")
    cfg["URL"] = cfg["URL"].rstrip("/")
    cfg["verify"] = _verify(pre if profile else "PLATFORM_")
    return cfg


class ItentialClient:
    """Minimal Itential Platform client for Inventory Manager (read-only)."""

    def __init__(self, profile: Optional[str] = None, session: Optional[requests.Session] = None):
        self.cfg = itential_config(profile)
        self.s = session or requests.Session()
        self.s.verify = self.cfg["verify"]
        self._headers: Dict[str, str] = {}
        self._params: Dict[str, str] = {}

    def _login(self) -> None:
        c, base = self.cfg, self.cfg["URL"]
        method = (c["AUTH_METHOD"] or ("oauth" if c["CLIENT_ID"] else "local")).lower()
        if method == "oauth":
            if not (c["CLIENT_ID"] and c["CLIENT_SECRET"]):
                raise TestbedBuildError("OAuth selected but CLIENT_ID/CLIENT_SECRET are not set on the server")
            r = self.s.post(
                f"{base}/oauth/token",
                data={"grant_type": "client_credentials", "client_id": c["CLIENT_ID"], "client_secret": c["CLIENT_SECRET"]},
                timeout=HTTP_TIMEOUT,
            )
            r.raise_for_status()
            self._headers = {"Authorization": f"Bearer {r.json()['access_token']}"}
        else:
            if not (c["USERNAME"] and c["PASSWORD"]):
                raise TestbedBuildError("local auth selected but username/password are not set on the server")
            r = self.s.post(
                f"{base}/login", json={"username": c["USERNAME"], "password": c["PASSWORD"]}, timeout=HTTP_TIMEOUT
            )
            r.raise_for_status()
            self._params = {"token": r.text.strip().strip('"')}

    def get(self, path: str, params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        if not self._headers and not self._params:
            self._login()
        q = {**(params or {}), **self._params}
        r = self.s.get(f"{self.cfg['URL']}{path}", headers=self._headers, params=q, timeout=HTTP_TIMEOUT)
        if r.status_code in (401, 403):  # token expired mid-run: re-auth once
            self._headers, self._params = {}, {}
            self._login()
            q = {**(params or {}), **self._params}
            r = self.s.get(f"{self.cfg['URL']}{path}", headers=self._headers, params=q, timeout=HTTP_TIMEOUT)
        r.raise_for_status()
        body = r.json()
        return body.get("result", body) if isinstance(body, dict) else body

    def list_inventories(self) -> List[Dict[str, Any]]:
        out, page = [], 1
        while True:
            res = self.get("/inventory_manager/v1/inventories", {"page": page, "pageSize": 100})
            out.extend(res.get("data", []))
            if page >= int(res.get("totalPages") or 1):
                return out
            page += 1

    def nodes(self, inventory: str, max_nodes: int) -> List[Dict[str, Any]]:
        out, page = [], 1
        while True:
            res = self.get(
                f"/inventory_manager/v1/inventories/{requests.utils.quote(inventory, safe='')}/nodes",
                {"page": page, "pageSize": 100},
            )
            for n in res.get("data", []):
                n["inventory_name"] = inventory
                out.append(n)
            if page >= int(res.get("totalPages") or 1) or len(out) >= max_nodes:
                return out[:max_nodes]
            page += 1


def fetch_itential(
    inventories: List[str], profile: Optional[str], max_nodes: int, client: Optional[ItentialClient] = None
) -> List[Dict[str, Any]]:
    cl = client or ItentialClient(profile)
    available = None
    recs: List[Dict[str, Any]] = []
    for inv in inventories:
        try:
            recs.extend(cl.nodes(inv, max_nodes - len(recs)))
        except requests.HTTPError as exc:
            if exc.response is not None and exc.response.status_code == 404:
                available = available or [i["name"] for i in cl.list_inventories()]
                raise TestbedBuildError(f"inventory {inv!r} not found; available: {available}") from exc
            raise
        if len(recs) >= max_nodes:
            break
    return recs


def _bearer_headers(spec: Dict[str, str]) -> Dict[str, str]:
    """headers_env maps header name -> ENV VAR NAME holding the full header value."""
    out = {}
    for h, var in (spec or {}).items():
        _check_env_name(f"headers_env[{h}]", var)
        val = _env(var)
        if not val:
            raise TestbedBuildError(f"environment variable {var} (for header {h}) is not set on the server")
        out[h] = val
    return out


def fetch_rest(
    url: str,
    *,
    params: Optional[Dict[str, Any]] = None,
    headers_env: Optional[Dict[str, str]] = None,
    pagination: Optional[Dict[str, Any]] = None,
    records_path: Optional[str] = None,
    max_records: int = DEFAULT_MAX_DEVICES,
    verify_tls: bool = True,
    session: Optional[requests.Session] = None,
) -> List[Dict[str, Any]]:
    """
    GET a JSON REST endpoint with optional pagination.

    pagination: {"type": "next_url", "path": "next"}                       (NetBox/Nautobot/DRF)
                {"type": "offset", "offset_param": "offset", "limit_param": "limit", "size": 500, "start": 1}
                {"type": "page", "page_param": "page", "size_param": "pageSize", "size": 100, "total_pages_path": "totalPages"}
    """
    host = _hostname(url)
    if urlparse(url).scheme not in ("http", "https") or not host:
        raise TestbedBuildError("url must be an absolute http(s) URL")
    if host not in allowed_hosts():
        raise TestbedBuildError(
            f"host {host!r} is not an allowed source. Allowed: {sorted(allowed_hosts())}. "
            "Add it to PYATS_MCP_TESTBED_SOURCE_HOSTS on the server."
        )
    s = session or requests.Session()
    s.verify = verify_tls
    headers = _bearer_headers(headers_env or {})
    pg = pagination or {}
    ptype = pg.get("type")
    recs: List[Dict[str, Any]] = []
    q = dict(params or {})
    cur_url, page, offset = url, 1, int(pg.get("start", 0))
    for _ in range(10_000):
        qq = dict(q)
        if ptype == "offset":
            qq[pg.get("limit_param", "limit")] = pg.get("size", 500)
            qq[pg.get("offset_param", "offset")] = offset
        elif ptype == "page":
            qq[pg.get("page_param", "page")] = page
            qq[pg.get("size_param", "pageSize")] = pg.get("size", 100)
        r = s.get(cur_url, headers=headers, params=qq if cur_url == url else None, timeout=HTTP_TIMEOUT)
        r.raise_for_status()
        body = r.json()
        chunk = extract_records(body, records_path) if (records_path or isinstance(body, (list, dict))) else []
        recs.extend(chunk)
        if len(recs) >= max_records:
            return recs[:max_records]
        if ptype == "next_url":
            nxt = get_path(body, pg.get("path", "next"))
            if not nxt or _hostname(nxt) != host:
                return recs
            cur_url = nxt
        elif ptype == "offset":
            if len(chunk) < int(pg.get("size", 500)):
                return recs
            offset += int(pg.get("size", 500)) if pg.get("start", 0) == 0 else len(chunk)
        elif ptype == "page":
            tp = get_path(body, pg["total_pages_path"]) if pg.get("total_pages_path") else None
            if (tp is not None and page >= int(tp)) or not chunk:
                return recs
            page += 1
        else:
            return recs
    return recs


def fetch_catalyst_center(max_records: int, session: Optional[requests.Session] = None) -> List[Dict[str, Any]]:
    base = _env("CATALYST_CENTER_URL").rstrip("/")
    user, pw = _env("CATALYST_CENTER_USERNAME"), _env("CATALYST_CENTER_PASSWORD")
    if not (base and user and pw):
        raise TestbedBuildError("set CATALYST_CENTER_URL, CATALYST_CENTER_USERNAME, CATALYST_CENTER_PASSWORD on the server")
    s = session or requests.Session()
    verify = _env("CATALYST_CENTER_VERIFY_TLS") or _env("CATALYST_CENTER_VERIFY_SSL")
    s.verify = verify not in ("0", "false", "False")
    r = s.post(f"{base}/dna/system/api/v1/auth/token", auth=(user, pw), timeout=HTTP_TIMEOUT)
    r.raise_for_status()
    hdr = {"X-Auth-Token": r.json()["Token"]}
    recs, offset, size = [], 1, 500
    while len(recs) < max_records:
        r = s.get(f"{base}/dna/intent/api/v1/network-device", headers=hdr, params={"offset": offset, "limit": size}, timeout=HTTP_TIMEOUT)
        r.raise_for_status()
        chunk = r.json().get("response", [])
        recs.extend(chunk)
        if len(chunk) < size:
            break
        offset += len(chunk)
    return recs[:max_records]


def fetch_token_api(kind: str, max_records: int, session: Optional[requests.Session] = None) -> List[Dict[str, Any]]:
    """NetBox / Nautobot: ``Authorization: Token <NETBOX_TOKEN>`` against /api/dcim/devices/."""
    up = kind.upper()
    base, token = _env(f"{up}_URL").rstrip("/"), _env(f"{up}_TOKEN")
    if not (base and token):
        raise TestbedBuildError(f"set {up}_URL and {up}_TOKEN on the server")
    s = session or requests.Session()
    s.verify = _env(f"{up}_VERIFY_TLS") not in ("0", "false", "False")
    url: Optional[str] = f"{base}/api/dcim/devices/"
    params: Optional[Dict[str, Any]] = {"limit": 200, "status": "active"} if kind == "netbox" else {"limit": 200, "depth": 1}
    recs: List[Dict[str, Any]] = []
    while url and len(recs) < max_records:
        r = s.get(url, headers={"Authorization": f"Token {token}"}, params=params, timeout=HTTP_TIMEOUT)
        r.raise_for_status()
        body = r.json()
        recs.extend(body.get("results", []))
        nxt = body.get("next")
        url = nxt if nxt and _hostname(nxt) == _hostname(base) else None
        params = None
    return recs[:max_records]


# ---------------------------------------------------------------------------
# Registry of in-memory testbeds
# ---------------------------------------------------------------------------
class Entry:
    def __init__(self, name: str, raw: Dict[str, Any], source: Dict[str, Any]):
        self.name = name
        self.raw = raw
        self.source = source
        self.created = time.time()
        self.version = f"{int(self.created * 1000):x}"
        self.loaded: Any = None  # pyATS Testbed object, set lazily by the server
        self.file: Optional[Path] = None


class TestbedRegistry:
    """Thread-safe named in-memory testbeds plus the currently active one."""

    __test__ = False

    def __init__(self, files_dir: Path):
        self._lock = threading.Lock()
        self._entries: Dict[str, Entry] = {}
        self._active: Optional[str] = None
        self._dir = files_dir

    def put(self, name: str, raw: Dict[str, Any], source: Dict[str, Any]) -> Entry:
        if not NAME_RE.match(name):
            raise TestbedBuildError("name must match [A-Za-z0-9][A-Za-z0-9_.-]{0,63}")
        with self._lock:
            if name not in self._entries and len(self._entries) >= MAX_REGISTRY_ENTRIES:
                raise TestbedBuildError(f"registry full ({MAX_REGISTRY_ENTRIES}); delete a testbed first")
            old = self._entries.get(name)
            if old and old.file:
                old.file.unlink(missing_ok=True)
            e = self._entries[name] = Entry(name, raw, source)
            return e

    def get(self, name: str) -> Entry:
        with self._lock:
            if name not in self._entries:
                raise TestbedBuildError(f"no dynamic testbed named {name!r}; have {sorted(self._entries)}")
            return self._entries[name]

    def names(self) -> List[str]:
        with self._lock:
            return sorted(self._entries)

    def delete(self, name: str) -> None:
        with self._lock:
            e = self._entries.pop(name, None)
            if e is None:
                raise TestbedBuildError(f"no dynamic testbed named {name!r}")
            if e.file:
                e.file.unlink(missing_ok=True)
            if self._active == name:
                self._active = None

    def activate(self, name: Optional[str]) -> None:
        with self._lock:
            if name is not None and name not in self._entries:
                raise TestbedBuildError(f"no dynamic testbed named {name!r}; have {sorted(self._entries)}")
            self._active = name

    def active_entry(self) -> Optional[Entry]:
        with self._lock:
            return self._entries.get(self._active) if self._active else None

    @property
    def active_name(self) -> Optional[str]:
        with self._lock:
            return self._active

    def materialize(self, entry: Entry) -> str:
        """Write the (secret-free) testbed to a 0600 file for subprocess tools."""
        with self._lock:
            if entry.file and entry.file.exists():
                return str(entry.file)
            self._dir.mkdir(parents=True, exist_ok=True)
            path = self._dir / f"{entry.name}.{entry.version}.yaml"
            fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                yaml.safe_dump(entry.raw, f, sort_keys=False)
            entry.file = path
            return str(path)


# ---------------------------------------------------------------------------
# Validation helpers
# ---------------------------------------------------------------------------
def env_refs(raw: Dict[str, Any]) -> List[str]:
    return sorted(set(_ENV_REF_RE.findall(json.dumps(raw))))


def unresolved_env(raw: Dict[str, Any]) -> List[str]:
    return [v for v in env_refs(raw) if not os.getenv(v)]


def tcp_probe(raw: Dict[str, Any], names: Optional[List[str]] = None, timeout: float = 3.0, workers: int = 32) -> Dict[str, Any]:
    """Parallel TCP connect test to each device's default connection (no login)."""
    targets: Dict[str, Tuple[str, int]] = {}
    for dname, dev in (raw.get("devices") or {}).items():
        if names and dname not in names:
            continue
        conns = dev.get("connections") or {}
        via = (conns.get("defaults") or {}).get("via") or next((k for k in conns if k != "defaults"), None)
        c = conns.get(via) or {}
        host = c.get("ip") or c.get("host")
        if host and not str(host).startswith("%"):
            targets[dname] = (str(host), int(c.get("port") or 22))

    def probe(item: Tuple[str, Tuple[str, int]]) -> Tuple[str, str]:
        n, (h, p) = item
        try:
            with socket.create_connection((h, p), timeout=timeout):
                return n, "reachable"
        except OSError as exc:
            return n, f"unreachable ({exc.__class__.__name__})"

    with ThreadPoolExecutor(max_workers=workers) as ex:
        results = dict(ex.map(probe, targets.items()))
    ok = sorted(n for n, s in results.items() if s == "reachable")
    bad = {n: s for n, s in results.items() if s != "reachable"}
    return {"checked": len(results), "reachable": len(ok), "unreachable": bad if len(bad) <= 50 else dict(list(bad.items())[:50])}
